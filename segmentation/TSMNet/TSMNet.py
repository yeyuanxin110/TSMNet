import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.core import add_prefix
from mmseg.ops import resize
from mmseg.models import builder
from mmseg.models.builder import SEGMENTORS
from mmseg.models.segmentors.base import BaseSegmentor
import numpy as np
from .untils import tokenize
from timm.models.layers import trunc_normal_
import math

# Feature Rectify Module
class ChannelWeights(nn.Module):
    def __init__(self, dim, reduction=1):
        super(ChannelWeights, self).__init__()
        self.dim = dim
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(self.dim * 4, self.dim * 4 // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(self.dim * 4 // reduction, self.dim * 2),
            nn.Sigmoid())

    def forward(self, x1, x2):
        B, _, H, W = x1.shape
        x = torch.cat((x1, x2), dim=1)
        avg = self.avg_pool(x).view(B, self.dim * 2)
        max = self.max_pool(x).view(B, self.dim * 2)
        y = torch.cat((avg, max), dim=1)  # B 4C
        y = self.mlp(y).view(B, self.dim * 2, 1)
        channel_weights = y.reshape(B, 2, self.dim, 1, 1).permute(1, 0, 2, 3, 4)  # 2 B C 1 1
        return channel_weights


class SpatialWeights(nn.Module):
    def __init__(self, dim, reduction=1):
        super(SpatialWeights, self).__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Conv2d(self.dim * 2, self.dim // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.dim // reduction, 2, kernel_size=1),
            nn.Sigmoid())

    def forward(self, x1, x2):
        B, _, H, W = x1.shape
        x = torch.cat((x1, x2), dim=1)  # B 2C H W
        spatial_weights = self.mlp(x).reshape(B, 2, 1, H, W).permute(1, 0, 2, 3, 4)  # 2 B 1 H W
        return spatial_weights


class FeatureRectifyModule(nn.Module):
    def __init__(self, dim, reduction=1, lambda_c=.5, lambda_s=.5):
        super(FeatureRectifyModule, self).__init__()
        self.lambda_c = lambda_c
        self.lambda_s = lambda_s
        self.channel_weights = ChannelWeights(dim=dim, reduction=reduction)
        self.spatial_weights = SpatialWeights(dim=dim, reduction=reduction)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x1, x2):
        channel_weights = self.channel_weights(x1, x2)
        out1 = x1 + self.lambda_c * channel_weights[1] * x2
        out2 = x2 + self.lambda_c * channel_weights[0] * x1
        spatial_weights = self.spatial_weights(out1, out2)
        out_x1 = out1 + self.lambda_s * spatial_weights[1] * x2
        out_x2 = out2 + self.lambda_s * spatial_weights[0] * x1

        # out_x1 = x1 + self.lambda_c * channel_weights[1] * x2 + self.lambda_s * spatial_weights[1] * x2
        # out_x2 = x2 + self.lambda_c * channel_weights[0] * x1 + self.lambda_s * spatial_weights[0] * x1
        return out_x1, out_x2


# Stage 1
class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None):
        super(CrossAttention, self).__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.kv1 = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.kv2 = nn.Linear(dim, dim * 2, bias=qkv_bias)

    def forward(self, x1, x2):
        B, N, C = x1.shape
        q1 = x1.reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        q2 = x2.reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        k1, v1 = self.kv1(x1).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        k2, v2 = self.kv2(x2).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()

        ctx1 = (k1.transpose(-2, -1) @ v1) * self.scale
        ctx1 = ctx1.softmax(dim=-2)
        ctx2 = (k2.transpose(-2, -1) @ v2) * self.scale
        ctx2 = ctx2.softmax(dim=-2)

        x1 = (q1 @ ctx2).permute(0, 2, 1, 3).reshape(B, N, C).contiguous()
        x2 = (q2 @ ctx1).permute(0, 2, 1, 3).reshape(B, N, C).contiguous()

        return x1, x2


class CrossPath(nn.Module):
    def __init__(self, dim, reduction=1, num_heads=None, norm_layer=nn.LayerNorm):
        super().__init__()
        self.channel_proj1 = nn.Linear(dim, dim // reduction * 2)
        self.channel_proj2 = nn.Linear(dim, dim // reduction * 2)
        self.act1 = nn.ReLU(inplace=True)
        self.act2 = nn.ReLU(inplace=True)
        self.cross_attn = CrossAttention(dim // reduction, num_heads=num_heads)
        self.end_proj1 = nn.Linear(dim // reduction * 2, dim)
        self.end_proj2 = nn.Linear(dim // reduction * 2, dim)
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)

    def forward(self, x1, x2):
        y1, u1 = self.act1(self.channel_proj1(x1)).chunk(2, dim=-1)
        y2, u2 = self.act2(self.channel_proj2(x2)).chunk(2, dim=-1)
        v1, v2 = self.cross_attn(u1, u2)
        y1 = torch.cat((y1, v1), dim=-1)
        y2 = torch.cat((y2, v2), dim=-1)
        out_x1 = self.norm1(x1 + self.end_proj1(y1))
        out_x2 = self.norm2(x2 + self.end_proj2(y2))
        return out_x1, out_x2


# Stage 2
class ChannelEmbed(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=1, norm_layer=nn.BatchNorm2d):
        super(ChannelEmbed, self).__init__()
        self.out_channels = out_channels
        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.channel_embed = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // reduction, kernel_size=1, bias=True),
            nn.Conv2d(out_channels // reduction, out_channels // reduction, kernel_size=3, stride=1, padding=1,
                      bias=True, groups=out_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // reduction, out_channels, kernel_size=1, bias=True),
            norm_layer(out_channels)
        )
        self.norm = norm_layer(out_channels)

    def forward(self, x, H, W):
        B, N, _C = x.shape
        x = x.permute(0, 2, 1).reshape(B, _C, H, W).contiguous()
        residual = self.residual(x)
        # x = self.channel_embed(x)
        # out = self.norm(residual + x)
        return residual
class FeatureFusionModule(nn.Module):
    def __init__(self, dim, reduction=1, num_heads=None, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.cross = CrossPath(dim=dim, reduction=reduction, num_heads=num_heads)
        self.channel_emb = ChannelEmbed(in_channels=dim * 2, out_channels=dim, reduction=reduction,
                                        norm_layer=norm_layer)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x1, x2):
        B, C, H, W = x1.shape
        x1 = x1.flatten(2).transpose(1, 2)
        x2 = x2.flatten(2).transpose(1, 2)
        x1, x2 = self.cross(x1, x2)
        # merge=x1+x2
        merge = torch.cat((x1, x2), dim=-1)
        merge = self.channel_emb(merge, H, W)

        return merge
class ChannelConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ChannelConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)

    def forward(self, x):
        return self.conv(x)


class CrossModalAttention(nn.Module):
    def __init__(self, img_channels, text_channels=512):
        """
        Args:
            img_channels: 图像特征的通道数（如x[i]的通道数）
            text_channels: 文本特征的通道数（默认512）
        """
        super().__init__()
        # 将文本特征投影到与图像相同的通道空间
        self.text_proj = nn.Conv2d(text_channels, img_channels, kernel_size=1)
        # 注意力计算层
        self.query = nn.Conv2d(img_channels, img_channels, 1)  # 图像作为Query
        self.key = nn.Conv2d(img_channels, img_channels, 1)  # 文本作为Key
        self.value = nn.Conv2d(img_channels, img_channels, 1)  # 文本作为Value
        # 可学习的缩放因子
        self.scale = nn.Parameter(torch.tensor([0.1]))

    def forward(self, img_feat, text_feat):
        """
        Args:
            img_feat: 图像特征 [B, C_img, H, W]
            text_feat: 文本特征 [B, 512, H, W]
        Returns:
            融合后的特征 [B, C_img, H, W]
        """
        # 投影文本特征到图像通道空间
        text_feat = self.text_proj(text_feat)  # [B, C_img, H, W]

        # 计算Query/Key/Value
        Q = self.query(img_feat)  # [B, C_img, H, W]
        K = self.key(text_feat)  # [B, C_img, H, W]
        V = self.value(text_feat)  # [B, C_img, H, W]

        # 计算注意力权重 (空间维度加权)
        attn = torch.einsum("bchw,bcHW->bhwHW", Q, K)  # [B, H, W, H, W]
        attn = attn * self.scale  # 缩放
        attn = torch.softmax(attn.view(*attn.shape[:3], -1), dim=-1)  # 按空间位置归一化
        attn = attn.view(*attn.shape[:3], K.size(2), K.size(3))  # [B, H, W, H, W]

        # 应用注意力到Value
        out = torch.einsum("bhwHW,bcHW->bchw", attn, V)  # [B, C_img, H, W]

        # 残差连接
        return img_feat + out

@SEGMENTORS.register_module()
class TSMNet(BaseSegmentor):
    """Encoder Decoder segmentors.

    EncoderDecoder typically consists of backbone, decode_head, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be dumped during inference.
    """

    def __init__(self,
                 backbone,
                 text_encoder,
                 context_decoder,
                 decode_head,
                 class_names,
                 context_length,
                 context_feature='attention',
                 score_concat_index=3,
                 text_head=False,
                 neck=None,
                 tau=0.07,
                 auxiliary_head=None,
                 identity_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None,
                 token_embed_dim=512, text_dim=1024,
                 norm_fuse=nn.BatchNorm2d,
                 **args):
        super(TSMNet, self).__init__(init_cfg)
        if pretrained is not None:
            assert backbone.get('pretrained') is None, \
                'both backbone and segmentor set pretrained weight'
            backbone.pretrained = pretrained

            assert text_encoder.get('pretrained') is None, \
                'both text encoder and segmentor set pretrained weight'
            
            if 'RN50' not in pretrained and 'RN101' not in pretrained and 'ViT-B' not in pretrained:
                print('not CLIP pre-trained weight, using CLIP ViT-B-16')
                text_encoder.pretrained = 'pretrained/ViT-B-16.pt'
            else:
                text_encoder.pretrained = pretrained
        self.channel_conv = ChannelConv(in_channels=1, out_channels=3)
        self.backbone = builder.build_backbone(backbone)
        self.backbone_sar= builder.build_backbone(backbone)
        self.text_encoder = builder.build_backbone(text_encoder)
        # 冻结参数
        # for net in [self.backbone, self.backbone_sar, self.text_encoder]:
        #     for param in net.parameters():
        #         param.requires_grad = False
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        self.context_decoder = builder.build_backbone(context_decoder)
        self.context_length = context_length
        self.score_concat_index = score_concat_index
        self.FRM = FeatureRectifyModule(dim=768, reduction=1)
        self.FFM = FeatureFusionModule(dim=768, reduction=1, num_heads=1, norm_layer=norm_fuse)
        self.conv1x11 = nn.Conv2d(512, 256, kernel_size=1)
        # self.FRM2 = FeatureRectifyModule(dim=256, reduction=1)
        # self.FFM2 = FeatureFusionModule(dim=256, reduction=1, num_heads=1, norm_layer=norm_fuse)

        assert context_feature in ['attention', 'backbone']
        self.context_feature = context_feature

        self.text_head = text_head
        self.tau = tau

        if neck is not None:
            self.neck = builder.build_neck(neck)

        self._init_decode_head(decode_head)
        self._init_auxiliary_head(auxiliary_head)

        self.with_identity_head = False
        self.identity_head = None
        self._init_identity_head(identity_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.texts = torch.cat([tokenize(c, context_length=self.context_length) for c in class_names])
        self.num_classes = len(self.texts)


        context_length = self.text_encoder.context_length - self.context_length
        self.contexts = nn.Parameter(torch.randn(1, context_length, token_embed_dim))
        nn.init.trunc_normal_(self.contexts)
        self.gamma = nn.Parameter(torch.ones(text_dim) * 1e-4)
        # 1x1卷积用于通道数对齐
        self.inner_blocks = nn.ModuleList()
        for in_channels in [256, 256, 256, 256]:
            self.inner_blocks.append(
                nn.Conv2d(in_channels, 256, kernel_size=1)
            )
        # 上采样方法
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.inner_blocks2 = nn.ModuleList()
        for in_channels in [768, 768, 768+512, 768]:
            self.inner_blocks2.append(
                nn.Conv2d(in_channels, 768, kernel_size=1)
            )
        # 上采样方法
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)


        self.Linear=nn.Sequential(
                        nn.Linear(512, 512),
                        nn.GELU(),
                        nn.Linear(512, 256)
            )
        #self.text_normalize=F.normalize(p=2,dim=-1)
        #self.image_normalize = F.normalize(p=2, dim=-1)
        #self.sim=torch.einsum('b c h w, b t c -> b t h w')
        #self.up= F.interpolate( size=(256, 256), mode='bilinear')
        self.up= nn.Upsample(size=(256, 256),mode='bilinear',align_corners=True)
        self.loss1=nn.CrossEntropyLoss()
        self.cls_seg = nn.Conv2d(256, 7, kernel_size=1)

        self.cross_attentions = None
        # 其他原有组件（如conv1x11等）
        self.conv1x11 = nn.Conv2d(512, 512, kernel_size=1)
        self.conv768_512 = nn.Conv2d(768, 512, kernel_size=1)
        self.up2 = nn.Upsample(size=(64, 64), mode='bilinear', align_corners=True)
        self.up3 = nn.Upsample(size=(32, 32), mode='bilinear', align_corners=True)
        self.up4 = nn.Upsample(size=(8, 8), mode='bilinear', align_corners=True)

        self.cross_attn = nn.MultiheadAttention(embed_dim=256, num_heads=8)
        self.proj = nn.Linear(256, 64 * 64)

        # 初始化投影头（如果未初始化）
        if not hasattr(self, 'projection_heads'):
            # 图像特征投影头（每个层级独立）
            self.img_projectors = nn.ModuleList([
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    nn.Flatten(),
                    nn.Linear(256, 256),
                    nn.BatchNorm1d(256),
                    nn.ReLU(inplace=True)
                ) for i in range(4)
            ])
            # 文本特征投影头
            self.text_projector = nn.Sequential(
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True)
            )
            # 温度参数（可学习）
            self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))



        assert self.with_decode_head

    def _init_decode_head(self, decode_head):
        """Initialize ``decode_head``"""
        self.decode_head = builder.build_head(decode_head)
        self.align_corners = self.decode_head.align_corners
        self.num_classes = self.decode_head.num_classes

    def _init_auxiliary_head(self, auxiliary_head):
        """Initialize ``auxiliary_head``"""
        if auxiliary_head is not None:
            if isinstance(auxiliary_head, list):
                self.auxiliary_head = nn.ModuleList()
                for head_cfg in auxiliary_head:
                    self.auxiliary_head.append(builder.build_head(head_cfg))
            else:
                self.auxiliary_head = builder.build_head(auxiliary_head)
    
    def _init_identity_head(self, identity_head):
        """Initialize ``auxiliary_head``"""
        if identity_head is not None:
            self.with_identity_head = True
            self.identity_head = builder.build_head(identity_head)
    def extract_feat1(self, img):
        """Extract features from images."""
        x = self.backbone(img)

        return x

    def extract_feat(self, img,img_sar):
        """Extract features from images."""
        x = self.backbone(img)
        x_sar=self.backbone_sar(img_sar)
        alpha = 0.9
        result = []
        x = list(x)
        x_sar = list(x_sar)
        for i in range(4):
             x[i], x_sar[i] = self.FRM(alpha * x[i], (1 - alpha) * x_sar[i])
             result.append(self.FFM(alpha*x[i],(1-alpha)*x_sar[i]))
        result.append(x[4])
        result = tuple(result)
        return result

    def ronghe(self,x,text_embeddings_gobal):
        B = text_embeddings_gobal.size(0)
        text_embeddings_gobal = text_embeddings_gobal.view(B, 512, 1, 1)
        text_embeddings_gobal=self.conv1x11(text_embeddings_gobal)
        text_embeddings_gobals=[]
        text_embeddings_gobal1 = text_embeddings_gobal.expand(-1, -1, 64, 64)
        text_embeddings_gobal2 = text_embeddings_gobal.expand(-1, -1, 32, 32)
        text_embeddings_gobal3 = text_embeddings_gobal.expand(-1, -1, 16, 16)
        text_embeddings_gobal4 = text_embeddings_gobal.expand(-1, -1, 8, 8)
        text_embeddings_gobals.append(text_embeddings_gobal1)
        text_embeddings_gobals.append(text_embeddings_gobal2)
        text_embeddings_gobals.append(text_embeddings_gobal3)
        text_embeddings_gobals.append(text_embeddings_gobal4)
        y=[]
        for i in range(4):
            m=x[i]+0.1*text_embeddings_gobals[i]
            y.append(m)



        return y

    def contrastive_learning(self, x, text_embeddings_global):
        """
        多尺度对比学习版本
        输入结构保持不变，但处理逻辑改为对比学习
        """
        # 图像特征处理 --------------------------------------------------
        img_embeddings = []
        for i in range(4):
            # 提取层级特征并投影
            feat = self.img_projectors[i](x[i])  # [B, 256]
            img_embeddings.append(feat)
        # 文本特征处理 -------------------------------------------------
        text_emb = self.text_projector(text_embeddings_global)  # [B, 256]
        # 多层级对比计算 -----------------------------------------------
        total_loss = 0
        temperature = torch.clamp(self.logit_scale.exp(), max=100)
        for img_emb in img_embeddings:
            # 归一化处理
            img_emb = F.normalize(img_emb, p=2, dim=-1)
            text_emb = F.normalize(text_emb, p=2, dim=-1)
            # 计算相似度矩阵
            logits = torch.matmul(img_emb, text_emb.t()) * temperature  # [B, B]
            # 构建标签（对角线为正样本）
            labels = torch.arange(logits.size(0), device=logits.device)
            # 对称对比损失
            loss_i = F.cross_entropy(logits, labels)
            loss_t = F.cross_entropy(logits.t(), labels)
            total_loss += (loss_i + loss_t) / 2
        # 返回平均损失（可根据需要调整加权方式）
        return total_loss / len(img_embeddings)
    def ronghe1(self, x, text_embeddings_gobal):
        if self.cross_attentions is None:
            self.cross_attentions = nn.ModuleList([
                CrossModalAttention(img_channels=x[i].size(1))  # 此时x已存在
                for i in range(4)
            ]).to(x[0].device)
        B = text_embeddings_gobal.size(0)
        # 文本特征处理（保持原有逻辑）
        text_embeddings_gobal = text_embeddings_gobal.view(B, 512, 1, 1)
        text_embeddings_gobal = self.conv1x11(text_embeddings_gobal)
        # 生成多尺度文本特征
        text_scales = []
        for size in [64, 32, 16, 8]:  # 对应不同层级的空间尺寸
            # 动态调整文本特征到目标尺寸
            text_feat = torch.repeat_interleave(text_embeddings_gobal, size, dim=2)
            text_feat = torch.repeat_interleave(text_feat, size, dim=3)
            text_scales.append(text_feat)
        # 跨模态注意力融合
        y = []
        for i in range(4):
            # 获取对应层级的特征
            img_feat = x[i]  # 图像特征 [B, C_img, H, W]
            text_feat = text_scales[i]  # 文本特征 [B, 512, H, W]
            # 应用跨模态注意力
            fused_feat = self.cross_attentions[i](img_feat, text_feat)
            y.append(fused_feat)
        return y


    def FPN(self, x):
        """
        Args:
            x (list[Tensor]): 输入特征列表，按从大到小的尺度排列(如[H/4, H/8, H/16, H/32])

        Returns:
            merged_feature (Tensor): 融合后的特征图，尺寸与最大输入特征相同
        """
        # 通道对齐
        inner_features = [block(feat) for block, feat in zip(self.inner_blocks, x)]
        # 逐步上采样并融合
        merged_feature = inner_features[0]  # 最大的特征图
        # 遍历后续特征进行上采样和融合
        for i in range(1, len(inner_features)):
            # 计算需要上采样的次数
            for _ in range(i):
                inner_features[i] = self.upsample(inner_features[i])
            # 特征相加融合
            merged_feature += inner_features[i]
        return merged_feature

    def FPN2(self, x):
        # 通道对齐
        inner_features = [block(feat) for block, feat in zip(self.inner_blocks2, x)]
        # 逐步上采样并融合
        merged_feature = inner_features[0]  # 最大的特征图
        # 遍历后续特征进行上采样和融合
        for i in range(1, len(inner_features)):
            # 计算需要上采样的次数
            for _ in range(i):
                inner_features[i] = self.upsample2(inner_features[i])
            # 特征相加融合
            merged_feature += inner_features[i]
        return merged_feature

    def _decode_head_forward_train(self, x, img_metas, gt_semantic_seg):
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()
        loss_decode = self.decode_head.forward_train(x, img_metas,
                                                     gt_semantic_seg,
                                                     self.train_cfg)
        losses.update(add_prefix(loss_decode, 'decode'))
        return losses

    def _decode_head_forward_test(self, x, img_metas):
        """Run forward function and calculate loss for decode head in
        inference."""
        seg_logits = self.decode_head.forward_test(x, img_metas, self.test_cfg)
        return seg_logits

    def _auxiliary_head_forward_train(self, x, img_metas, gt_semantic_seg):
        """Run forward function and calculate loss for auxiliary head in
        training."""
        losses = dict()
        if isinstance(self.auxiliary_head, nn.ModuleList):
            for idx, aux_head in enumerate(self.auxiliary_head):
                loss_aux = aux_head.forward_train(x, img_metas,
                                                  gt_semantic_seg,
                                                  self.train_cfg)
                losses.update(add_prefix(loss_aux, f'aux_{idx}'))
        else:
            loss_aux = self.auxiliary_head.forward_train(
                x, img_metas, gt_semantic_seg, self.train_cfg)
            losses.update(add_prefix(loss_aux, 'aux'))

        return losses

    def _identity_head_forward_train(self, x, img_metas, gt_semantic_seg):
        """Run forward function and calculate loss for auxiliary head in
        training."""
        losses = dict()
        loss_aux = self.identity_head.forward_train(
            x, img_metas, gt_semantic_seg, self.train_cfg)
        losses.update(add_prefix(loss_aux, 'aux_identity'))
        return losses

    def forward_dummy(self, img):
        """Dummy forward function."""
        seg_logit = self.encode_decode(img, None)

        return seg_logit

    def after_extract_feat(self, x,texts):
        x_orig = list(x[0:4])
        global_feat, visual_embeddings = x[4]
        # x_orig1=x_orig.copy()
        # x_orig1[self.score_concat_index] = torch.cat([x_orig[self.score_concat_index], visual_embeddings], dim=1)
        # x_orig2=self.FPN2(x_orig1)
        # visual_embeddings=self.conv768_512(x_orig2)

        B, C, H, W = visual_embeddings.shape
        if self.context_feature == 'attention':
            visual_context = torch.cat([global_feat.reshape(B, C, 1), visual_embeddings.reshape(B, C, H*W)], dim=2).permute(0, 2, 1)  # B, N, C
        text_embeddings_gobal=self.text_encoder(texts.to(global_feat.device), self.contexts).squeeze(0)
        # (B, K, C)
        text_embeddings = self.text_encoder(self.texts.to(global_feat.device), self.contexts).expand(B, -1, -1)
        # update text_embeddings by visual_context!
        # (B, 1, C)
        text_diff = self.context_decoder(text_embeddings, visual_context)
        # (B, K, C)
        text_embeddings = text_embeddings + self.gamma * text_diff

        # compute score map and concat
        B, K, C = text_embeddings.shape
        visual_embeddings = F.normalize(visual_embeddings, dim=1, p=2)
        text = F.normalize(text_embeddings, dim=2, p=2)
        score_map = torch.einsum('bchw,bkc->bkhw', visual_embeddings, text)
        # x_orig[0] = torch.cat([x_orig[0], score_map], dim=1)
        score_map2=self.up2(score_map)
        score_map3 = self.up3(score_map)
        score_map4 = self.up4(score_map)
        x_orig[self.score_concat_index] = torch.cat([x_orig[self.score_concat_index], score_map], dim=1)
        # x_orig[0] = torch.cat([x_orig[0], score_map2], dim=1)
        # x_orig[1] = torch.cat([x_orig[1], score_map3], dim=1)
        # x_orig[3] = torch.cat([x_orig[3], score_map4], dim=1)

        return text_embeddings, x_orig, score_map,text_embeddings_gobal

    def forward_train(self, img, img_metas, gt_semantic_seg):
        """Forward function for training.

        Args:
            img (Tensor): Input images.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        img_rgb = img[:, :3, :, :]  # 取前三个通道 (RGB)
        img_sar = img[:, 3:4, :, :]  # 取第四个通道 (SAR)
        # 将单波段图像复制成三波段图像 (SAR)
        #img_sar_3_channel = img_sar.expand(-1, 3, -1, -1)
        # img_sar_3_channel.zero_()
        img_sar_3_channel=self.channel_conv(img_sar)
        # 遍历 img_metas 并读取每个图像的 filename 参数
        # 用于存储所有 filename 的列表
        filenames = []
        file_contents=[]
        for i, meta in enumerate(img_metas):
            filename = meta.get('filename', 'default_filename.jpg')
            # 修改 filename
            # 1. 将文件路径中的扩展名改为 .txt
            # 2. 在目录结构中添加 txt 子目录
            import os
            # 分离目录和文件名
            dir_path, file_name = os.path.split(filename)
            # 获取文件名和扩展名
            name, ext = os.path.splitext(file_name)
            # 创建新的文件名，扩展名为 .txt
            new_file_name = name + '.txt'
            # 创建新的目录路径，添加 txt 子目录
            new_dir_path = os.path.join(dir_path, 'txt')
            # 组合新的文件路径
            new_filename = os.path.join(new_dir_path, new_file_name)
            # 打开文件并读取内容
            with open(new_filename, 'r', encoding='utf-8') as file:
                content = file.read()
                # 将文件内容添加到 file_contents 列表
                file_contents.append(content)
                filenames.append(new_filename)
        texts = torch.cat([tokenize(c, context_length=self.context_length) for c in file_contents])

        x = self.extract_feat(img_rgb,img_sar_3_channel)
        #x = self.extract_feat1(img_rgb)
        _x_orig = [x[i] for i in range(4)]
        text_embeddings, x_orig, score_map,text_embeddings_gobal = self.after_extract_feat(x,texts)

        if self.with_neck:
            x_orig = list(self.neck(x_orig))
            _x_orig = x_orig

        losses = dict()
        if self.text_head:
            x = [text_embeddings,] + x_orig
        else:
            x = x_orig

        loss_contrastive_learning=self.contrastive_learning(x,text_embeddings_gobal)
        x=self.ronghe1(x,text_embeddings_gobal)
        x1=self.FPN(x)#(12,256,64,64)
        # 1. 文本特征投影
        # text_proj = nn.Linear(512, 256)(text_embeddings)  # (12,8,256)
        # text_proj = F.normalize(text_proj, p=2, dim=-1)  # L2归一化
        text_proj=self.Linear(text_embeddings)
        text_proj = F.normalize(text_proj, p=2, dim=-1)  # L2归一化
        # text_proj=self.text_normalize(text_proj)
        # 2. 图像特征归一化
        image_norm = F.normalize(x1, p=2, dim=1)  # (12,256,64,64)
        #x1 = self.up(x1)
        # image_norm=self.image_normalize(x)
        # 3. 逐token计算相似性图
        #similarity = torch.einsum('b c h w, b t c -> b t h w', image_norm, text_proj)  # (12,8,64,64)

        # 调整 image_norm 的形状
        B, C, H, W = image_norm.shape
        image_flat = image_norm.view(B, C, H * W)  # (12, 256, 4096)
        image_norm_reshaped = image_flat.permute(2, 0, 1)  # (4096, 12, 256)
        # 调整 text_proj 的维度顺序
        text_proj_reshaped = text_proj.permute(1, 0, 2)  # (7, 12, 256)
        attn_output, _ = self.cross_attn(text_proj_reshaped, image_norm_reshaped, image_norm_reshaped)
        attn_output = attn_output.permute(1, 0, 2)
        attn_output_proj = self.proj(attn_output)  # (12, 7, 4096)
        attn_output_reshaped = attn_output_proj.view(10, 7, 64, 64)  # (12, 7, 64, 64)
        similarity = attn_output_reshaped  # 生成更丰富的对齐特征

        #similarity=self.sim(image_norm,text_proj)
        # 4. 上采样到256x256
        # upsampled = F.interpolate(similarity, size=(256, 256), mode='bilinear')  # (12,8,256,256)
        upsampled = self.up(similarity)
        #upsampled=self.up(x1)
        #upsampled=self.cls_seg(upsampled)
        # 5. 聚合多token预测（示例：取最大响应）
        #pred_mask = upsampled.max(dim=1, keepdim=True)[0]  # (12,1,256,256)
        # 6. 计算损失
        # loss_fn = nn.CrossEntropyLoss()
        # loss = loss_fn(pred_mask, gt_semantic_seg)
        gt_semantic_seg[gt_semantic_seg == 7] = 0
        loss=self.loss1(upsampled,gt_semantic_seg.squeeze(1))
        pred_mask = upsampled.argmax(dim=1)   # (12,1,256,256)
        correct = (pred_mask == gt_semantic_seg.squeeze(1)).float()  # 正确的位置为1.0，错误为0.0
        accuracy = correct.mean()  # 所有像素的平均正确率
        loss_dict = {
            'loss_ce': loss,  # CrossEntropy损失
            'loss_contrastive_learning': loss_contrastive_learning,  # InfoNCE损失
            'acc_seg': accuracy  # 分割准确率
        }
        loss2 = dict()
        loss2.update(add_prefix(loss_dict, 'decode_similarity'))
        losses.update(loss2)


        loss_decode = self._decode_head_forward_train(x, img_metas, gt_semantic_seg)
        losses.update(loss_decode)

        if self.with_identity_head:
            loss_identity = self._identity_head_forward_train(
                score_map/self.tau, img_metas, gt_semantic_seg)
            losses.update(loss_identity)

        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(
                _x_orig, img_metas, gt_semantic_seg)
            losses.update(loss_aux)

        # 假设权重字典已定义，例如：
        weights = {
            'decode_similarity.loss_ce': 0,
            'decode_similarity.loss_contrastive_learning': 0,
            'decode_similarity.acc_seg': 0,
            'decode.loss_ce': 1,
            'decode.acc_seg': 1,
            'aux_identity.loss_ce': 1,
            'aux_identity.acc_seg': 1,
        }
        # 计算加权总损失
        total_loss = 0.0
        for key in losses:
            loss10 = losses[key].squeeze()
            total_loss += loss10 * weights[key]
        # 创建仅包含总损失的新字典
        final_loss_dict = {'loss': total_loss}
        return losses

    def encode_decode(self, img, img_metas):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        img_rgb = img[:, :3, :, :]  # 取前三个通道 (RGB)
        img_sar = img[:, 3:4, :, :]  # 取第四个通道 (SAR)
        # 将单波段图像复制成三波段图像 (SAR)
        #img_sar_3_channel = img_sar.expand(-1, 3, -1, -1)
        img_sar_3_channel = self.channel_conv(img_sar)
        #img_sar_3_channel.zero_()
        filenames = []
        file_contents = []
        for i, meta in enumerate(img_metas):
            filename = meta.get('filename', 'default_filename.jpg')
            # 修改 filename
            # 1. 将文件路径中的扩展名改为 .txt
            # 2. 在目录结构中添加 txt 子目录
            import os
            # 分离目录和文件名
            dir_path, file_name = os.path.split(filename)
            # 获取文件名和扩展名
            name, ext = os.path.splitext(file_name)
            # 创建新的文件名，扩展名为 .txt
            new_file_name = name + '.txt'
            # 创建新的目录路径，添加 txt 子目录
            new_dir_path = os.path.join(dir_path, 'txt')
            # 组合新的文件路径
            new_filename = os.path.join(new_dir_path, new_file_name)
            # 打开文件并读取内容
            with open(new_filename, 'r', encoding='utf-8') as file:
                content = file.read()
                # 将文件内容添加到 file_contents 列表
                file_contents.append(content)
                filenames.append(new_filename)
        texts = torch.cat([tokenize(c, context_length=self.context_length) for c in file_contents])
        x = self.extract_feat(img_rgb,img_sar_3_channel)
        #x = self.extract_feat1(img_rgb)

        _x_orig = [x[i] for i in range(4)]
        text_embeddings, x_orig, score_map,text_embeddings_gobal = self.after_extract_feat(x,texts)

        if self.with_neck:
            x_orig = list(self.neck(x_orig))

        if self.text_head:
            x = [text_embeddings,] + x_orig
        else:
            x = x_orig
        #x = self.ronghe1(x, text_embeddings_gobal)
        # print('text_embedding=', text_embeddings[0])
        out = self._decode_head_forward_test(x, img_metas)
        # # print('cls_map=', out[0,:,40, 40])
        # x1 = self.FPN(x)
        # text_proj = self.Linear(text_embeddings)
        # text_proj = F.normalize(text_proj, p=2, dim=-1)  # L2归一化
        # #text_proj = self.Linear(text_proj)
        # image_norm = F.normalize(x1, p=2, dim=1)  # (12,256,64,64)
        # # 调整 image_norm 的形状
        # B, C, H, W = image_norm.shape
        # image_flat = image_norm.view(B, C, H * W)  # (12, 256, 4096)
        # image_norm_reshaped = image_flat.permute(2, 0, 1)  # (4096, 12, 256)
        # # 调整 text_proj 的维度顺序
        # text_proj_reshaped = text_proj.permute(1, 0, 2)  # (7, 12, 256)
        # attn_output, _ = self.cross_attn(text_proj_reshaped, image_norm_reshaped, image_norm_reshaped)
        # attn_output = attn_output.permute(1, 0, 2)
        # attn_output_proj = self.proj(attn_output)  # (12, 7, 4096)
        # attn_output_reshaped = attn_output_proj.view(1,7, 64, 64)  # (12, 7, 64, 64)
        # similarity = attn_output_reshaped  # 生成更丰富的对齐特征
        # #similarity = torch.einsum('b c h w, b t c -> b t h w', image_norm, text_proj)  # (12,8,64,64)
        # #upsampled = self.up(similarity)
        # #upsampled = self.up(x1)
        # #upsampled = self.cls_seg(upsampled)

        out = resize(
            input=out,
            size=img.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        return out


    # TODO refactor
    def slide_inference(self, img, img_meta, rescale):
        """Inference by sliding-window with overlap.

        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.
        """

        h_stride, w_stride = self.test_cfg.stride
        h_crop, w_crop = self.test_cfg.crop_size
        batch_size, _, h_img, w_img = img.size()
        num_classes = self.num_classes
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = img.new_zeros((batch_size, num_classes, h_img, w_img))
        count_mat = img.new_zeros((batch_size, 1, h_img, w_img))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = img[:, :, y1:y2, x1:x2]
                crop_seg_logit = self.encode_decode(crop_img, img_meta)
                preds += F.pad(crop_seg_logit,
                               (int(x1), int(preds.shape[3] - x2), int(y1),
                                int(preds.shape[2] - y2)))

                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0
        if torch.onnx.is_in_onnx_export():
            # cast count_mat to constant while exporting to ONNX
            count_mat = torch.from_numpy(
                count_mat.cpu().detach().numpy()).to(device=img.device)
        preds = preds / count_mat
        if rescale:
            preds = resize(
                preds,
                size=img_meta[0]['ori_shape'][:2],
                mode='bilinear',
                align_corners=self.align_corners,
                warning=False)
        return preds

    def whole_inference(self, img, img_meta, rescale):
        """Inference with full image."""

        seg_logit = self.encode_decode(img, img_meta)
        if rescale:
            # support dynamic shape for onnx
            if torch.onnx.is_in_onnx_export():
                size = img.shape[2:]
            else:
                size = img_meta[0]['ori_shape'][:2]
            seg_logit = resize(
                seg_logit,
                size=size,
                mode='bilinear',
                align_corners=self.align_corners,
                warning=False)
        
        if  torch.isnan(seg_logit).any():
            print('########### find NAN #############')

        return seg_logit

    def inference(self, img, img_meta, rescale):
        """Inference with slide/whole style.

        Args:
            img (Tensor): The input image of shape (N, 3, H, W).
            img_meta (dict): Image info dict where each dict has: 'img_shape',
                'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            rescale (bool): Whether rescale back to original shape.

        Returns:
            Tensor: The output segmentation map.
        """

        assert self.test_cfg.mode in ['slide', 'whole']
        ori_shape = img_meta[0]['ori_shape']
        assert all(_['ori_shape'] == ori_shape for _ in img_meta)
        if self.test_cfg.mode == 'slide':
            seg_logit = self.slide_inference(img, img_meta, rescale)
        else:
            seg_logit = self.whole_inference(img, img_meta, rescale)
        output = F.softmax(seg_logit, dim=1)
        flip = img_meta[0]['flip']
        if flip:
            flip_direction = img_meta[0]['flip_direction']
            assert flip_direction in ['horizontal', 'vertical']
            if flip_direction == 'horizontal':
                output = output.flip(dims=(3, ))
            elif flip_direction == 'vertical':
                output = output.flip(dims=(2, ))

        return output

    def simple_test(self, img, img_meta, rescale=True):
        """Simple test with single image."""
        seg_logit = self.inference(img, img_meta, rescale)
        seg_pred = seg_logit.argmax(dim=1)
        if torch.onnx.is_in_onnx_export():
            # our inference backend only support 4D output
            seg_pred = seg_pred.unsqueeze(0)
            return seg_pred
        seg_pred = seg_pred.cpu().numpy()
        # unravel batch dim
        seg_pred = list(seg_pred)
        return seg_pred

    def aug_test(self, imgs, img_metas, rescale=True):
        """Test with augmentations.

        Only rescale=True is supported.
        """
        # aug_test rescale all imgs back to ori_shape for now
        assert rescale
        # to save memory, we get augmented seg logit inplace
        seg_logit = self.inference(imgs[0], img_metas[0], rescale)
        for i in range(1, len(imgs)):
            cur_seg_logit = self.inference(imgs[i], img_metas[i], rescale)
            seg_logit += cur_seg_logit
        seg_logit /= len(imgs)
        seg_pred = seg_logit.argmax(dim=1)
        seg_pred = seg_pred.cpu().numpy()
        seg_pred = list(seg_pred)
        return seg_pred
