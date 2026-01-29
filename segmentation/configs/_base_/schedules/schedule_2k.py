# # optimizer
# optimizer = dict(type='SGD', lr=0.01, weight_decay=0.0005)
# optimizer_config = dict()
# # learning policy
# lr_config = dict(policy='poly', power=0.9, min_lr=1e-4, by_epoch=False)
# # runtime settings
# runner = dict(type='IterBasedRunner', max_iters=2000)
# checkpoint_config = dict(by_epoch=False, interval=200)
# evaluation = dict(interval=200, metric='mIoU')
#evaluation = dict(interval=200, metric='mIoU')


# optimizer
optimizer = dict(type='SGD', lr=0.01, weight_decay=0.0005)
optimizer_config = dict()
# learning policy
lr_config = dict(policy='poly', power=0.9, min_lr=1e-4, by_epoch=True)
# runtime settings

runner = dict(type='EpochBasedRunner', max_epochs=500)
# checkpoint and evaluation
checkpoint_config = dict(by_epoch=True, interval=20)
evaluation = dict(interval=20, metric=['mIoU', 'mDice', 'mFscore'], save_best='mIoU')