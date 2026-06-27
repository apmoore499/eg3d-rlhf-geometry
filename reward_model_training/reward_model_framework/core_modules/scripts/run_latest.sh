#!/bin/bash 


# python train_rwd_model.py experiment=aw98_2d_lmks.yaml dloader.num_workers=8 trainer.max_epochs=30 model.act_type=cos callbacks.early_stopping.patience=10 





# python train_rwd_model.py experiment=triple_depth_map_resnet_50.yaml dloader.num_workers=8 trainer.max_epochs=30 model.act_type=cos callbacks.early_stopping.patience=10 





# python train_rwd_model.py experiment=triple_depth_map_resnet_50_notransform.yaml dloader.num_workers=8 trainer.max_epochs=30 model.act_type=cos callbacks.early_stopping.patience=10 




python train_rwd_model.py experiment=aw98_3d_kpts.yaml dloader.num_workers=8 trainer.max_epochs=30 model.act_type=cos callbacks.early_stopping.patience=10
