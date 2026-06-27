


# #-------------------------------------------------------------------
# #-------------------------------------------------------------------


#trainer.limit_train_batches=1 trainer.limit_test_batches=1 trainer.limit_val_batches=1 
# # aw98


experiment=conv3d_recreating_rwd_model_f12.yaml



# train_on=datamodule_first_binary
# cd /home/krillman/Documents/eg3dredo/supp_plus_code/eg3d_rlhf_code/reward_model_training/reward_model_framework/core_modules
# python train_rwd_model.py experiment=$experiment prop_data=1.0 model.compile=false train_on=$train_on model.optimizer.lr=5e-6 model.act_type=cos trainer.max_epochs=10 dloader.num_workers=6 dloader.prefetch_factor=2 batch_size_all=4 # dloader.prefetch_factor=2 batch_size_all=2


# train_on=datamodule_third_ranked
# cd /home/krillman/Documents/eg3dredo/supp_plus_code/eg3d_rlhf_code/reward_model_training/reward_model_framework/core_modules
# python train_rwd_model.py experiment=$experiment prop_data=1.0 model.compile=false train_on=$train_on model.optimizer.lr=5e-6 model.act_type=cos trainer.max_epochs=10 dloader.num_workers=6 dloader.prefetch_factor=2 batch_size_all=2


#batch_augmentations=default #no batch augmentations like good/bat/whatever. 
#batch_augmentations=@data/batch_augmentations/aug_sigma_field_batch_aug_all.yaml #all batch augmentations like good/bat/whatever, keep all #taking 40 min for 1 epoch  			 76pc acc, 0.91 reco                                  
#batch_augmentations=@data/batch_augmentations/default.yaml #augment none in batch. #taking 22 min for 1 epoch 																		 77pc acc, 0.9reco



#batch_augmentations=@data/batch_augmentations/no_aug_drop_pair.yaml #################	cjowd93t									 12 min	(newer drop pair selection)						 82.3pc acc, 0.54 reco
#batch_augmentations=@data/batch_augmentations/no_aug_drop_pair.yaml #################	8m4igjv6									 12 min	(older drop pair selection)						 82.3pc acc, 0.55 reco
# ^^^^ this is wrong, need to retry with older pair selection

batch_augmentations=@data/batch_augmentations/no_aug_drop_pair.yaml #################	cjowd93t									 12 min	(newer drop pair selection)						 82.3pc acc, 0.54 reco


#batch_augmentations=@data/batch_augmentations/aug_sigma_field_batch_pair_seed.yaml #convert batch to a pair and augment one of them 0.5p #taking 20 min for 1 epoch  				 83.7pc acc, 0.64 reco
#batch_augmentations=@data/batch_augmentations/aug_sigma_field_batch.yaml #augment best in batch to be good, do not remove any from batch #30 min for 1 epoch                        77?pc acc, 1.0 reco


#batch_augmentations=@aug_sigma_field_batch_aug_all #all batch augmentations like good/bat/whatever, keep all
#batch_augmentations=aug_sigma_field_batch_pair_seed #convert batch to a pair and augment one of them 0.5p
#batch_augmentations=aug_sigma_field_batch #augment best in batch to be good, do not remove any from batch


train_on=datamodule_third

ltb=1.0

cd /home/krillman/Documents/eg3dredo/supp_plus_code/eg3d_rlhf_code/reward_model_training/reward_model_framework/core_modules



#ckpt_path

ckpt_path=/home/krillman/Documents/eg3dredo/supp_plus_code/eg3d_rlhf_code/reward_model_training/reward_model_framework/core_modules/RWD_MODELS_FOR_TUNING/f12n5jti/orig_trial/best_model_orige.pt

# model.optimizer.lr=5e-6

python train_rwd_model_bk.py ckpt_path=$ckpt_path experiment=$experiment ++data.batch_augmentations=$batch_augmentations trainer.limit_val_batches=1 trainer.limit_train_batches=1 trainer.limit_test_batches=1.0 model.compile=false train_on=$train_on model.act_type=softplus trainer.max_epochs=1 dloader.num_workers=10 dloader.prefetch_factor=2 batch_size_all=1


