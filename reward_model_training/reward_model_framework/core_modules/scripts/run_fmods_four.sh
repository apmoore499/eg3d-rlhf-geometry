


# #-------------------------------------------------------------------
# #-------------------------------------------------------------------


#trainer.limit_train_batches=1 trainer.limit_test_batches=1 trainer.limit_val_batches=1 
# # aw98


experiment=zzz_lmks_aw98.yaml

# cd /home/krillman/Documents/eg3dredo/supp_plus_code/eg3d_rlhf_code/reward_model_training/reward_model_framework/core_modules
# python train_rwd_model.py experiment=$experiment model.compile=false train_on=datamodule_first_binary model.optimizer.lr=5e-6 model.act_type=cos trainer.max_epochs=10 dloader.num_workers=6 dloader.prefetch_factor=2 


# cd /home/krillman/Documents/eg3dredo/supp_plus_code/eg3d_rlhf_code/reward_model_training/reward_model_framework/core_modules
# python train_rwd_model.py experiment=$experiment model.compile=false train_on=datamodule_third_ranked model.optimizer.lr=5e-6 model.act_type=cos trainer.max_epochs=10 dloader.num_workers=6 dloader.prefetch_factor=2 


# cd /home/krillman/Documents/eg3dredo/supp_plus_code/eg3d_rlhf_code/reward_model_training/reward_model_framework/core_modules
# python train_rwd_model.py experiment=$experiment model.compile=false train_on=datamodule_third_ranked_with_goodseed model.optimizer.lr=5e-6 model.act_type=cos trainer.max_epochs=10 dloader.num_workers=6 dloader.prefetch_factor=2 





# #-------------------------------------------------------------------
# #-------------------------------------------------------------------


#trainer.limit_train_batches=1 trainer.limit_test_batches=1 trainer.limit_val_batches=1 
# # single dmap


# resnet-50

experiment=zzz_sdmap_resnet_50_debug_dloaders.yaml

batch_augmentations=default #no augmentations. still works best.


train_on=datamodule_first_binary
cd /home/krillman/Documents/eg3dredo/supp_plus_code/eg3d_rlhf_code/reward_model_training/reward_model_framework/core_modules
python train_rwd_model.py experiment=$experiment prop_data=1.0 model.compile=false train_on=$train_on model.optimizer.lr=5e-6 model.act_type=cos trainer.max_epochs=10 dloader.num_workers=6 dloader.prefetch_factor=2 data.batch_augmentations=$batch_augmentations


# train_on=datamodule_third_ranked
# cd /home/krillman/Documents/eg3dredo/supp_plus_code/eg3d_rlhf_code/reward_model_training/reward_model_framework/core_modules
# python train_rwd_model.py experiment=$experiment prop_data=1.0 model.compile=false train_on=$train_on model.optimizer.lr=5e-6 model.act_type=cos trainer.max_epochs=10 dloader.num_workers=6 dloader.prefetch_factor=2 



# train_on=datamodule_third_ranked_with_goodseed
# cd /home/krillman/Documents/eg3dredo/supp_plus_code/eg3d_rlhf_code/reward_model_training/reward_model_framework/core_modules
# python train_rwd_model.py experiment=$experiment prop_data=1.0 model.compile=false train_on=$train_on model.optimizer.lr=5e-6 model.act_type=cos trainer.max_epochs=10 dloader.num_workers=6 dloader.prefetch_factor=2 




# # resnet-50


# experiment=zzz_sdmap_resnet_50_debug_dloaders.yaml  


# cd /home/krillman/Documents/eg3dredo/supp_plus_code/eg3d_rlhf_code/reward_model_training/reward_model_framework/core_modules
# python train_rwd_model.py experiment=$experiment prop_data=1.0 model.compile=false data.dset_dict.badseeds_models=[] train_on=datamodule_third model.optimizer.lr=5e-6 model.act_type=cos data.dset_dict.remove_middle_data=true trainer.max_epochs=10 dloader.num_workers=6 dloader.prefetch_factor=2 data.dset_dict.include_goodseed=True test_on_first=true test_on_third=true





# # vggface512



# experiment=zzz_sdmap_vggface2_debug_dloaders.yaml

# cd /home/krillman/Documents/eg3dredo/supp_plus_code/eg3d_rlhf_code/reward_model_training/reward_model_framework/core_modules
# python train_rwd_model.py experiment=$experiment prop_data=1.0 model.compile=false data.dset_dict.badseeds_models=[] train_on=datamodule_third model.optimizer.lr=5e-6 model.act_type=cos data.dset_dict.remove_middle_data=true trainer.max_epochs=10 dloader.num_workers=6 dloader.prefetch_factor=2 data.dset_dict.include_goodseed=True test_on_first=true test_on_third=true

# # python train_rwd_model.py experiment=zzz_sdmap_vggface2_debug_dloaders.yaml prop_data=1.0 train_on=datamodule_third model.optimizer.lr=5e-6 model.act_type=cos data.dset_dict.badseeds_models=[] data.dset_dict.remove_middle_data=true dloader.pin_memory=true trainer.max_epochs=10 dloader.num_workers=6 dloader.prefetch_factor=2 batch_size_all=16 data.dset_dict.include_goodseed=True test_on_first=true test_on_third=true



# # vgg-4096



# experiment=zzz_sdmap_vgg4096_debug_dloaders.yaml


# cd /home/krillman/Documents/eg3dredo/supp_plus_code/eg3d_rlhf_code/reward_model_training/reward_model_framework/core_modules
# python train_rwd_model.py experiment=$experiment prop_data=1.0 model.compile=false data.dset_dict.badseeds_models=[] train_on=datamodule_third model.optimizer.lr=5e-6 model.act_type=cos data.dset_dict.remove_middle_data=true trainer.max_epochs=10 dloader.num_workers=6 dloader.prefetch_factor=2 data.dset_dict.include_goodseed=True test_on_first=true test_on_third=true

# # python train_rwd_model.py experiment=zzz_sdmap_vgg4096_debug_dloaders.yaml prop_data=1.0 train_on=datamodule_third model.optimizer.lr=5e-6 model.act_type=cos data.dset_dict.badseeds_models=[] data.dset_dict.remove_middle_data=true dloader.pin_memory=true trainer.max_epochs=1 dloader.num_workers=6 dloader.prefetch_factor=2 batch_size_all=16 data.dset_dict.include_goodseed=True test_on_first=true test_on_third=true





# # #-------------------------------------------------------------------
# # #-------------------------------------------------------------------



# # # # triple dmap



experiment=zzz_tdmap_resnet_50_debug_dloaders.yaml


# # cd /home/krillman/Documents/eg3dredo/supp_plus_code/eg3d_rlhf_code/reward_model_training/reward_model_framework/core_modules
# # # python train_rwd_model.py experiment=zzz_tdmap_resnet_50_debug_dloaders.yaml prop_data=1.0 train_on=datamodule_first model.optimizer.lr=5e-6 model.act_type=cos data.dset_dict.remove_middle_data=true    trainer.max_epochs=10 dloader.num_workers=6 dloader.prefetch_factor=2 data.dset_dict.include_goodseed=True test_on_first=true test_on_third=true


# train_on=datamodule_first_binary
# cd /home/krillman/Documents/eg3dredo/supp_plus_code/eg3d_rlhf_code/reward_model_training/reward_model_framework/core_modules
# python train_rwd_model.py experiment=$experiment prop_data=1.0 model.compile=false train_on=$train_on model.optimizer.lr=5e-6 model.act_type=cos trainer.max_epochs=10 dloader.num_workers=6 dloader.prefetch_factor=2 


# train_on=datamodule_third_ranked
# cd /home/krillman/Documents/eg3dredo/supp_plus_code/eg3d_rlhf_code/reward_model_training/reward_model_framework/core_modules
# python train_rwd_model.py experiment=$experiment prop_data=1.0 model.compile=false train_on=$train_on model.optimizer.lr=5e-6 model.act_type=cos trainer.max_epochs=10 dloader.num_workers=6 dloader.prefetch_factor=2 



# train_on=datamodule_third_ranked_with_goodseed
# cd /home/krillman/Documents/eg3dredo/supp_plus_code/eg3d_rlhf_code/reward_model_training/reward_model_framework/core_modules
# python train_rwd_model.py experiment=$experiment prop_data=1.0 model.compile=false train_on=$train_on model.optimizer.lr=5e-6 model.act_type=cos trainer.max_epochs=10 dloader.num_workers=6 dloader.prefetch_factor=2 


#exit

# cd /home/krillman/Documents/eg3dredo/supp_plus_code/eg3d_rlhf_code/reward_model_training/reward_model_framework/core_modules
# python train_rwd_model.py experiment=$experiment prop_data=1.0 model.compile=false train_on=datamodule_first model.optimizer.lr=5e-6 model.act_type=cos dloader.pin_memory=true trainer.max_epochs=10 dloader.num_workers=6 dloader.prefetch_factor=2 batch_size_all=16 


# cd /home/krillman/Documents/eg3dredo/supp_plus_code/eg3d_rlhf_code/reward_model_training/reward_model_framework/core_modules
# python train_rwd_model.py experiment=$experiment prop_data=1.0 model.compile=false train_on=datamodule_third model.optimizer.lr=5e-6 model.act_type=cos data.dset_dict.remove_middle_data=true dloader.pin_memory=true trainer.max_epochs=10 dloader.num_workers=6 dloader.prefetch_factor=2 data.dset_dict.include_goodseed=True test_on_first=true test_on_third=true



# # #-------------------------------------------------------------------
# # #-------------------------------------------------------------------





# # # point cloud (Curve net)

# # cd /home/krillman/Documents/eg3dredo/supp_plus_code/eg3d_rlhf_code/reward_model_training/reward_model_framework/core_modules
# # # python train_rwd_model.py experiment=zzz_pcd_debug_dloaders.yaml prop_data=1.0 train_on=datamodule_first model.optimizer.lr=5e-6 model.act_type=cos    data.dset_dict.remove_middle_data=true trainer.max_epochs=10 dloader.num_workers=6 dloader.prefetch_factor=2 data.dset_dict.include_goodseed=True test_on_first=true test_on_third=true

# # cd /home/krillman/Documents/eg3dredo/supp_plus_code/eg3d_rlhf_code/reward_model_training/reward_model_framework/core_modules
# # python train_rwd_model.py experiment=zzz_pcd_debug_dloaders.yaml prop_data=1.0 train_on=datamodule_third model.optimizer.lr=5e-6 model.act_type=cos  data.dset_dict.badseeds_models=[] data.dset_dict.remove_middle_data=true trainer.max_epochs=10 dloader.num_workers=6 dloader.prefetch_factor=2 data.dset_dict.include_goodseed=True test_on_first=true test_on_third=true

# experiment=zzz_pcd_cvn_debug_dloaders.yaml


# cd /home/krillman/Documents/eg3dredo/supp_plus_code/eg3d_rlhf_code/reward_model_training/reward_model_framework/core_modules
# python train_rwd_model.py experiment=$experiment prop_data=1.0 model.compile=false data.dset_dict.badseeds_models=[] train_on=datamodule_third model.optimizer.lr=5e-6 model.act_type=cos data.dset_dict.remove_middle_data=true trainer.max_epochs=10 dloader.num_workers=6 dloader.prefetch_factor=2 data.dset_dict.include_goodseed=True test_on_first=true test_on_third=true



# # # point cloud (pointnet++ / pointnet 2)


# experiment=zzz_pcd_pn2_debug_dloaders.yaml




# train_on=datamodule_first_binary
# cd /home/krillman/Documents/eg3dredo/supp_plus_code/eg3d_rlhf_code/reward_model_training/reward_model_framework/core_modules
# python train_rwd_model.py experiment=$experiment prop_data=1.0 model.compile=false train_on=$train_on model.optimizer.lr=5e-6 model.act_type=cos trainer.max_epochs=10 dloader.num_workers=6 dloader.prefetch_factor=2 


# train_on=datamodule_third_ranked
# cd /home/krillman/Documents/eg3dredo/supp_plus_code/eg3d_rlhf_code/reward_model_training/reward_model_framework/core_modules
# # python train_rwd_model.py experiment=$experiment prop_data=1.0 model.compile=false train_on=$train_on model.optimizer.lr=5e-6 model.act_type=cos trainer.max_epochs=10 dloader.num_workers=6 dloader.prefetch_factor=2 batch_size_all=4



# train_on=datamodule_third_ranked_with_goodseed
# cd /home/krillman/Documents/eg3dredo/supp_plus_code/eg3d_rlhf_code/reward_model_training/reward_model_framework/core_modules
# python train_rwd_model.py experiment=$experiment prop_data=1.0 model.compile=false train_on=$train_on model.optimizer.lr=5e-6 model.act_type=cos trainer.max_epochs=10 dloader.num_workers=6 dloader.prefetch_factor=2 batch_size_all=4



#exit


# cd /home/krillman/Documents/eg3dredo/supp_plus_code/eg3d_rlhf_code/reward_model_training/reward_model_framework/core_modules
# python train_rwd_model.py experiment=$experiment prop_data=1.0 model.compile=false data.dset_dict.badseeds_models=[] train_on=datamodule_third model.optimizer.lr=5e-6 model.act_type=cos data.dset_dict.remove_middle_data=true trainer.max_epochs=10 dloader.num_workers=6 dloader.prefetch_factor=2 data.dset_dict.include_goodseed=False test_on_first=true test_on_third=true



# cd /home/krillman/Documents/eg3dredo/supp_plus_code/eg3d_rlhf_code/reward_model_training/reward_model_framework/core_modules
# python train_rwd_model.py experiment=zzz_pcd_pn2_debug_dloaders.yaml prop_data=1.0 train_on=datamodule_third model.optimizer.lr=5e-6 model.act_type=cos data.dset_dict.badseeds_models=[] data.dset_dict.remove_middle_data=true dloader.pin_memory=true    trainer.max_epochs=10 dloader.num_workers=6 dloader.prefetch_factor=2 batch_size_all=8 data.dset_dict.include_goodseed=True test_on_first=true test_on_third=true




# python train_rwd_model.py experiment=zzz_pcd_pn2_debug_dloaders.yaml prop_data=1.0 train_on=datamodule_third model.optimizer.lr=5e-6 model.act_type=cos data.dset_dict.badseeds_models=[] data.dset_dict.remove_middle_data=true dloader.pin_memory=true    trainer.max_epochs=10 dloader.num_workers=6 dloader.prefetch_factor=2 batch_size_all=2 data.dset_dict.include_goodseed=True test_on_first=true test_on_third=true



# # # point cloud (pointnet )

experiment=zzz_pcd_pn1_debug_dloaders.yaml

# cd /home/krillman/Documents/eg3dredo/supp_plus_code/eg3d_rlhf_code/reward_model_training/reward_model_framework/core_modules



# train_on=datamodule_first_binary
# cd /home/krillman/Documents/eg3dredo/supp_plus_code/eg3d_rlhf_code/reward_model_training/reward_model_framework/core_modules
# python train_rwd_model.py experiment=$experiment prop_data=1.0 model.compile=false train_on=$train_on model.optimizer.lr=5e-6 model.act_type=cos trainer.max_epochs=10 dloader.num_workers=6 dloader.prefetch_factor=2 batch_size_all=4


# train_on=datamodule_third_ranked
# cd /home/krillman/Documents/eg3dredo/supp_plus_code/eg3d_rlhf_code/reward_model_training/reward_model_framework/core_modules
# python train_rwd_model.py experiment=$experiment prop_data=1.0 model.compile=false train_on=$train_on model.optimizer.lr=5e-6 model.act_type=cos trainer.max_epochs=10 dloader.num_workers=6 dloader.prefetch_factor=2 batch_size_all=4



# train_on=datamodule_third_ranked_with_goodseed
# cd /home/krillman/Documents/eg3dredo/supp_plus_code/eg3d_rlhf_code/reward_model_training/reward_model_framework/core_modules
# python train_rwd_model.py experiment=$experiment prop_data=1.0 model.compile=false train_on=$train_on model.optimizer.lr=5e-6 model.act_type=cos trainer.max_epochs=10 dloader.num_workers=6 dloader.prefetch_factor=2 batch_size_all=4





# python train_rwd_model.py experiment=$experiment prop_data=1.0 model.compile=false data.dset_dict.badseeds_models=[] train_on=datamodule_third model.optimizer.lr=5e-6 model.act_type=cos data.dset_dict.remove_middle_data=true trainer.max_epochs=10 dloader.num_workers=6 dloader.prefetch_factor=2 data.dset_dict.include_goodseed=False test_on_first=true test_on_third=true

# python train_rwd_model.py experiment=zzz_pcd_pn1_debug_dloaders.yaml prop_data=1.0 train_on=datamodule_third model.optimizer.lr=5e-6 model.act_type=cos data.dset_dict.badseeds_models=[] data.dset_dict.remove_middle_data=true dloader.pin_memory=true    trainer.max_epochs=10 dloader.num_workers=6 dloader.prefetch_factor=2 batch_size_all=2 data.dset_dict.include_goodseed=True test_on_first=true test_on_third=true



# #  coatnet on the centroid patches (slow)

# # python train_rwd_model.py experiment=zzz_coatnet_patches_88.yaml prop_data=1.0 train_on=datamodule_third model.optimizer.lr=5e-6 model.act_type=cos data.dset_dict.remove_middle_data=true trainer.max_epochs=10 dloader.num_workers=6 dloader.prefetch_factor=2 data.dset_dict.include_goodseed=True batch_size_all=8 test_on_first=true test_on_third=true

# experiment=zzz_coatnet_patches_88.yaml

# cd /home/krillman/Documents/eg3dredo/supp_plus_code/eg3d_rlhf_code/reward_model_training/reward_model_framework/core_modules
# python train_rwd_model.py experiment=$experiment prop_data=1.0 model.compile=false data.dset_dict.badseeds_models=[] train_on=datamodule_third model.optimizer.lr=5e-6 model.act_type=cos data.dset_dict.remove_middle_data=true trainer.max_epochs=10 dloader.num_workers=6 dloader.prefetch_factor=2 data.dset_dict.include_goodseed=True test_on_first=true test_on_third=true


# # #-------------------------------------------------------------------
# # #-------------------------------------------------------------------


# # sigma field 256


# # cd /home/krillman/Documents/eg3dredo/supp_plus_code/eg3d_rlhf_code/reward_model_training/reward_model_framework/core_modules
# # python train_rwd_model.py experiment=sfield_256.yaml prop_data=1.0 train_on=datamodule_first model.optimizer.lr=5e-6 model.act_type=cos data.dset_dict.remove_middle_data=true trainer.max_epochs=10 dloader.num_workers=6 dloader.prefetch_factor=2 data.dset_dict.include_goodseed=True test_on_first=true test_on_third=true

#conda activate pointface_env


experiment=sfield_256.yaml



# train_on=datamodule_first_binary
# cd /home/krillman/Documents/eg3dredo/supp_plus_code/eg3d_rlhf_code/reward_model_training/reward_model_framework/core_modules
# python train_rwd_model.py experiment=$experiment prop_data=1.0 model.compile=false train_on=$train_on model.optimizer.lr=5e-6 model.act_type=cos trainer.max_epochs=10 dloader.num_workers=6 dloader.prefetch_factor=2 batch_size_all=4 # dloader.prefetch_factor=2 batch_size_all=2


# train_on=datamodule_third_ranked
# cd /home/krillman/Documents/eg3dredo/supp_plus_code/eg3d_rlhf_code/reward_model_training/reward_model_framework/core_modules
# python train_rwd_model.py experiment=$experiment prop_data=1.0 model.compile=false train_on=$train_on model.optimizer.lr=5e-6 model.act_type=cos trainer.max_epochs=10 dloader.num_workers=6 dloader.prefetch_factor=2 batch_size_all=2



# train_on=datamodule_third_ranked_with_goodseed
# cd /home/krillman/Documents/eg3dredo/supp_plus_code/eg3d_rlhf_code/reward_model_training/reward_model_framework/core_modules
# python train_rwd_model.py experiment=$experiment prop_data=1.0 model.compile=false train_on=$train_on model.optimizer.lr=5e-6 model.act_type=cos trainer.max_epochs=2 dloader.num_workers=6 dloader.prefetch_factor=2 batch_size_all=1






# #n workoer=0 cos we want to load direc ton cuda
# cd /home/krillman/Documents/eg3dredo/supp_plus_code/eg3d_rlhf_code/reward_model_training/reward_model_framework/core_modules
# python train_rwd_model.py experiment=$experiment prop_data=1.0 model.compile=false data.dset_dict.badseeds_models=[] train_on=datamodule_third model.optimizer.lr=5e-6 model.act_type=cos data.dset_dict.remove_middle_data=true trainer.max_epochs=10 dloader.num_workers=6 dloader.prefetch_factor=2 data.dset_dict.include_goodseed=True test_on_first=true test_on_third=true




# cd /home/krillman/Documents/eg3dredo/supp_plus_code/eg3d_rlhf_code/reward_model_training/reward_model_framework/core_modules
# python train_rwd_model.py experiment=sfield_256.yaml prop_data=1.0 train_on=datamodule_third model.optimizer.lr=5e-6 model.act_type=cos data.dset_dict.remove_middle_data=true trainer.max_epochs=10 dloader.num_workers=6 dloader.prefetch_factor=2 data.dset_dict.include_goodseed=True test_on_first=true test_on_third=true






# # #-------------------------------------------------------------------
# # #-------------------------------------------------------------------



# # cd /home/krillman/Documents/eg3dredo/supp_plus_code/eg3d_rlhf_code/reward_model_training/reward_model_framework/core_modules






# data.dset_dict.badseeds_models=[]
# model.n_dmaps=1
# model.n_dmaps=3
