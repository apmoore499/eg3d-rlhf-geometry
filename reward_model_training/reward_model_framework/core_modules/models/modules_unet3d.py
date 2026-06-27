import monai
import omegaconf
from core_modules.models.utils_base import reduce_x

import torch
import torch.nn as nn
from pytorch3dunet.unet3d.buildingblocks import (
    DoubleConv,
    ResNetBlock,
    ResNetBlockSE,
    create_decoders,
    create_encoders,
)
from pytorch3dunet.unet3d.utils import get_class, number_of_features_per_level


class AbstractUNet(nn.Module):
    """Base class for standard and residual UNet.

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output segmentation masks;
            Note that the of out_channels might correspond to either
            different semantic classes or to different binary segmentation mask.
            It's up to the user of the class to interpret the out_channels and
            use the proper loss criterion during training (i.e. CrossEntropyLoss (multi-class)
            or BCEWithLogitsLoss (two-class) respectively)
        f_maps (int, tuple): number of feature maps at each level of the encoder; if it's an integer the number
            of feature maps is given by the geometric progression: f_maps ^ k, k=1,2,3,4
        final_sigmoid (bool): if True apply element-wise nn.Sigmoid after the final 1x1 convolution,
            otherwise apply nn.Softmax. In effect only if `self.training == False`, i.e. during validation/testing
        basic_module: basic model for the encoder/decoder (DoubleConv, ResNetBlock, ....)
        layer_order (string): determines the order of layers in `SingleConv` module.
            E.g. 'crg' stands for GroupNorm3d+Conv3d+ReLU. See `SingleConv` for more info
        num_groups (int): number of groups for the GroupNorm
        num_levels (int): number of levels in the encoder/decoder path (applied only if f_maps is an int)
            default: 4
        is_segmentation (bool): if True and the model is in eval mode, Sigmoid/Softmax normalization is applied
            after the final convolution; if False (regression problem) the normalization layer is skipped
        conv_kernel_size (int or tuple): size of the convolving kernel in the basic_module
        pool_kernel_size (int or tuple): the size of the window
        conv_padding (int or tuple): add zero-padding added to all three sides of the input
        conv_upscale (int): number of the convolution to upscale in encoder if DoubleConv, default: 2
        upsample (str): algorithm used for decoder upsampling:
            InterpolateUpsampling:   'nearest' | 'linear' | 'bilinear' | 'trilinear' | 'area'
            TransposeConvUpsampling: 'deconv'
            No upsampling:           None
            Default: 'default' (chooses automatically)
        dropout_prob (float or tuple): dropout probability, default: 0.1
        is3d (bool): if True the model is 3D, otherwise 2D, default: True
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        final_sigmoid,
        basic_module,
        f_maps=64,
        layer_order="gcr",
        num_groups=8,
        num_levels=4,
        is_segmentation=True,
        conv_kernel_size=3,
        pool_kernel_size=2,
        conv_padding=1,
        conv_upscale=2,
        upsample="default",
        dropout_prob=0.1,
        is3d=True,
        forward_type="patches",
        crop_each_to_positive_only=False,
        softplus_sigma_reconstruction=False,
        final_size_out=1024,
        n_channels_for_recon=1,
        n_channels_for_feature=1,
        recon_pool_type="avg",
        feature_pool_type="max",
        final_sigma_decoder_initialise_minimal_noise=False,
        **kwargs,
    ):  # false works better #if we are using sigma_256_self_norm augmentations....  # false works better #if we are using sigma_256_self_norm augmentations....
        super(AbstractUNet, self).__init__()

        if isinstance(f_maps, int):
            f_maps = number_of_features_per_level(f_maps, num_levels=num_levels)

        assert isinstance(f_maps, list) or isinstance(f_maps, tuple) or isinstance(f_maps, omegaconf.listconfig.ListConfig)
        assert len(f_maps) > 1, "Required at least 2 levels in the U-Net"
        if "g" in layer_order:
            assert num_groups is not None, "num_groups must be specified if GroupNorm is used"
        self.final_size_out = final_size_out

        # create encoder path
        self.encoders = create_encoders(in_channels, f_maps, basic_module, conv_kernel_size, conv_padding, conv_upscale, dropout_prob, layer_order, num_groups, pool_kernel_size, is3d)

        # create decoder path
        self.decoders = create_decoders(f_maps, basic_module, conv_kernel_size, conv_padding, layer_order, num_groups, upsample, dropout_prob, is3d)

        # in the last layer a 1×1 convolution reduces the number of output channels to the number of labels
        if is3d:
            self.final_conv = nn.Conv3d(f_maps[0], out_channels, 1)
        else:
            self.final_conv = nn.Conv2d(f_maps[0], out_channels, 1)

        if forward_type == "patches":
            self.adap_final = nn.AdaptiveMaxPool3d((128, 1, 1))  # .cuda()

        elif forward_type == "entire":
            self.adap_final = nn.AdaptiveMaxPool3d((self.final_size_out, 1, 1))  # .cuda()

        if is_segmentation:
            # semantic segmentation problem
            if final_sigmoid:
                self.final_activation = nn.Sigmoid()
            else:
                self.final_activation = nn.Softmax(dim=1)
        else:
            # regression problem
            self.final_activation = None

        self.rsc = monai.transforms.RandSpatialCrop(roi_size=[1, 100, 100, 100])

        # self.adap_over_all=nn.AdaptiveMaxPool3d((512,1,1))#.cuda()

        self.cropper = monai.transforms.CropForeground(select_fn=self.threshold_at_one, margin=0)
        self.cropper_inside = monai.transforms.CropForeground(select_fn=self.threshold_l_one, margin=0)

        # s_crop=cropper(sigmas)

        self.crop_each_to_positive_only = crop_each_to_positive_only

        self.forward_type = forward_type

        self.conv1d_for_recon = nn.Conv1d(1, 1, 1)
        self.conv3d_for_recon = nn.Conv3d(1, 1, 1)

        self.softplus = torch.nn.Softplus()
        # set params manually...

        # self.conv3d_for_recon.named_parameters()
        self.final_sigma_decoder_initialise_minimal_noise = final_sigma_decoder_initialise_minimal_noise  # whether we are initialising the sigma decoder for very small noise, ideal if sigma scaled between 0-1

        if final_sigma_decoder_initialise_minimal_noise:
            self.setup_params_init_for_decoder_final()

        self.softplus_sigma_reconstruction = softplus_sigma_reconstruction

        if out_channels == 2:
            print(f"out channels=2, hence n_channels for recon/feature arguemnts have no effect")

            self.recon_pool_layer_final = torch.nn.Identity()
            self.feature_pool_layer_final = torch.nn.Identity()

        else:
            assert out_channels == n_channels_for_recon + n_channels_for_feature, f"error out_channels != recon + feature reecon channels. recon:\t{n_channels_for_recon},\tfeature:\t{n_channels_for_feature}"

            self.n_channels_for_recon = n_channels_for_recon
            self.n_channels_for_feature = n_channels_for_feature
            # self.recon_pool_layer_final=self.get_adapt_layer(recon_pool_type)
            # self.feature_pool_layer_final=self.get_adapt_layer(feature_pool_type)
            self.recon_pool_type = recon_pool_type
            self.feature_pool_type = feature_pool_type

    # def get_adapt_layer(self,ltype): #ltype=='max' or 'avg' or 'mean

    #     if ltype=='max':
    #         return nn.AdaptiveMaxPool3d((1024, 1, 1))#.cuda()

    #     elif ltype in ['avg','mean']
    #         return nn.AdaptiveAvgPool3d((1024, 1, 1))#.cuda()

    #     else:
    #         assert False, f'error the type for ltype is not max or avg/mean. ltype: {ltype}'

    def setup_params_init_for_decoder_final(self):
        sd = self.state_dict()
        sd["conv3d_for_recon.weight"] = 1 / sd["conv3d_for_recon.weight"] * sd["conv3d_for_recon.weight"]  # set to 1.0
        sd["conv3d_for_recon.bias"] = sd["conv3d_for_recon.bias"] - sd["conv3d_for_recon.bias"] + (torch.rand_like(sd["conv3d_for_recon.bias"]) - 0.5) * 1e-2  # +1 #set to zero

        # sd['conv3d_for_recon.weight']=1/sd['conv3d_for_recon.weight']*sd['conv3d_for_recon.weight'] #set to 1.0
        # sd['conv3d_for_recon.bias']=sd['conv3d_for_recon.bias']-sd['conv3d_for_recon.bias'] -(torch.rand_like(sd['conv3d_for_recon.bias']))*1e1 -50   #+1 #set to zero

        self.load_state_dict(sd)

        return

    def threshold_l_one(self, x):
        return x < 1.0

    def threshold_at_one(self, x):
        # threshold at 1
        return x > 0.0

    def split_into_sub(self, orig_x):
        xx = orig_x
        patches = xx.unfold(2, 128, 128).unfold(1, 128, 128).unfold(0, 128, 128)
        patches = patches.contiguous().view(-1, 128, 128, 128)
        return patches

    def forward_thru_one_set_of_patches(self, patches):
        x = patches
        encoders_features = []
        for encoder in self.encoders:
            x = encoder(x)
            encoders_features.insert(0, x)  # reverse the encoder outputs to be aligned with the decoder

        # remove the last encoder's output from the list
        # !!remember: it's the 1st in the list
        encoders_features = encoders_features[1:]

        # decoder part
        for decoder, encoder_features in zip(self.decoders, encoders_features):
            # pass the output from the corresponding encoder and the output
            # of the previous decoder
            x = decoder(encoder_features, x)

        x = self.final_conv(x)  # +patches

        recon_x_pre_conv3d = x[:, : self.n_channels_for_recon, ...]

        recon_x_pre_conv3d = reduce_x(x=recon_x_pre_conv3d, agg_type=self.recon_pool_type, sel_dim=1)
        # recon_x_pre_conv3d=reduce_x(x=recon_x_pre_conv3d,agg_type='mean',sel_dim=1)
        # reduce_x(x, agg_type,return_idx=False,sel_dim=2):

        recon_x = self.conv3d_for_recon(recon_x_pre_conv3d)  # .view(recon_x_pre_conv1d_shape)

        if self.softplus_sigma_reconstruction:
            recon_x = self.softplus(recon_x)

        # recon_x_pre_conv3d=reduce_x(recon_x_pre_conv3d,agg_type=self.recon_pool_type,sel_dim=1)

        features_intermediate = x[:, -self.n_channels_for_feature :, ...]  # .squeeze(-1)
        # features_intermediate=x[:,-self.n_channels_for_feature:,...]#.squeeze(-1)

        features_intermediate = reduce_x(features_intermediate, agg_type=self.feature_pool_type, sel_dim=1)

        global_vec = self.adap_final(features_intermediate)  # .squeeze(-1))

        # ef=encoders_features[-1]
        # efs=ef.shape

        # global_vector=self.adap_final(ef.view(efs[0],efs[1],efs[2],-1))#.shape

        return dict(global_vector=global_vec, recon_x=recon_x)
        # then reshape here

        # print('testing')
        #

    def forward(self, x):
        # encoder part
        encoders_features = []

        # get the rand spatial crop of it

        # unfold it to smaller subvolumes
        bsize = x.shape[0]
        indiv_x = [xx.squeeze(0) for xx in x]

        if self.forward_type == "patches":
            indiv_split = [self.split_into_sub(x).unsqueeze(1) for x in indiv_x]

        elif self.forward_type == "entire":
            patches = [x.unsqueeze(0).unsqueeze(0) for x in indiv_x]

        if self.crop_each_to_positive_only:
            patches = [self.cropper_inside(self.cropper(x)) for x in patches]

        processed_total = [self.forward_thru_one_set_of_patches(p) for p in patches]

        processed = [p["global_vector"].view(-1, self.final_size_out) for p in processed_total]
        recon_x = [p["recon_x"] for p in processed_total]

        gvb = torch.cat(processed, 0)

        return dict(global_vector=gvb, recon_x=recon_x, orig_x=patches)

    def forward_to_global_vec(self, x, return_global_only=False):
        xx = self.forward(x)

        if return_global_only:
            return xx["global_vector"]

        else:
            return xx


class UNet3D(AbstractUNet):
    """
    3DUnet model from
    `"3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation"
        <https://arxiv.org/pdf/1606.06650.pdf>`.

    Uses `DoubleConv` as a basic_module and nearest neighbor upsampling in the decoder
    """

    def __init__(self, in_channels, out_channels, final_sigmoid=True, f_maps=64, layer_order="gcr", num_groups=8, num_levels=4, is_segmentation=True, conv_padding=1, conv_upscale=2, upsample="default", dropout_prob=0.1, **kwargs):
        super(UNet3D, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            final_sigmoid=final_sigmoid,
            basic_module=DoubleConv,
            f_maps=f_maps,
            layer_order=layer_order,
            num_groups=num_groups,
            num_levels=num_levels,
            is_segmentation=is_segmentation,
            conv_padding=conv_padding,
            conv_upscale=conv_upscale,
            upsample=upsample,
            dropout_prob=dropout_prob,
            is3d=True,
            **kwargs,
        )


class ResidualUNet3D(AbstractUNet):
    """
    Residual 3DUnet model implementation based on https://arxiv.org/pdf/1706.00120.pdf.
    Uses ResNetBlock as a basic building block, summation joining instead
    of concatenation joining and transposed convolutions for upsampling (watch out for block artifacts).
    Since the model effectively becomes a residual net, in theory it allows for deeper UNet.
    """

    def __init__(self, in_channels, out_channels, final_sigmoid=True, f_maps=64, layer_order="gcr", num_groups=8, num_levels=5, is_segmentation=True, conv_padding=1, final_size_out=1024, conv_upscale=2, upsample="default", dropout_prob=0.1, **kwargs):
        super(ResidualUNet3D, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            final_sigmoid=final_sigmoid,
            basic_module=ResNetBlock,
            final_size_out=final_size_out,
            f_maps=f_maps,
            layer_order=layer_order,
            num_groups=num_groups,
            num_levels=num_levels,
            is_segmentation=is_segmentation,
            conv_padding=conv_padding,
            conv_upscale=conv_upscale,
            upsample=upsample,
            dropout_prob=dropout_prob,
            is3d=True,
            **kwargs,
        )


class ResidualUNetSE3D(AbstractUNet):
    """_summary_
    Residual 3DUnet model implementation with squeeze and excitation based on
    https://arxiv.org/pdf/1706.00120.pdf.
    Uses ResNetBlockSE as a basic building block, summation joining instead
    of concatenation joining and transposed convolutions for upsampling (watch
    out for block artifacts). Since the model effectively becomes a residual
    net, in theory it allows for deeper UNet.
    """  # conv_padding=1,

    def __init__(self, in_channels, out_channels, final_sigmoid=True, f_maps=64, layer_order="gcr", num_groups=8, num_levels=5, is_segmentation=True, conv_upscale=2, upsample="default", dropout_prob=0.1, final_size_out=1024, **kwargs):
        super(ResidualUNetSE3D, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            final_sigmoid=final_sigmoid,
            basic_module=ResNetBlockSE,
            final_size_out=final_size_out,
            f_maps=f_maps,
            layer_order=layer_order,
            num_groups=num_groups,
            num_levels=num_levels,
            is_segmentation=is_segmentation,
            # conv_padding=conv_padding,
            conv_upscale=conv_upscale,
            upsample=upsample,
            dropout_prob=dropout_prob,
            is3d=True,
            **kwargs,
        )


class UNet2D(AbstractUNet):
    """
    2DUnet model from
    `"U-Net: Convolutional Networks for Biomedical Image Segmentation" <https://arxiv.org/abs/1505.04597>`
    """

    def __init__(self, in_channels, out_channels, final_sigmoid=True, f_maps=64, layer_order="gcr", num_groups=8, num_levels=4, is_segmentation=True, conv_padding=1, conv_upscale=2, upsample="default", dropout_prob=0.1, **kwargs):
        super(UNet2D, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            final_sigmoid=final_sigmoid,
            basic_module=DoubleConv,
            f_maps=f_maps,
            layer_order=layer_order,
            num_groups=num_groups,
            num_levels=num_levels,
            is_segmentation=is_segmentation,
            conv_padding=conv_padding,
            conv_upscale=conv_upscale,
            upsample=upsample,
            dropout_prob=dropout_prob,
            is3d=False,
        )


class ResidualUNet2D(AbstractUNet):
    """
    Residual 2DUnet model implementation based on https://arxiv.org/pdf/1706.00120.pdf.
    """

    def __init__(self, in_channels, out_channels, final_sigmoid=True, f_maps=64, layer_order="gcr", num_groups=8, num_levels=5, is_segmentation=True, conv_padding=1, conv_upscale=2, upsample="default", dropout_prob=0.1, **kwargs):
        super(ResidualUNet2D, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            final_sigmoid=final_sigmoid,
            basic_module=ResNetBlock,
            f_maps=f_maps,
            layer_order=layer_order,
            num_groups=num_groups,
            num_levels=num_levels,
            is_segmentation=is_segmentation,
            conv_padding=conv_padding,
            conv_upscale=conv_upscale,
            upsample=upsample,
            dropout_prob=dropout_prob,
            is3d=False,
        )


def get_model(model_config):
    model_class = get_class(model_config["name"], modules=["pytorch3dunet.unet3d.model"])
    return model_class(**model_config)
