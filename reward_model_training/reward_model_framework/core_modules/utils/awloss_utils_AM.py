import cv2
import torch
import torch.nn.functional as F
import torchlm
from torchlm.models import pipnet
from torchlm.tools import faceboxesv2


# moved to here 26122025 from 000_RLHF/external modules dir
def return_awloss_model_98():
    # initialize detector + landmark model
    torchlm.runtime.bind(faceboxesv2(device="cuda"))  # or device="cuda"
    landmark_model = pipnet(backbone="resnet18", pretrained=True, num_nb=10, num_lms=98, net_stride=32, input_size=256, meanface_type="wflw")

    torchlm.runtime.bind(landmark_model)

    return landmark_model


def predict_landmarks_from_rgb_on_gpu(model_ft, im):
    """Predict AW98 landmarks from a normalized RGB tensor on GPU.

    Expects im of shape (1,3,H,W) in range [0,1]. Returns tensor (1,98,2) on the same device.
    """
    model_ft.eval()
    device = im.device
    b, c, h, w = im.shape
    target = getattr(model_ft, "input_size", 256)

    img = F.interpolate(im, size=(target, target), mode="bilinear", align_corners=False)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    img = (img - mean) / std

    with torch.no_grad():
        outputs_cls, outputs_x, outputs_y, outputs_nb_x, outputs_nb_y = model_ft(img)

    tmp_batch, tmp_channel, tmp_height, tmp_width = outputs_cls.size()
    outputs_cls = outputs_cls.view(tmp_batch * tmp_channel, -1)
    max_ids = torch.argmax(outputs_cls, 1, keepdim=True)
    max_ids_nb = max_ids.repeat(1, model_ft.num_nb).view(-1, 1)

    outputs_x = outputs_x.view(tmp_batch * tmp_channel, -1)
    outputs_y = outputs_y.view(tmp_batch * tmp_channel, -1)
    outputs_x_select = torch.gather(outputs_x, 1, max_ids).squeeze(1)
    outputs_y_select = torch.gather(outputs_y, 1, max_ids).squeeze(1)

    outputs_nb_x = outputs_nb_x.view(tmp_batch * model_ft.num_nb * tmp_channel, -1)
    outputs_nb_y = outputs_nb_y.view(tmp_batch * model_ft.num_nb * tmp_channel, -1)
    outputs_nb_x_select = torch.gather(outputs_nb_x, 1, max_ids_nb).squeeze(1).view(-1, model_ft.num_nb)
    outputs_nb_y_select = torch.gather(outputs_nb_y, 1, max_ids_nb).squeeze(1).view(-1, model_ft.num_nb)

    lms_pred_x = (max_ids % tmp_width).view(-1, 1).float() + outputs_x_select.view(-1, 1)
    lms_pred_y = torch.floor(max_ids / tmp_width).view(-1, 1).float() + outputs_y_select.view(-1, 1)
    stride_ratio = float(model_ft.input_size) / model_ft.net_stride
    lms_pred_x /= stride_ratio
    lms_pred_y /= stride_ratio

    lms_pred_nb_x = (max_ids % tmp_width).view(-1, 1).float() + outputs_nb_x_select
    lms_pred_nb_y = torch.floor(max_ids / tmp_width).view(-1, 1).float() + outputs_nb_y_select
    lms_pred_nb_x = lms_pred_nb_x.view(-1, model_ft.num_nb) / stride_ratio
    lms_pred_nb_y = lms_pred_nb_y.view(-1, model_ft.num_nb) / stride_ratio

    tmp_nb_x = lms_pred_nb_x[model_ft.reverse_index1, model_ft.reverse_index2].view(model_ft.num_lms, model_ft.max_len)
    tmp_nb_y = lms_pred_nb_y[model_ft.reverse_index1, model_ft.reverse_index2].view(model_ft.num_lms, model_ft.max_len)
    tmp_x = torch.mean(torch.cat((lms_pred_x, tmp_nb_x), dim=1), dim=1).view(-1, 1)
    tmp_y = torch.mean(torch.cat((lms_pred_y, tmp_nb_y), dim=1), dim=1).view(-1, 1)
    lms_pred_merge = torch.cat((tmp_x, tmp_y), dim=1)

    lms_pred_merge[:, 0] *= float(w)
    lms_pred_merge[:, 1] *= float(h)

    return lms_pred_merge.view(b, model_ft.num_lms, 2)


class AW98Helper:
    """Lightweight wrapper around the AW98 landmark model."""

    def __init__(self):
        self.M_aw98 = return_awloss_model_98()

    def get_device(self):
        device = self.M_aw98.get_parameter("conv1.conv.weight").device
        return device

    def predict_landmarks_from_rgb_on_gpu(self, rgb, detach: bool = False):
        lmks = predict_landmarks_from_rgb_on_gpu(model_ft=self.M_aw98, im=rgb)
        if detach:
            lmks = lmks.detach()
        return lmks

    def pil_image_to_tensor_for_lmks(self, img):
        import torchvision.transforms as tf

        img = tf.functional.to_tensor(img).unsqueeze(0)
        img = torch.nn.functional.interpolate(img, size=(256, 256))
        assert list(img.shape) == [1, 3, 256, 256], f"error shape wrong: {img.shape}"
        return img
