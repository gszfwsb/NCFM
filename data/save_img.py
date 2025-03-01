from .dataset_statistics import STDS,MEANS
import torch
from torchvision.utils import save_image
import torch.nn.functional as F

def img_denormlaize(img, dataname='imagenet'):
    """Scaling and shift a batch of images (NCHW)
    """
    mean = MEANS[dataname] 
    std = STDS[dataname]
    nch = img.shape[1]

    mean = torch.tensor(mean, device=img.device).reshape(1, nch, 1, 1)
    std = torch.tensor(std, device=img.device).reshape(1, nch, 1, 1)

    return img * std + mean






def save_img(save_dir, img, unnormalize=True, max_num=200, size=64, nrow=10, dataname='imagenet'):
    img = img[:max_num].detach()
    if unnormalize:
        img = img_denormlaize(img, dataname=dataname)
    img = torch.clamp(img, min=0., max=1.)

    if img.shape[-1] > size:
        img = F.interpolate(img, size)
    save_image(img.cpu(), save_dir, nrow=nrow)