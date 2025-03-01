import torch
import random


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class Lighting(object):
    """Lighting noise(AlexNet - style PCA - based noise)"""

    def __init__(self, alphastd, eigval, eigvec, device="cpu"):
        self.alphastd = alphastd
        self.eigval = torch.tensor(eigval, device=device)
        self.eigvec = torch.tensor(eigvec, device=device)

    def __call__(self, img):
        if self.alphastd == 0:
            return img

        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = (
            self.eigvec.type_as(img)
            .clone()
            .mul(alpha.view(1, 3).expand(3, 3))
            .mul(self.eigval.view(1, 3).expand(3, 3))
            .sum(1)
            .squeeze()
        )

        # make differentiable
        if len(img.shape) == 4:
            return img + rgb.view(1, 3, 1, 1).expand_as(img)
        else:
            return img + rgb.view(3, 1, 1).expand_as(img)


class Grayscale(object):
    def __call__(self, img):
        gs = img.clone()
        gs[0].mul_(0.299).add_(0.587, gs[1]).add_(0.114, gs[2])
        gs[1].copy_(gs[0])
        gs[2].copy_(gs[0])
        return gs


class Saturation(object):
    def __init__(self, var):
        self.var = var

    def __call__(self, img):
        gs = Grayscale()(img)
        alpha = random.uniform(-self.var, self.var)
        return img.lerp(gs, alpha)


class Brightness(object):
    def __init__(self, var):
        self.var = var

    def __call__(self, img):
        gs = img.new().resize_as_(img).zero_()
        alpha = random.uniform(-self.var, self.var)
        return img.lerp(gs, alpha)


class Contrast(object):
    def __init__(self, var):
        self.var = var

    def __call__(self, img):
        gs = Grayscale()(img)
        gs.fill_(gs.mean())
        alpha = random.uniform(-self.var, self.var)
        return img.lerp(gs, alpha)


class ColorJitter(object):
    def __init__(self, brightness=0.4, contrast=0.4, saturation=0.4):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation

    def __call__(self, img):
        self.transforms = []
        if self.brightness != 0:
            self.transforms.append(Brightness(self.brightness))
        if self.contrast != 0:
            self.transforms.append(Contrast(self.contrast))
        if self.saturation != 0:
            self.transforms.append(Saturation(self.saturation))

        random.shuffle(self.transforms)
        transform = Compose(self.transforms)
        # print(transform)
        return transform(img)


class CutOut:
    def __init__(self, ratio, device="cpu"):
        self.ratio = ratio
        self.device = device

    def __call__(self, x):
        n, _, h, w = x.shape
        cutout_size = [int(h * self.ratio + 0.5), int(w * self.ratio + 0.5)]
        offset_x = torch.randint(
            h + (1 - cutout_size[0] % 2), size=[1], device=self.device
        )[0]
        offset_y = torch.randint(
            w + (1 - cutout_size[1] % 2), size=[1], device=self.device
        )[0]

        grid_batch, grid_x, grid_y = torch.meshgrid(
            torch.arange(n, dtype=torch.long, device=self.device),
            torch.arange(cutout_size[0], dtype=torch.long, device=self.device),
            torch.arange(cutout_size[1], dtype=torch.long, device=self.device),
        )
        grid_x = torch.clamp(grid_x + offset_x - cutout_size[0] // 2, min=0, max=h - 1)
        grid_y = torch.clamp(grid_y + offset_y - cutout_size[1] // 2, min=0, max=w - 1)
        mask = torch.ones(n, h, w, dtype=x.dtype, device=self.device)
        mask[grid_batch, grid_x, grid_y] = 0

        x = x * mask.unsqueeze(1)
        return x


class Normalize:
    def __init__(self, mean, std, device="cpu"):
        self.mean = torch.tensor(mean, device=device).reshape(1, len(mean), 1, 1)
        self.std = torch.tensor(std, device=device).reshape(1, len(mean), 1, 1)

    def __call__(self, x, seed=-1):
        return (x - self.mean) / self.std
