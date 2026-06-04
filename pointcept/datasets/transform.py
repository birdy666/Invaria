"""
3D point cloud augmentation

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com), Yujia Zhang (yujia.zhang.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import random
import numbers
import scipy
import scipy.ndimage
import scipy.interpolate
import scipy.stats
import numpy as np
import torch
from torchvision import transforms
import copy
from collections.abc import Sequence, Mapping
from pointcept.utils.registry import Registry

TRANSFORMS = Registry("transforms")


def index_operator(data_dict, index, duplicate=False):
    # index selection operator for keys in "index_valid_keys"
    # custom these keys by "Update" transform in config
    if "index_valid_keys" not in data_dict:
        data_dict["index_valid_keys"] = [
            "coord",
            "color",
            "normal",
            "superpoint",
            "strength",
            "segment",
            "instance",
        ]
    if not duplicate:
        for key in data_dict["index_valid_keys"]:
            if key in data_dict:
                data_dict[key] = data_dict[key][index]
        return data_dict
    else:
        data_dict_ = dict()
        for key in data_dict.keys():
            if key in data_dict["index_valid_keys"]:
                data_dict_[key] = data_dict[key][index]
            elif key == "index_valid_keys":
                data_dict_[key] = copy.copy(data_dict[key])
            else:
                data_dict_[key] = data_dict[key]
        return data_dict_


@TRANSFORMS.register_module()
class Collect(object):
    def __init__(self, keys, offset_keys_dict=None, **kwargs):
        """
        e.g. Collect(keys=[coord], feat_keys=[coord, color])
        """
        if offset_keys_dict is None:
            offset_keys_dict = dict(offset="coord")
        self.keys = keys
        self.offset_keys = offset_keys_dict
        self.kwargs = kwargs

    def __call__(self, data_dict):
        data = dict()
        if isinstance(self.keys, str):
            self.keys = [self.keys]
        for key in self.keys:
            data[key] = data_dict[key]
        for key, value in self.offset_keys.items():
            data[key] = torch.tensor([data_dict[value].shape[0]])
        for name, keys in self.kwargs.items():
            name = name.replace("_keys", "")
            assert isinstance(keys, Sequence)
            data[name] = torch.cat([data_dict[key].float() for key in keys], dim=1)
        return data


@TRANSFORMS.register_module()
class Copy(object):
    def __init__(self, keys_dict=None):
        if keys_dict is None:
            keys_dict = dict(coord="origin_coord", segment="origin_segment")
        self.keys_dict = keys_dict

    def __call__(self, data_dict):
        for key, value in self.keys_dict.items():
            if isinstance(data_dict[key], np.ndarray):
                data_dict[value] = data_dict[key].copy()
            elif isinstance(data_dict[key], torch.Tensor):
                data_dict[value] = data_dict[key].clone().detach()
            else:
                data_dict[value] = copy.deepcopy(data_dict[key])
        return data_dict


@TRANSFORMS.register_module()
class Update(object):
    def __init__(self, keys_dict=None):
        if keys_dict is None:
            keys_dict = dict()
        self.keys_dict = keys_dict

    def __call__(self, data_dict):
        for key, value in self.keys_dict.items():
            data_dict[key] = value
        return data_dict


@TRANSFORMS.register_module()
class ToTensor(object):
    def __call__(self, data):
        if isinstance(data, torch.Tensor):
            return data
        elif isinstance(data, str):
            # note that str is also a kind of sequence, judgement should before sequence
            return data
        elif isinstance(data, int):
            return torch.LongTensor([data])
        elif isinstance(data, float):
            return torch.FloatTensor([data])
        elif isinstance(data, np.ndarray) and np.issubdtype(data.dtype, bool):
            return torch.from_numpy(data)
        elif isinstance(data, np.ndarray) and np.issubdtype(data.dtype, np.integer):
            return torch.from_numpy(data).long()
        elif isinstance(data, np.ndarray) and np.issubdtype(data.dtype, np.floating):
            return torch.from_numpy(data).float()
        elif isinstance(data, Mapping):
            result = {sub_key: self(item) for sub_key, item in data.items()}
            return result
        elif isinstance(data, Sequence):
            result = [self(item) for item in data]
            return result
        else:
            raise TypeError(f"type {type(data)} cannot be converted to tensor.")


@TRANSFORMS.register_module()
class NormalizeColor(object):
    def __call__(self, data_dict):
        if "color" in data_dict.keys():
            data_dict["color"] = data_dict["color"] / 255
        return data_dict


@TRANSFORMS.register_module()
class NormalizeCoord(object):
    def __call__(self, data_dict):
        if "coord" in data_dict.keys():
            # modified from pointnet2
            centroid = np.mean(data_dict["coord"], axis=0)
            data_dict["coord"] -= centroid
            m = np.max(np.sqrt(np.sum(data_dict["coord"] ** 2, axis=1)))
            data_dict["coord"] = data_dict["coord"] / m
        return data_dict


@TRANSFORMS.register_module()
class PositiveShift(object):
    def __call__(self, data_dict):
        if "coord" in data_dict.keys():
            coord_min = np.min(data_dict["coord"], 0)
            data_dict["coord"] -= coord_min
        return data_dict


@TRANSFORMS.register_module()
class CenterShift(object):
    def __init__(self, apply_z=True):
        self.apply_z = apply_z

    def __call__(self, data_dict):
        if "coord" in data_dict.keys():
            x_min, y_min, z_min = data_dict["coord"].min(axis=0)
            x_max, y_max, _ = data_dict["coord"].max(axis=0)
            if self.apply_z:
                shift = [(x_min + x_max) / 2, (y_min + y_max) / 2, z_min]
            else:
                shift = [(x_min + x_max) / 2, (y_min + y_max) / 2, 0]
            data_dict["coord"] -= shift
        return data_dict


@TRANSFORMS.register_module()
class RandomShift(object):
    def __init__(self, shift=((-0.2, 0.2), (-0.2, 0.2), (0, 0))):
        self.shift = shift

    def __call__(self, data_dict):
        if "coord" in data_dict.keys():
            shift_x = np.random.uniform(self.shift[0][0], self.shift[0][1])
            shift_y = np.random.uniform(self.shift[1][0], self.shift[1][1])
            shift_z = np.random.uniform(self.shift[2][0], self.shift[2][1])
            data_dict["coord"] += [shift_x, shift_y, shift_z]
        return data_dict


@TRANSFORMS.register_module()
class PointClip(object):
    def __init__(self, point_cloud_range=(-80, -80, -3, 80, 80, 1)):
        self.point_cloud_range = point_cloud_range

    def __call__(self, data_dict):
        if "coord" in data_dict.keys():
            data_dict["coord"] = np.clip(
                data_dict["coord"],
                a_min=self.point_cloud_range[:3],
                a_max=self.point_cloud_range[3:],
            )
        return data_dict


@TRANSFORMS.register_module()
class RandomDropout(object):
    def __init__(self, dropout_ratio=0.2, dropout_application_ratio=0.5):
        """
        upright_axis: axis index among x,y,z, i.e. 2 for z
        """
        self.dropout_ratio = dropout_ratio
        self.dropout_application_ratio = dropout_application_ratio

    def __call__(self, data_dict):
        if random.random() < self.dropout_application_ratio:
            n = len(data_dict["coord"])
            idx = np.random.choice(n, int(n * (1 - self.dropout_ratio)), replace=False)
            if "sampled_index" in data_dict:
                # for ScanNet data efficient, we need to make sure labeled point is sampled.
                idx = np.unique(np.append(idx, data_dict["sampled_index"]))
                mask = np.zeros_like(data_dict["segment"]).astype(bool)
                mask[data_dict["sampled_index"]] = True
                data_dict["sampled_index"] = np.where(mask[idx])[0]
            data_dict = index_operator(data_dict, idx)
        return data_dict


@TRANSFORMS.register_module()
class RandomRotate(object):
    def __init__(self, angle=None, center=None, axis="z", always_apply=False, p=0.5):
        self.angle = [-1, 1] if angle is None else angle
        self.axis = axis
        self.always_apply = always_apply
        self.p = p if not self.always_apply else 1
        self.center = center

    def __call__(self, data_dict):
        if random.random() > self.p:
            return data_dict
        angle = np.random.uniform(self.angle[0], self.angle[1]) * np.pi
        rot_cos, rot_sin = np.cos(angle), np.sin(angle)
        if self.axis == "x":
            rot_t = np.array([[1, 0, 0], [0, rot_cos, -rot_sin], [0, rot_sin, rot_cos]])
        elif self.axis == "y":
            rot_t = np.array([[rot_cos, 0, rot_sin], [0, 1, 0], [-rot_sin, 0, rot_cos]])
        elif self.axis == "z":
            rot_t = np.array([[rot_cos, -rot_sin, 0], [rot_sin, rot_cos, 0], [0, 0, 1]])
        else:
            raise NotImplementedError
        if "coord" in data_dict.keys():
            if self.center is None:
                x_min, y_min, z_min = data_dict["coord"].min(axis=0)
                x_max, y_max, z_max = data_dict["coord"].max(axis=0)
                center = [(x_min + x_max) / 2, (y_min + y_max) / 2, (z_min + z_max) / 2]
            else:
                center = self.center
            data_dict["coord"] -= center
            data_dict["coord"] = np.dot(data_dict["coord"], np.transpose(rot_t))
            data_dict["coord"] += center
        if "normal" in data_dict.keys():
            data_dict["normal"] = np.dot(data_dict["normal"], np.transpose(rot_t))
        return data_dict


@TRANSFORMS.register_module()
class RandomRotateTargetAngle(object):
    def __init__(
        self, angle=(1 / 2, 1, 3 / 2), center=None, axis="z", always_apply=False, p=0.75
    ):
        self.angle = angle
        self.axis = axis
        self.always_apply = always_apply
        self.p = p if not self.always_apply else 1
        self.center = center

    def __call__(self, data_dict):
        if random.random() > self.p:
            return data_dict
        angle = np.random.choice(self.angle) * np.pi
        rot_cos, rot_sin = np.cos(angle), np.sin(angle)
        if self.axis == "x":
            rot_t = np.array([[1, 0, 0], [0, rot_cos, -rot_sin], [0, rot_sin, rot_cos]])
        elif self.axis == "y":
            rot_t = np.array([[rot_cos, 0, rot_sin], [0, 1, 0], [-rot_sin, 0, rot_cos]])
        elif self.axis == "z":
            rot_t = np.array([[rot_cos, -rot_sin, 0], [rot_sin, rot_cos, 0], [0, 0, 1]])
        else:
            raise NotImplementedError
        if "coord" in data_dict.keys():
            if self.center is None:
                x_min, y_min, z_min = data_dict["coord"].min(axis=0)
                x_max, y_max, z_max = data_dict["coord"].max(axis=0)
                center = [(x_min + x_max) / 2, (y_min + y_max) / 2, (z_min + z_max) / 2]
            else:
                center = self.center
            data_dict["coord"] -= center
            data_dict["coord"] = np.dot(data_dict["coord"], np.transpose(rot_t))
            data_dict["coord"] += center
        if "normal" in data_dict.keys():
            data_dict["normal"] = np.dot(data_dict["normal"], np.transpose(rot_t))
        return data_dict


@TRANSFORMS.register_module()
class RandomScale(object):
    def __init__(self, scale=None, anisotropic=False):
        self.scale = scale if scale is not None else [0.95, 1.05]
        self.anisotropic = anisotropic

    def __call__(self, data_dict):
        if "coord" in data_dict.keys():
            scale = np.random.uniform(
                self.scale[0], self.scale[1], 3 if self.anisotropic else 1
            )
            data_dict["coord"] *= scale
        return data_dict


@TRANSFORMS.register_module()
class RandomFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, data_dict):
        if np.random.rand() < self.p:
            if "coord" in data_dict.keys():
                data_dict["coord"][:, 0] = -data_dict["coord"][:, 0]
            if "normal" in data_dict.keys():
                data_dict["normal"][:, 0] = -data_dict["normal"][:, 0]
        if np.random.rand() < self.p:
            if "coord" in data_dict.keys():
                data_dict["coord"][:, 1] = -data_dict["coord"][:, 1]
            if "normal" in data_dict.keys():
                data_dict["normal"][:, 1] = -data_dict["normal"][:, 1]
        return data_dict


@TRANSFORMS.register_module()
class RandomJitter(object):
    def __init__(self, sigma=0.01, clip=0.05):
        assert clip > 0
        self.sigma = sigma
        self.clip = clip

    def __call__(self, data_dict):
        if "coord" in data_dict.keys():
            jitter = np.clip(
                self.sigma * np.random.randn(data_dict["coord"].shape[0], 3),
                -self.clip,
                self.clip,
            )
            data_dict["coord"] += jitter
        return data_dict


@TRANSFORMS.register_module()
class ClipGaussianJitter(object):
    def __init__(self, scalar=0.02, store_jitter=False):
        self.scalar = scalar
        self.mean = np.mean(3)
        self.cov = np.identity(3)
        self.quantile = 1.96
        self.store_jitter = store_jitter

    def __call__(self, data_dict):
        if "coord" in data_dict.keys():
            jitter = np.random.multivariate_normal(
                self.mean, self.cov, data_dict["coord"].shape[0]
            )
            jitter = self.scalar * np.clip(jitter / 1.96, -1, 1)
            data_dict["coord"] += jitter
            if self.store_jitter:
                data_dict["jitter"] = jitter
        return data_dict


@TRANSFORMS.register_module()
class ChromaticAutoContrast(object):
    def __init__(self, p=0.2, blend_factor=None):
        self.p = p
        self.blend_factor = blend_factor

    def __call__(self, data_dict):
        if "color" in data_dict.keys() and np.random.rand() < self.p:
            lo = np.min(data_dict["color"], 0, keepdims=True)
            hi = np.max(data_dict["color"], 0, keepdims=True)
            diff = hi - lo
            if not np.any(diff > 0):
                return data_dict
            scale = np.divide(
                255,
                diff,
                out=np.ones_like(diff, dtype=data_dict["color"].dtype),
                where=diff > 0,
            )
            contrast_feat = (data_dict["color"][:, :3] - lo) * scale
            blend_factor = (
                np.random.rand() if self.blend_factor is None else self.blend_factor
            )
            data_dict["color"][:, :3] = (1 - blend_factor) * data_dict["color"][
                :, :3
            ] + blend_factor * contrast_feat
        return data_dict


@TRANSFORMS.register_module()
class ChromaticTranslation(object):
    def __init__(self, p=0.95, ratio=0.05):
        self.p = p
        self.ratio = ratio

    def __call__(self, data_dict):
        if "color" in data_dict.keys() and np.random.rand() < self.p:
            tr = (np.random.rand(1, 3) - 0.5) * 255 * 2 * self.ratio
            data_dict["color"][:, :3] = np.clip(tr + data_dict["color"][:, :3], 0, 255)
        return data_dict


@TRANSFORMS.register_module()
class ChromaticJitter(object):
    def __init__(self, p=0.95, std=0.005):
        self.p = p
        self.std = std

    def __call__(self, data_dict):
        if "color" in data_dict.keys() and np.random.rand() < self.p:
            noise = np.random.randn(data_dict["color"].shape[0], 3)
            noise *= self.std * 255
            data_dict["color"][:, :3] = np.clip(
                noise + data_dict["color"][:, :3], 0, 255
            )
        return data_dict


@TRANSFORMS.register_module()
class RandomColorGrayScale(object):
    def __init__(self, p):
        self.p = p

    @staticmethod
    def rgb_to_grayscale(color, num_output_channels=1):
        if color.shape[-1] < 3:
            raise TypeError(
                "Input color should have at least 3 dimensions, but found {}".format(
                    color.shape[-1]
                )
            )

        if num_output_channels not in (1, 3):
            raise ValueError("num_output_channels should be either 1 or 3")

        r, g, b = color[..., 0], color[..., 1], color[..., 2]
        gray = (0.2989 * r + 0.587 * g + 0.114 * b).astype(color.dtype)
        gray = np.expand_dims(gray, axis=-1)

        if num_output_channels == 3:
            gray = np.broadcast_to(gray, color.shape)

        return gray

    def __call__(self, data_dict):
        if np.random.rand() < self.p:
            data_dict["color"] = self.rgb_to_grayscale(data_dict["color"], 3)
        return data_dict


@TRANSFORMS.register_module()
class RandomColorJitter(object):
    """
    Random Color Jitter for 3D point cloud (refer torchvision)
    """

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0, p=0.95):
        self.brightness = self._check_input(brightness, "brightness")
        self.contrast = self._check_input(contrast, "contrast")
        self.saturation = self._check_input(saturation, "saturation")
        self.hue = self._check_input(
            hue, "hue", center=0, bound=(-0.5, 0.5), clip_first_on_zero=False
        )
        self.p = p

    @staticmethod
    def _check_input(
        value, name, center=1, bound=(0, float("inf")), clip_first_on_zero=True
    ):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError(
                    "If {} is a single number, it must be non negative.".format(name)
                )
            value = [center - float(value), center + float(value)]
            if clip_first_on_zero:
                value[0] = max(value[0], 0.0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError("{} values should be between {}".format(name, bound))
        else:
            raise TypeError(
                "{} should be a single number or a list/tuple with length 2.".format(
                    name
                )
            )

        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            value = None
        return value

    @staticmethod
    def blend(color1, color2, ratio):
        ratio = float(ratio)
        bound = 255.0
        return (
            (ratio * color1 + (1.0 - ratio) * color2)
            .clip(0, bound)
            .astype(color1.dtype)
        )

    @staticmethod
    def rgb2hsv(rgb):
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        maxc = np.max(rgb, axis=-1)
        minc = np.min(rgb, axis=-1)
        eqc = maxc == minc
        cr = maxc - minc
        s = cr / (np.ones_like(maxc) * eqc + maxc * (1 - eqc))
        cr_divisor = np.ones_like(maxc) * eqc + cr * (1 - eqc)
        rc = (maxc - r) / cr_divisor
        gc = (maxc - g) / cr_divisor
        bc = (maxc - b) / cr_divisor

        hr = (maxc == r) * (bc - gc)
        hg = ((maxc == g) & (maxc != r)) * (2.0 + rc - bc)
        hb = ((maxc != g) & (maxc != r)) * (4.0 + gc - rc)
        h = hr + hg + hb
        h = (h / 6.0 + 1.0) % 1.0
        return np.stack((h, s, maxc), axis=-1)

    @staticmethod
    def hsv2rgb(hsv):
        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        i = np.floor(h * 6.0)
        f = (h * 6.0) - i
        i = i.astype(np.int32)

        p = np.clip((v * (1.0 - s)), 0.0, 1.0)
        q = np.clip((v * (1.0 - s * f)), 0.0, 1.0)
        t = np.clip((v * (1.0 - s * (1.0 - f))), 0.0, 1.0)
        i = i % 6
        mask = np.expand_dims(i, axis=-1) == np.arange(6)

        a1 = np.stack((v, q, p, p, t, v), axis=-1)
        a2 = np.stack((t, v, v, q, p, p), axis=-1)
        a3 = np.stack((p, p, t, v, v, q), axis=-1)
        a4 = np.stack((a1, a2, a3), axis=-1)

        return np.einsum("...na, ...nab -> ...nb", mask.astype(hsv.dtype), a4)

    def adjust_brightness(self, color, brightness_factor):
        if brightness_factor < 0:
            raise ValueError(
                "brightness_factor ({}) is not non-negative.".format(brightness_factor)
            )

        return self.blend(color, np.zeros_like(color), brightness_factor)

    def adjust_contrast(self, color, contrast_factor):
        if contrast_factor < 0:
            raise ValueError(
                "contrast_factor ({}) is not non-negative.".format(contrast_factor)
            )
        mean = np.mean(RandomColorGrayScale.rgb_to_grayscale(color))
        return self.blend(color, mean, contrast_factor)

    def adjust_saturation(self, color, saturation_factor):
        if saturation_factor < 0:
            raise ValueError(
                "saturation_factor ({}) is not non-negative.".format(saturation_factor)
            )
        gray = RandomColorGrayScale.rgb_to_grayscale(color)
        return self.blend(color, gray, saturation_factor)

    def adjust_hue(self, color, hue_factor):
        if not (-0.5 <= hue_factor <= 0.5):
            raise ValueError(
                "hue_factor ({}) is not in [-0.5, 0.5].".format(hue_factor)
            )
        orig_dtype = color.dtype
        hsv = self.rgb2hsv(color / 255.0)
        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        h = (h + hue_factor) % 1.0
        hsv = np.stack((h, s, v), axis=-1)
        color_hue_adj = (self.hsv2rgb(hsv) * 255.0).astype(orig_dtype)
        return color_hue_adj

    @staticmethod
    def get_params(brightness, contrast, saturation, hue):
        fn_idx = torch.randperm(4)
        b = (
            None
            if brightness is None
            else np.random.uniform(brightness[0], brightness[1])
        )
        c = None if contrast is None else np.random.uniform(contrast[0], contrast[1])
        s = (
            None
            if saturation is None
            else np.random.uniform(saturation[0], saturation[1])
        )
        h = None if hue is None else np.random.uniform(hue[0], hue[1])
        return fn_idx, b, c, s, h

    def __call__(self, data_dict):
        (
            fn_idx,
            brightness_factor,
            contrast_factor,
            saturation_factor,
            hue_factor,
        ) = self.get_params(self.brightness, self.contrast, self.saturation, self.hue)

        for fn_id in fn_idx:
            if (
                fn_id == 0
                and brightness_factor is not None
                and np.random.rand() < self.p
            ):
                data_dict["color"] = self.adjust_brightness(
                    data_dict["color"], brightness_factor
                )
            elif (
                fn_id == 1 and contrast_factor is not None and np.random.rand() < self.p
            ):
                data_dict["color"] = self.adjust_contrast(
                    data_dict["color"], contrast_factor
                )
            elif (
                fn_id == 2
                and saturation_factor is not None
                and np.random.rand() < self.p
            ):
                data_dict["color"] = self.adjust_saturation(
                    data_dict["color"], saturation_factor
                )
            elif fn_id == 3 and hue_factor is not None and np.random.rand() < self.p:
                data_dict["color"] = self.adjust_hue(data_dict["color"], hue_factor)
        return data_dict


@TRANSFORMS.register_module()
class HueSaturationTranslation(object):
    @staticmethod
    def rgb_to_hsv(rgb):
        # Translated from source of colorsys.rgb_to_hsv
        # r,g,b should be a numpy arrays with values between 0 and 255
        # rgb_to_hsv returns an array of floats between 0.0 and 1.0.
        rgb = rgb.astype("float")
        hsv = np.zeros_like(rgb)
        # in case an RGBA array was passed, just copy the A channel
        hsv[..., 3:] = rgb[..., 3:]
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        maxc = np.max(rgb[..., :3], axis=-1)
        minc = np.min(rgb[..., :3], axis=-1)
        hsv[..., 2] = maxc
        mask = maxc != minc
        hsv[mask, 1] = (maxc - minc)[mask] / maxc[mask]
        rc = np.zeros_like(r)
        gc = np.zeros_like(g)
        bc = np.zeros_like(b)
        rc[mask] = (maxc - r)[mask] / (maxc - minc)[mask]
        gc[mask] = (maxc - g)[mask] / (maxc - minc)[mask]
        bc[mask] = (maxc - b)[mask] / (maxc - minc)[mask]
        hsv[..., 0] = np.select(
            [r == maxc, g == maxc], [bc - gc, 2.0 + rc - bc], default=4.0 + gc - rc
        )
        hsv[..., 0] = (hsv[..., 0] / 6.0) % 1.0
        return hsv

    @staticmethod
    def hsv_to_rgb(hsv):
        # Translated from source of colorsys.hsv_to_rgb
        # h,s should be a numpy arrays with values between 0.0 and 1.0
        # v should be a numpy array with values between 0.0 and 255.0
        # hsv_to_rgb returns an array of uints between 0 and 255.
        rgb = np.empty_like(hsv)
        rgb[..., 3:] = hsv[..., 3:]
        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        i = (h * 6.0).astype("uint8")
        f = (h * 6.0) - i
        p = v * (1.0 - s)
        q = v * (1.0 - s * f)
        t = v * (1.0 - s * (1.0 - f))
        i = i % 6
        conditions = [s == 0.0, i == 1, i == 2, i == 3, i == 4, i == 5]
        rgb[..., 0] = np.select(conditions, [v, q, p, p, t, v], default=v)
        rgb[..., 1] = np.select(conditions, [v, v, v, q, p, p], default=t)
        rgb[..., 2] = np.select(conditions, [v, p, t, v, v, q], default=p)
        return rgb.astype("uint8")

    def __init__(self, hue_max=0.5, saturation_max=0.2):
        self.hue_max = hue_max
        self.saturation_max = saturation_max

    def __call__(self, data_dict):
        if "color" in data_dict.keys():
            # Assume color[:, :3] is rgb
            hsv = HueSaturationTranslation.rgb_to_hsv(data_dict["color"][:, :3])
            hue_val = (np.random.rand() - 0.5) * 2 * self.hue_max
            sat_ratio = 1 + (np.random.rand() - 0.5) * 2 * self.saturation_max
            hsv[..., 0] = np.remainder(hue_val + hsv[..., 0] + 1, 1)
            hsv[..., 1] = np.clip(sat_ratio * hsv[..., 1], 0, 1)
            data_dict["color"][:, :3] = np.clip(
                HueSaturationTranslation.hsv_to_rgb(hsv), 0, 255
            )
        return data_dict


@TRANSFORMS.register_module()
class RandomDropColor(object):
    def __init__(self, drop_ratio=0.2, drop_application_ratio=0.5):
        self.drop_ratio = drop_ratio
        self.drop_application_ratio = drop_application_ratio
        self.drop_value = 0.0

    def __call__(self, data_dict):
        if (
            "color" in data_dict.keys()
            and random.random() < self.drop_application_ratio
        ):
            n = len(data_dict["color"])
            idx = np.random.choice(n, int(n * self.drop_ratio), replace=False)
            data_dict["color"][idx] = self.drop_value
        return data_dict


@TRANSFORMS.register_module()
class RandomDropNormal(object):
    def __init__(self, drop_ratio=0.2, drop_application_ratio=0.5):
        self.drop_ratio = drop_ratio
        self.drop_application_ratio = drop_application_ratio
        self.drop_value = 0.0

    def __call__(self, data_dict):
        if (
            "normal" in data_dict.keys()
            and random.random() < self.drop_application_ratio
        ):
            n = len(data_dict["normal"])
            num_to_drop = int(n * self.drop_ratio)
            idx = np.random.choice(n, num_to_drop, replace=False)
            data_dict["normal"][idx] = self.drop_value
        return data_dict


@TRANSFORMS.register_module()
class RandomColorDrop(object):
    def __init__(self, p=0.2, color_augment=0.0):
        self.p = p
        self.color_augment = color_augment

    def __call__(self, data_dict):
        if "color" in data_dict.keys() and np.random.rand() < self.p:
            data_dict["color"] *= self.color_augment
        return data_dict

    def __repr__(self):
        return "RandomColorDrop(color_augment: {}, p: {})".format(
            self.color_augment, self.p
        )


@TRANSFORMS.register_module()
class ElasticDistortion(object):
    def __init__(self, distortion_params=None):
        self.distortion_params = (
            [[0.2, 0.4], [0.8, 1.6]] if distortion_params is None else distortion_params
        )

    @staticmethod
    def elastic_distortion(coords, granularity, magnitude):
        """
        Apply elastic distortion on sparse coordinate space.
        pointcloud: numpy array of (number of points, at least 3 spatial dims)
        granularity: size of the noise grid (in same scale[m/cm] as the voxel grid)
        magnitude: noise multiplier
        """
        blurx = np.ones((3, 1, 1, 1)).astype("float32") / 3
        blury = np.ones((1, 3, 1, 1)).astype("float32") / 3
        blurz = np.ones((1, 1, 3, 1)).astype("float32") / 3
        coords_min = coords.min(0)

        # Create Gaussian noise tensor of the size given by granularity.
        noise_dim = ((coords - coords_min).max(0) // granularity).astype(int) + 3
        noise = np.random.randn(*noise_dim, 3).astype(np.float32)

        # Smoothing.
        for _ in range(2):
            noise = scipy.ndimage.filters.convolve(
                noise, blurx, mode="constant", cval=0
            )
            noise = scipy.ndimage.filters.convolve(
                noise, blury, mode="constant", cval=0
            )
            noise = scipy.ndimage.filters.convolve(
                noise, blurz, mode="constant", cval=0
            )

        # Trilinear interpolate noise filters for each spatial dimensions.
        ax = [
            np.linspace(d_min, d_max, d)
            for d_min, d_max, d in zip(
                coords_min - granularity,
                coords_min + granularity * (noise_dim - 2),
                noise_dim,
            )
        ]
        interp = scipy.interpolate.RegularGridInterpolator(
            ax, noise, bounds_error=False, fill_value=0
        )
        coords += interp(coords) * magnitude
        return coords

    def __call__(self, data_dict):
        if "coord" in data_dict.keys() and self.distortion_params is not None:
            if random.random() < 0.95:
                for granularity, magnitude in self.distortion_params:
                    data_dict["coord"] = self.elastic_distortion(
                        data_dict["coord"], granularity, magnitude
                    )
        return data_dict


@TRANSFORMS.register_module()
class GridSample(object):
    def __init__(
        self,
        grid_size=0.05,
        hash_type="fnv",
        mode="train",
        return_inverse=False,
        return_grid_coord=False,
        return_min_coord=False,
        return_displacement=False,
        project_displacement=False,
    ):
        self.grid_size = grid_size
        self.hash = self.fnv_hash_vec if hash_type == "fnv" else self.ravel_hash_vec
        assert mode in ["train", "test"]
        self.mode = mode
        self.return_inverse = return_inverse
        self.return_grid_coord = return_grid_coord
        self.return_min_coord = return_min_coord
        self.return_displacement = return_displacement
        self.project_displacement = project_displacement

    def __call__(self, data_dict):
        assert "coord" in data_dict.keys()
        scaled_coord = data_dict["coord"] / np.array(self.grid_size)
        grid_coord = np.floor(scaled_coord).astype(int)
        min_coord = grid_coord.min(0)
        grid_coord -= min_coord
        scaled_coord -= min_coord
        min_coord = min_coord * np.array(self.grid_size)
        key = self.hash(grid_coord)
        idx_sort = np.argsort(key)
        key_sort = key[idx_sort]
        _, inverse, count = np.unique(key_sort, return_inverse=True, return_counts=True)
        if self.mode == "train":  # train mode
            idx_select = (
                np.cumsum(np.insert(count, 0, 0)[0:-1])
                + np.random.randint(0, count.max(), count.size) % count
            )
            idx_unique = idx_sort[idx_select]
            if "sampled_index" in data_dict:
                # for ScanNet data efficient, we need to make sure labeled point is sampled.
                idx_unique = np.unique(
                    np.append(idx_unique, data_dict["sampled_index"])
                )
                mask = np.zeros_like(data_dict["segment"]).astype(bool)
                mask[data_dict["sampled_index"]] = True
                data_dict["sampled_index"] = np.where(mask[idx_unique])[0]
            data_dict = index_operator(data_dict, idx_unique)
            if self.return_inverse:
                data_dict["inverse"] = np.zeros_like(inverse)
                data_dict["inverse"][idx_sort] = inverse
            if self.return_grid_coord:
                data_dict["grid_coord"] = grid_coord[idx_unique]
                if "grid_coord" not in data_dict["index_valid_keys"]:
                    data_dict["index_valid_keys"].append("grid_coord")
            if self.return_min_coord:
                data_dict["min_coord"] = min_coord.reshape([1, 3])
            if self.return_displacement:
                displacement = (
                    scaled_coord - grid_coord - 0.5
                )  # [0, 1] -> [-0.5, 0.5] displacement to center
                if self.project_displacement:
                    displacement = np.sum(
                        displacement * data_dict["normal"], axis=-1, keepdims=True
                    )
                data_dict["displacement"] = displacement[idx_unique]
                if "displacement" not in data_dict["index_valid_keys"]:
                    data_dict["index_valid_keys"].append("displacement")
            return data_dict

        elif self.mode == "test":  # test mode
            data_part_list = []
            for i in range(count.max()):
                idx_select = np.cumsum(np.insert(count, 0, 0)[0:-1]) + i % count
                idx_part = idx_sort[idx_select]
                data_part = index_operator(data_dict, idx_part, duplicate=True)
                data_part["index"] = idx_part
                if "index" not in data_part["index_valid_keys"]:
                    data_part["index_valid_keys"].append("index")
                if self.return_inverse:
                    data_part["inverse"] = np.zeros_like(inverse)
                    data_part["inverse"][idx_sort] = inverse
                if self.return_grid_coord:
                    data_part["grid_coord"] = grid_coord[idx_part]
                    if "grid_coord" not in data_part["index_valid_keys"]:
                        data_part["index_valid_keys"].append("grid_coord")
                if self.return_min_coord:
                    data_part["min_coord"] = min_coord.reshape([1, 3])
                if self.return_displacement:
                    displacement = (
                        scaled_coord - grid_coord - 0.5
                    )  # [0, 1] -> [-0.5, 0.5] displacement to center
                    if self.project_displacement:
                        displacement = np.sum(
                            displacement * data_dict["normal"], axis=-1, keepdims=True
                        )
                    data_part["displacement"] = displacement[idx_part]
                    if "displacement" not in data_part["index_valid_keys"]:
                        data_part["index_valid_keys"].append("displacement")
                data_part_list.append(data_part)
            return data_part_list
        else:
            raise NotImplementedError

    @staticmethod
    def ravel_hash_vec(arr):
        """
        Ravel the coordinates after subtracting the min coordinates.
        """
        assert arr.ndim == 2
        arr = arr.copy()
        arr -= arr.min(0)
        arr = arr.astype(np.uint64, copy=False)
        arr_max = arr.max(0).astype(np.uint64) + 1

        keys = np.zeros(arr.shape[0], dtype=np.uint64)
        # Fortran style indexing
        for j in range(arr.shape[1] - 1):
            keys += arr[:, j]
            keys *= arr_max[j + 1]
        keys += arr[:, -1]
        return keys

    @staticmethod
    def fnv_hash_vec(arr):
        """
        FNV64-1A
        """
        assert arr.ndim == 2
        # Floor first for negative coordinates
        arr = arr.copy()
        arr = arr.astype(np.uint64, copy=False)
        hashed_arr = np.uint64(14695981039346656037) * np.ones(
            arr.shape[0], dtype=np.uint64
        )
        for j in range(arr.shape[1]):
            hashed_arr *= np.uint64(1099511628211)
            hashed_arr = np.bitwise_xor(hashed_arr, arr[:, j])
        return hashed_arr


@TRANSFORMS.register_module()
class SphereCrop(object):
    def __init__(self, point_max=80000, sample_rate=None, mode="random"):
        self.point_max = point_max
        self.sample_rate = sample_rate
        assert mode in ["random", "center", "all", "given"]
        self.mode = mode

    def __call__(self, data_dict):
        point_max = (
            int(self.sample_rate * data_dict["coord"].shape[0])
            if self.sample_rate is not None
            else self.point_max
        )

        assert "coord" in data_dict.keys()
        if data_dict["coord"].shape[0] > point_max:
            if self.mode == "random":
                center = data_dict["coord"][
                    np.random.randint(data_dict["coord"].shape[0])
                ]
            elif self.mode == "center":
                center = data_dict["coord"][data_dict["coord"].shape[0] // 2]
            elif self.mode == "given":
                given_index = data_dict["correspondence"].reshape(
                    data_dict["correspondence"].shape[0], -1
                )
                given_index = np.all(
                    given_index != np.ones_like(given_index[0]) * -1, axis=1
                )
                given_coord = data_dict["coord"][given_index]
                if given_coord.shape[0] == 0:
                    center = data_dict["coord"][
                        np.random.randint(data_dict["coord"].shape[0])
                    ]
                else:
                    center = np.mean(given_coord, axis=0)
            else:
                raise NotImplementedError
            idx_crop = np.argsort(np.sum(np.square(data_dict["coord"] - center), 1))[
                :point_max
            ]
            data_dict = index_operator(data_dict, idx_crop)
        return data_dict


@TRANSFORMS.register_module()
class ShufflePoint(object):
    def __call__(self, data_dict):
        assert "coord" in data_dict.keys()
        shuffle_index = np.arange(data_dict["coord"].shape[0])
        np.random.shuffle(shuffle_index)
        data_dict = index_operator(data_dict, shuffle_index)
        return data_dict


@TRANSFORMS.register_module()
class CropBoundary(object):
    def __call__(self, data_dict):
        assert "segment" in data_dict
        segment = data_dict["segment"].flatten()
        mask = (segment != 0) * (segment != 1)
        data_dict = index_operator(data_dict, mask)
        return data_dict


@TRANSFORMS.register_module()
class ContrastiveViewsGenerator(object):
    def __init__(
        self,
        view_keys=("coord", "color", "normal", "origin_coord"),
        view_trans_cfg=None,
    ):
        self.view_keys = view_keys
        self.view_trans = Compose(view_trans_cfg)

    def __call__(self, data_dict):
        view1_dict = dict()
        view2_dict = dict()
        for key in self.view_keys:
            view1_dict[key] = data_dict[key].copy()
            view2_dict[key] = data_dict[key].copy()
        view1_dict = self.view_trans(view1_dict)
        view2_dict = self.view_trans(view2_dict)
        for key, value in view1_dict.items():
            data_dict["view1_" + key] = value
        for key, value in view2_dict.items():
            data_dict["view2_" + key] = value
        return data_dict


@TRANSFORMS.register_module()
class MultiViewGenerator(object):
    def __init__(
        self,
        global_view_num=2,
        global_view_scale=(0.4, 1.0),
        local_view_num=4,
        local_view_scale=(0.1, 0.4),
        global_shared_transform=None,
        global_transform=None,
        local_transform=None,
        max_size=65536,
        enc2d_max_size=102400,
        enc2d_scale=(0.8, 1),
        center_height_scale=(0, 1),
        shared_global_view=False,
        view_keys=("coord", "origin_coord", "color", "normal", "correspondence"),
        static_view_keys=("name", "img_num"),
    ):
        self.global_view_num = global_view_num
        self.global_view_scale = global_view_scale
        self.local_view_num = local_view_num
        self.local_view_scale = local_view_scale
        self.global_shared_transform = Compose(global_shared_transform)
        self.global_transform = Compose(global_transform)
        self.local_transform = Compose(local_transform)
        self.max_size = max_size
        self.enc2d_max_size = enc2d_max_size
        self.enc2d_scale = enc2d_scale
        self.center_height_scale = center_height_scale
        self.shared_global_view = shared_global_view
        self.view_keys = view_keys
        self.static_view_keys = static_view_keys
        assert "coord" in view_keys

    def get_view(self, point, center, scale, if_enc2d=False):
        coord = point["coord"]
        max_size = min(self.max_size, coord.shape[0])
        enc2d_max_size = min(self.enc2d_max_size, coord.shape[0])
        size = 0
        for _ in range(10):
            if if_enc2d:
                size = enc2d_max_size
            else:
                size = int(np.random.uniform(*scale) * max_size)
            if size > 0:
                break
        if size == 0:
            size = max(10, scale[-1] * max_size)
        assert size > 0
        index = np.argsort(np.sum(np.square(coord - center), axis=-1))[:size]
        view = dict(index=index)
        for key in point.keys():
            if key in self.view_keys:
                view[key] = point[key][index]
            if key in self.static_view_keys:
                view[key] = point[key]
        if "index_valid_keys" in point.keys():
            # inherit index_valid_keys from point
            view["index_valid_keys"] = point["index_valid_keys"]
        return view

    @staticmethod
    def match_point_image(major_view, data_dict):
        major_correspondence = major_view["correspondence"].transpose(1, 0, 2)
        correspondence = data_dict["correspondence"].transpose(1, 0, 2)
        is_all_neg1 = np.any(major_correspondence != np.array([-1, -1]), axis=(1, 2))
        indices = np.where(is_all_neg1)[0]
        img_dict = {
            "images": data_dict["images"][indices],
            "img_num": indices.shape[0],
            "major_correspondence": major_correspondence[indices].transpose(1, 0, 2),
            "correspondence": correspondence[indices].transpose(1, 0, 2),
        }
        return img_dict

    def __call__(self, data_dict):
        coord = data_dict["coord"]
        point = self.global_shared_transform(copy.deepcopy(data_dict))
        z_min = coord[:, 2].min()
        z_max = coord[:, 2].max()
        z_min_ = z_min + (z_max - z_min) * self.center_height_scale[0]
        z_max_ = z_min + (z_max - z_min) * self.center_height_scale[1]
        if "correspondence" not in data_dict.keys():
            center_mask = np.logical_and(coord[:, 2] >= z_min_, coord[:, 2] <= z_max_)
            major_center = coord[np.random.choice(np.where(center_mask)[0])]
            major_view = self.get_view(point, major_center, self.global_view_scale)
        else:
            given_index = data_dict["correspondence"].reshape(
                data_dict["correspondence"].shape[0], -1
            )
            given_index = np.all(
                given_index != np.ones_like(given_index[0]) * -1, axis=1
            )
            given_coord = data_dict["coord"][given_index]
            if given_coord.shape[0] == 0:
                center_mask = np.logical_and(
                    coord[:, 2] >= z_min_, coord[:, 2] <= z_max_
                )
                major_center = coord[np.random.choice(np.where(center_mask)[0])]
            else:
                major_center = np.mean(given_coord, axis=0)
            major_view = self.get_view(
                point, major_center, self.global_view_scale, if_enc2d=True
            )
            img_dict = self.match_point_image(major_view, data_dict)
            major_view["correspondence"] = img_dict["major_correspondence"]
            data_dict["correspondence"] = img_dict["correspondence"]
            point["correspondence"] = img_dict["correspondence"]
            data_dict["img_num"] = img_dict["img_num"]
            data_dict["images"] = img_dict["images"]
        major_coord = major_view["coord"]

        # get global views: restrict the center of left global view within the major global view
        if not self.shared_global_view:
            global_views = [
                self.get_view(
                    point=point,
                    center=major_coord[np.random.randint(major_coord.shape[0])],
                    scale=self.global_view_scale,
                )
                for _ in range(self.global_view_num - 1)
            ]
        else:
            global_views = [
                {key: value.copy() for key, value in major_view.items()}
                for _ in range(self.global_view_num - 1)
            ]

        global_views = [major_view] + global_views

        # get local views: restrict the center of local view within the major global view
        cover_mask = np.zeros_like(major_view["index"], dtype=bool)
        local_views = []
        for i in range(self.local_view_num):
            if sum(~cover_mask) == 0:
                # reset cover mask if all points are sampled
                cover_mask[:] = False
            local_view = self.get_view(
                point=data_dict,
                center=major_coord[np.random.choice(np.where(~cover_mask)[0])],
                scale=self.local_view_scale,
            )
            local_views.append(local_view)
            cover_mask[np.isin(major_view["index"], local_view["index"])] = True

        # augmentation and concat
        view_dict = {}
        for global_view in global_views:
            global_view.pop("index")
            global_view = self.global_transform(global_view)
            for key in self.view_keys:
                if f"global_{key}" in view_dict.keys():
                    view_dict[f"global_{key}"].append(global_view[key])
                else:
                    view_dict[f"global_{key}"] = [global_view[key]]
        view_dict["global_offset"] = np.cumsum(
            [data.shape[0] for data in view_dict["global_coord"]]
        )
        for local_view in local_views:
            local_view.pop("index")
            local_view = self.local_transform(local_view)
            for key in self.view_keys:
                if f"local_{key}" in view_dict.keys():
                    view_dict[f"local_{key}"].append(local_view[key])
                else:
                    view_dict[f"local_{key}"] = [local_view[key]]
        view_dict["local_offset"] = np.cumsum(
            [data.shape[0] for data in view_dict["local_coord"]]
        )

        for key in view_dict.keys():
            if "offset" not in key:
                if key in self.static_view_keys:
                    view_dict[key] = view_dict[key]
                else:
                    view_dict[key] = np.concatenate(view_dict[key], axis=0)
        data_dict.update(view_dict)
        return data_dict


@TRANSFORMS.register_module()
class InstanceParser(object):
    def __init__(self, segment_ignore_index=(-1, 0, 1), instance_ignore_index=-1):
        self.segment_ignore_index = segment_ignore_index
        self.instance_ignore_index = instance_ignore_index

    def __call__(self, data_dict):
        coord = data_dict["coord"]
        segment = data_dict["segment"]
        instance = data_dict["instance"]
        mask = ~np.in1d(segment, self.segment_ignore_index)
        # mapping ignored instance to ignore index
        instance[~mask] = self.instance_ignore_index
        # reorder left instance
        unique, inverse = np.unique(instance[mask], return_inverse=True)
        instance_num = len(unique)
        instance[mask] = inverse
        # init instance information
        centroid = np.ones((coord.shape[0], 3)) * self.instance_ignore_index
        bbox = np.ones((instance_num, 8)) * self.instance_ignore_index
        vacancy = [
            index for index in self.segment_ignore_index if index >= 0
        ]  # vacate class index

        for instance_id in range(instance_num):
            mask_ = instance == instance_id
            coord_ = coord[mask_]
            bbox_min = coord_.min(0)
            bbox_max = coord_.max(0)
            bbox_centroid = coord_.mean(0)
            bbox_center = (bbox_max + bbox_min) / 2
            bbox_size = bbox_max - bbox_min
            bbox_theta = np.zeros(1, dtype=coord_.dtype)
            bbox_class = np.array([segment[mask_][0]], dtype=coord_.dtype)
            # shift class index to fill vacate class index caused by segment ignore index
            bbox_class -= np.greater(bbox_class, vacancy).sum()

            centroid[mask_] = bbox_centroid
            bbox[instance_id] = np.concatenate(
                [bbox_center, bbox_size, bbox_theta, bbox_class]
            )  # 3 + 3 + 1 + 1 = 8
        data_dict["instance"] = instance
        data_dict["instance_centroid"] = centroid
        data_dict["bbox"] = bbox
        return data_dict


class Compose(object):
    def __init__(self, cfg=None):
        self.cfg = cfg if cfg is not None else []
        self.transforms = []
        for t_cfg in self.cfg:
            self.transforms.append(TRANSFORMS.build(t_cfg))

    def __call__(self, data_dict):
        for t in self.transforms:
            data_dict = t(data_dict)
        return data_dict


@TRANSFORMS.register_module()
class ImgToTensor(object):
    def __init__(self):
        self.totensor = transforms.ToTensor()

    def __call__(self, img):
        return self.totensor(img)


@TRANSFORMS.register_module()
class ImgGaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image.
    """

    def __init__(
        self, *, p: float = 0.5, radius_min: float = 0.1, radius_max: float = 2.0
    ):
        # NOTE: torchvision is applying 1 - probability to return the original image
        self.p = p
        self.transform = transforms.GaussianBlur(
            kernel_size=9, sigma=(radius_min, radius_max)
        )
        super().__init__()

    def __call__(self, img):
        if np.random.rand() < self.p:
            img = self.transform(img)
        return img


@TRANSFORMS.register_module()
class ImgChromaticJitter(object):
    def __init__(self, p=0.95, std=0.005):
        self.p = p
        self.std = std

    def __call__(self, img):
        if np.random.rand() < self.p:
            noise = torch.rand(3)
            noise *= self.std
            noise = noise[:, None, None].expand_as(img)
            img += noise
            img = torch.clip(img, 0, 1)
        return img


@TRANSFORMS.register_module()
class ImgPixelContrast(object):
    def __init__(self, threshold, p=0.2):
        super().__init__()
        self.p = p
        self.threshold = threshold

    def __call__(self, img):
        if np.random.rand() < self.p:
            n, h, w = img.shape[0], img.shape[2], img.shape[3]
            num_pixels = int(self.threshold * h * w * n)
            indices = torch.randint(0, n * h * w, (num_pixels,))
            img = img.permute(0, 2, 3, 1).reshape(-1, 3)
            img[indices, :] = 255.0 - img[indices, :]
            img = img.reshape(n, h, w, 3).permute(0, 3, 1, 2)
        return img


IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


@TRANSFORMS.register_module()
class Imgnormalize(object):
    def __init__(self, mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD):
        super().__init__()
        self.normalize = transforms.Normalize(mean=mean, std=std)

    def __call__(self, img):
        return self.normalize(img)


@TRANSFORMS.register_module()
class ImgRandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
        self.imgrandomhorizontalflip = transforms.RandomHorizontalFlip(p=p)

    def __call__(self, img):
        return self.imgrandomhorizontalflip(img)


@TRANSFORMS.register_module()
class ImgRandomResizedCrop(object):
    def __init__(self, size, scale, interpolation):
        super().__init__()
        self.imgrandomresizedcrop = transforms.RandomResizedCrop(
            size=size, scale=scale, interpolation=interpolation
        )

    def __call__(self, img):
        return self.imgrandomresizedcrop(img)


@TRANSFORMS.register_module()
class ImgRandomColorJitter(object):
    def __init__(self, brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8):
        colorjitter = transforms.ColorJitter(
            brightness=brightness, contrast=contrast, saturation=saturation, hue=hue
        )
        super().__init__()
        self.p = p
        self.colorjitter = colorjitter

    def __call__(self, img):
        return self.colorjitter(img)


@TRANSFORMS.register_module()
class ImgRandomGrayscale(object):
    def __init__(self, p=0.1):
        super().__init__()
        self.p = p
        self.imgrandomgrayscale = transforms.RandomGrayscale(p=p)

    def __call__(self, img):
        return self.imgrandomgrayscale(img)


@TRANSFORMS.register_module()
class ImgRandomSolarize(object):
    def __init__(self, threshold, p=0.1):
        super().__init__()
        self.p = p
        self.imgrandomsolarize = transforms.RandomSolarize(threshold=threshold, p=p)

    def __call__(self, img):
        return self.imgrandomsolarize(img)


@TRANSFORMS.register_module()
class ImgAugmentation(object):
    def __init__(
        self,
        imgtransforms,
        crop_h=518,
        crop_w=518,
        patch_h=37,
        patch_w=37,
        patch_size=14,
    ):
        self.transforms = []
        self.transforms_cfg = imgtransforms
        for t_cfg in self.transforms_cfg:
            self.transforms.append(TRANSFORMS.build(t_cfg))
        self.crop_h = crop_h
        self.crop_w = crop_w
        self.patch_h = patch_h
        self.patch_w = patch_w
        self.patch_size = patch_size
        self.crop_start = [
            random.randint(0, patch_h * patch_size - crop_h),
            random.randint(0, patch_w * patch_size - crop_w),
        ]

    def __call__(self, point):
        point["images"] = transforms.functional.crop(
            point["images"],
            top=self.crop_start[0],
            left=self.crop_start[1],
            height=self.crop_h,
            width=self.crop_w,
        )
        for id, t in enumerate(self.transforms):
            point["images"] = t(point["images"])
        correspondence = point["correspondence"]
        correspondence_shape = correspondence.shape
        correspondence = correspondence.reshape(-1, 2)
        mask = (
            (self.crop_start[0] <= correspondence[:, 0])
            & (correspondence[:, 0] < self.crop_start[0] + self.crop_h)
            & (self.crop_start[1] <= correspondence[:, 1])
            & (correspondence[:, 1] < self.crop_start[1] + self.crop_w)
        )
        correspondence[~mask] = np.array([-1, -1])
        correspondence[mask] -= np.array(self.crop_start)
        point["correspondence"] = correspondence.reshape(correspondence_shape)
        return point


@TRANSFORMS.register_module()
class OctreeMaskPoints(object):
    """
    Octree-based multi-level point masking as data augmentation.

    Builds a lightweight octree from grid coordinates and applies cascading
    masking at specified levels. Points belonging to masked nodes are removed.

    This transform should be applied AFTER GridSample (which provides grid_coord)
    and BEFORE SphereCrop.

    Example config:
        dict(type="OctreeMaskPoints",
             mask_levels=[(1, 0.5), (2, 0.5)],  # 50% at L1, then 50% of remaining at L2
             num_levels=5)

    With mask_levels=[(1, 0.5), (2, 0.5)]:
        - Level 1: mask 50% of nodes
        - Level 2: mask 50% of remaining unmasked nodes
        - Result: ~25% of points remain (0.5 * 0.5)
    """

    def __init__(
        self,
        mask_levels=None,
        num_levels=5,
        min_unmasked=1,
    ):
        """
        Args:
            mask_levels: List of (level, ratio) tuples for cascading masking.
                        Each ratio is applied to remaining unmasked nodes at that level.
            num_levels: Number of octree levels (0=coarsest, num_levels-1=finest)
            min_unmasked: Minimum nodes to keep unmasked per level
        """
        self.mask_levels = mask_levels if mask_levels is not None else []
        self.num_levels = num_levels
        self.min_unmasked = min_unmasked

    def _xyz_to_morton(self, x, y, z):
        """Convert xyz coordinates to Morton code (Z-order curve)."""
        def spread_bits(v):
            # Spread bits for 16-bit values
            v = v.astype(np.uint64)
            v = (v | (v << 16)) & 0x0000FFFF0000FFFF
            v = (v | (v << 8)) & 0x00FF00FF00FF00FF
            v = (v | (v << 4)) & 0x0F0F0F0F0F0F0F0F
            v = (v | (v << 2)) & 0x3333333333333333
            v = (v | (v << 1)) & 0x5555555555555555
            return v

        return spread_bits(x) | (spread_bits(y) << 1) | (spread_bits(z) << 2)

    def __call__(self, data_dict):
        if not self.mask_levels:
            return data_dict

        assert "grid_coord" in data_dict, \
            "OctreeMaskPoints requires 'grid_coord'. Place it after GridSample transform."

        grid_coord = data_dict["grid_coord"]
        N = grid_coord.shape[0]

        if N == 0:
            return data_dict

        # Ensure grid_coord is positive
        grid_min = grid_coord.min(axis=0)
        grid_coord_pos = grid_coord - grid_min

        # Build node assignments for each level
        # Level 0 = coarsest (largest nodes), Level num_levels-1 = finest
        finest_level = self.num_levels - 1

        # Compute scale factors: finest level = scale 1, coarser levels = higher scales
        # level_scale[l] = 2^(finest_level - l)
        level_scales = [2 ** (finest_level - l) for l in range(self.num_levels)]

        # Compute Morton codes at each level
        level_morton = []
        level_unique_nodes = []
        level_point_to_node = []

        for level in range(self.num_levels):
            scale = level_scales[level]
            # Grid coordinates at this level
            grid_level = (grid_coord_pos // scale).astype(np.int32)

            # Morton code for each point at this level
            morton = self._xyz_to_morton(
                grid_level[:, 0].astype(np.uint64),
                grid_level[:, 1].astype(np.uint64),
                grid_level[:, 2].astype(np.uint64)
            )

            # Find unique nodes
            unique_codes, inverse = np.unique(morton, return_inverse=True)

            level_morton.append(morton)
            level_unique_nodes.append(unique_codes)
            level_point_to_node.append(inverse)

        # Initialize node masks (False = unmasked, True = masked)
        node_masks = [np.zeros(len(codes), dtype=bool) for codes in level_unique_nodes]

        """# Debug: collect all output first, then print atomically
        debug_lines = []
        debug_lines.append(f"\n[DEBUG OctreeMaskPoints] mask_levels={self.mask_levels}")
        debug_lines.append(f"[DEBUG] Initial state ({N} points):")
        for level in range(self.num_levels):
            node_size = 0.02 * level_scales[level]
            debug_lines.append(f"  Level {level}: {len(level_unique_nodes[level])} nodes (node_size={node_size:.2f}m)")
        """
        # Sort mask_levels by level (coarse to fine)
        sorted_mask_levels = sorted(self.mask_levels, key=lambda x: x[0])

        # Apply masking at each specified level
        for mask_level, mask_ratio in sorted_mask_levels:
            if mask_level < 0 or mask_level >= self.num_levels:
                continue
            if mask_ratio <= 0:
                continue

            num_nodes = len(level_unique_nodes[mask_level])
            current_mask = node_masks[mask_level]

            # Find unmasked nodes (candidates for masking)
            unmasked_indices = np.where(~current_mask)[0]
            num_unmasked = len(unmasked_indices)

            if num_unmasked <= self.min_unmasked:
                continue

            # Calculate how many to mask
            num_to_mask = int(num_unmasked * mask_ratio)
            max_maskable = num_unmasked - self.min_unmasked
            num_to_mask = min(num_to_mask, max_maskable)
            num_to_mask = max(num_to_mask, 0)

            if num_to_mask > 0:
                # Randomly select nodes to mask
                perm = np.random.permutation(num_unmasked)
                nodes_to_mask = unmasked_indices[perm[:num_to_mask]]
                current_mask[nodes_to_mask] = True
                node_masks[mask_level] = current_mask

            # Propagate mask to finer levels (children inherit parent's mask)
            for level in range(mask_level + 1, self.num_levels):
                child_scale = level_scales[level]
                parent_scale = level_scales[level - 1]
                scale_ratio = parent_scale // child_scale  # Should be 2

                # For each child node, find its parent
                # Parent grid = child grid // 2
                child_grid = []
                for code in level_unique_nodes[level]:
                    # Decode Morton code back to xyz (simplified)
                    # This is approximate but works for our purpose
                    pass

                # Simpler approach: use point-to-node mapping
                # A child node is masked if any of its points' parent node is masked
                parent_point_to_node = level_point_to_node[level - 1]
                child_point_to_node = level_point_to_node[level]
                parent_mask = node_masks[level - 1]

                # For each point, check if its parent node is masked
                point_parent_masked = parent_mask[parent_point_to_node]

                # For each child node, check if all its points have masked parents
                child_mask = node_masks[level].copy()
                num_child_nodes = len(level_unique_nodes[level])

                for child_idx in range(num_child_nodes):
                    point_indices = np.where(child_point_to_node == child_idx)[0]
                    if len(point_indices) > 0:
                        # Child is masked if ANY of its points has masked parent
                        if point_parent_masked[point_indices].any():
                            child_mask[child_idx] = True

                node_masks[level] = child_mask

        # Compute point-level mask from finest level
        finest_mask = node_masks[finest_level]
        point_to_node_finest = level_point_to_node[finest_level]
        points_masked = finest_mask[point_to_node_finest]

        """# Debug: collect after masking info and print all at once
        debug_lines.append(f"[DEBUG] After masking:")
        for level in range(self.num_levels):
            total = len(level_unique_nodes[level])
            masked = node_masks[level].sum()
            remaining = total - masked
            pct = 100 * remaining / total if total > 0 else 0
            node_size = 0.02 * level_scales[level]
            debug_lines.append(f"  Level {level}: {remaining}/{total} nodes remaining ({pct:.1f}%) (node_size={node_size:.2f}m)")

        points_remaining = (~points_masked).sum()
        debug_lines.append(f"[DEBUG] Points: {points_remaining}/{N} remaining ({100*points_remaining/N:.1f}%)")

        # Print all at once (atomic)
        print("\n".join(debug_lines) + "\n")
        """

        # Keep only unmasked points
        keep_indices = np.where(~points_masked)[0]

        if len(keep_indices) == 0:
            # Edge case: all points masked, keep at least some
            keep_indices = np.arange(min(self.min_unmasked, N))

        # Apply index selection to all relevant keys
        data_dict = index_operator(data_dict, keep_indices)

        return data_dict


@TRANSFORMS.register_module()
class OctreeTopologyAugmentation(object):
    """
    Topology-preserving augmentation for octree-based models.

    Applies cube symmetry transformations (rotations and reflections) to grid_coord,
    which effectively permutes the octant ordering in the resulting octree while
    preserving the topology (parent-child relationships and sibling occupancy patterns).

    The cube symmetry group has 48 elements:
    - 24 proper rotations (orientation-preserving)
    - 24 improper rotations (rotations + reflection)

    Each transformation remaps octant IDs consistently throughout the tree.
    For example, a 90° rotation around Z maps:
    - octant 0 (0,0,0) → octant 2 (0,1,0)
    - octant 1 (1,0,0) → octant 0 (0,0,0)
    - etc.

    This transform should be applied AFTER GridSample (which provides grid_coord).

    Example config:
        dict(type="OctreeTopologyAugmentation",
             transforms="all",  # or "rotations", "reflections", "identity"
             prob=0.5)
    """

    # All 48 cube symmetry transformations
    # Each is (permutation, signs) where:
    # - permutation: which axis to take for (new_x, new_y, new_z)
    # - signs: whether to flip each axis (1 = no flip, -1 = flip)
    CUBE_SYMMETRIES = []

    @staticmethod
    def _generate_cube_symmetries():
        """Generate all 48 symmetry operations of the cube."""
        if OctreeTopologyAugmentation.CUBE_SYMMETRIES:
            return OctreeTopologyAugmentation.CUBE_SYMMETRIES

        from itertools import permutations, product

        symmetries = []

        # All axis permutations: 6 options
        for perm in permutations([0, 1, 2]):
            # All sign combinations: 8 options
            for signs in product([1, -1], repeat=3):
                # Check if this is a valid rotation/reflection
                # (determinant should be +1 for rotation, -1 for reflection)
                # All 48 combinations are valid cube symmetries
                symmetries.append((perm, signs))

        OctreeTopologyAugmentation.CUBE_SYMMETRIES = symmetries
        return symmetries

    # Pre-defined subsets of transformations
    TRANSFORM_SUBSETS = {
        "identity": [(((0, 1, 2), (1, 1, 1)))],
        "rotations_z": [
            ((0, 1, 2), (1, 1, 1)),    # 0°
            ((1, 0, 2), (-1, 1, 1)),   # 90° CCW
            ((0, 1, 2), (-1, -1, 1)),  # 180°
            ((1, 0, 2), (1, -1, 1)),   # 270° CCW
        ],
        "rotations_90": None,  # Will be computed
        "reflections": [
            ((0, 1, 2), (-1, 1, 1)),   # Reflect X
            ((0, 1, 2), (1, -1, 1)),   # Reflect Y
            ((0, 1, 2), (1, 1, -1)),   # Reflect Z
        ],
        "axis_flips": [
            ((0, 1, 2), (1, 1, 1)),    # Identity
            ((0, 1, 2), (-1, 1, 1)),   # Flip X
            ((0, 1, 2), (1, -1, 1)),   # Flip Y
            ((0, 1, 2), (1, 1, -1)),   # Flip Z
            ((0, 1, 2), (-1, -1, 1)),  # Flip XY
            ((0, 1, 2), (-1, 1, -1)),  # Flip XZ
            ((0, 1, 2), (1, -1, -1)),  # Flip YZ
            ((0, 1, 2), (-1, -1, -1)), # Flip XYZ
        ],
        "all": None,  # Will use full CUBE_SYMMETRIES
    }

    def __init__(
        self,
        transforms="all",
        prob=1.0,
        apply_to_coord=True,
        apply_to_normal=True,
    ):
        """
        Args:
            transforms: Which transformations to use. Options:
                - "all": All 48 cube symmetries
                - "identity": No transformation (useful for ablation)
                - "rotations_z": 4 rotations around Z axis (0°, 90°, 180°, 270°)
                - "reflections": 3 axis reflections
                - "axis_flips": 8 combinations of axis flips
                - List of (permutation, signs) tuples for custom transforms
            prob: Probability of applying augmentation (0-1)
            apply_to_coord: Also transform the continuous coordinates
            apply_to_normal: Also transform surface normals
        """
        self.prob = prob
        self.apply_to_coord = apply_to_coord
        self.apply_to_normal = apply_to_normal

        # Get the list of transformations to use
        if isinstance(transforms, str):
            if transforms == "all":
                self.transforms = self._generate_cube_symmetries()
            elif transforms in self.TRANSFORM_SUBSETS:
                self.transforms = self.TRANSFORM_SUBSETS[transforms]
            else:
                raise ValueError(f"Unknown transform set: {transforms}")
        else:
            # Custom list of transforms
            self.transforms = transforms

        # Ensure transforms is not None
        if self.transforms is None:
            self.transforms = self._generate_cube_symmetries()

    def _apply_transform(self, coords, perm, signs, max_coords):
        """
        Apply a cube symmetry transformation to coordinates.

        Args:
            coords: [N, 3] coordinates (can be grid_coord or continuous coord)
            perm: (p0, p1, p2) - axis permutation
            signs: (s0, s1, s2) - axis signs (+1 or -1)
            max_coords: [3] max coordinate values for flipping

        Returns:
            [N, 3] transformed coordinates
        """
        # Permute axes
        new_coords = coords[:, perm].copy()

        # Apply flips (for negative sign, flip around center)
        for i in range(3):
            if signs[i] < 0:
                # Flip: new = max - old
                new_coords[:, i] = max_coords[perm[i]] - new_coords[:, i]

        return new_coords

    def _apply_transform_to_vectors(self, vectors, perm, signs):
        """
        Apply transformation to vectors (like normals) - no translation needed.

        Args:
            vectors: [N, 3] vectors
            perm: axis permutation
            signs: axis signs

        Returns:
            [N, 3] transformed vectors
        """
        # Permute axes
        new_vectors = vectors[:, perm].copy()

        # Apply sign flips
        for i in range(3):
            if signs[i] < 0:
                new_vectors[:, i] = -new_vectors[:, i]

        return new_vectors

    def __call__(self, data_dict):
        # Check probability
        if random.random() > self.prob:
            return data_dict

        # Need grid_coord for octree topology transformation
        if "grid_coord" not in data_dict:
            return data_dict

        # Randomly select a transformation
        transform_idx = random.randint(0, len(self.transforms) - 1)
        perm, signs = self.transforms[transform_idx]

        grid_coord = data_dict["grid_coord"]

        # Compute max coordinates for flipping
        grid_max = grid_coord.max(axis=0)

        # Transform grid_coord
        data_dict["grid_coord"] = self._apply_transform(
            grid_coord, perm, signs, grid_max
        )

        # Optionally transform continuous coordinates
        if self.apply_to_coord and "coord" in data_dict:
            coord = data_dict["coord"]
            coord_max = coord.max(axis=0)
            data_dict["coord"] = self._apply_transform(
                coord, perm, signs, coord_max
            )

        # Optionally transform normals
        if self.apply_to_normal and "normal" in data_dict:
            data_dict["normal"] = self._apply_transform_to_vectors(
                data_dict["normal"], perm, signs
            )

        # Store which transform was applied (for debugging/analysis)
        data_dict["topology_transform"] = (perm, signs)

        return data_dict
