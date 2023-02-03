#!/usr/bin/python3

import cv2
import imageio
import numpy as np
import os
import os.path
import pdb
from torch.utils.data import Dataset
from typing import List, Tuple
from .builder import DATASETS
from .mseg_transform import *
from mseg.utils.dataset_config import infos
from mseg.utils.names_utils import get_universal_class_names
from mmcv.parallel import DataContainer as DC

"""
Could duplicate samples here to reduce overhead between epochs.
"""

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']


def is_image_file(filename):
    filename_lower = filename.lower()
    return any(filename_lower.endswith(extension) for extension in IMG_EXTENSIONS)


def get_imagenet_mean_std() -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    """ See use here in Pytorch ImageNet script:
        https://github.com/pytorch/examples/blob/master/imagenet/main.py#L197

        Returns:
        -   mean: Tuple[float,float,float],
        -   std: Tuple[float,float,float] = None
    """
    value_scale = 255
    mean = [0.485, 0.456, 0.406]
    mean = [item * value_scale for item in mean]
    std = [0.229, 0.224, 0.225]
    std = [item * value_scale for item in std]
    return mean, std


def get_dataset_split_transform(
        args, split: str
) -> Compose:
    """Return the input data transform (w/ data augmentations)

    Args:
        args: experiment parameters
        split: dataset split, either 'train' or 'val'

    Return:
        Runtime data transformation object that is callable
    """

    mean, std = get_imagenet_mean_std()
    if split == "train":
        transform_list = [
            ResizeShort(args.short_size),
            RandScale([args.scale_min, args.scale_max]),
            RandRotate(
                [args.rotate_min, args.rotate_max],
                padding=mean,
                ignore_label=args.ignore_label,
            ),
            RandomGaussianBlur(),
            RandomHorizontalFlip(),
            Crop(
                [args.train_h, args.train_w],
                crop_type="rand",
                padding=mean,
                ignore_label=args.ignore_label,
            ),
            ToTensor(),
            Normalize(mean=mean, std=std),
        ]
    elif split == "val":
        transform_list = [
            Crop(
                [args.train_h, args.train_w],
                crop_type="center",
                padding=mean,
                ignore_label=args.ignore_label,
            ),
            ToTensor(),
            Normalize(mean=mean, std=std),
        ]
    else:
        raise RuntimeError("Unknown split. Quitting ...")

    transform_list += [
        ToUniversalLabel(
            args.dataset_name, use_naive_taxonomy=args.use_naive_taxonomy
        )
    ]

    return Compose(transform_list)


def make_dataset(
        split: str = 'train',
        data_root: str = None,
        data_list=None
) -> List[Tuple[str, str]]:
    """
        Args:
        -   split: string representing split of data set to use, must be either 'train','val','test'
        -   data_root: path to where data lives, and where relative image paths are relative to
        -   data_list: path to .txt file with relative image paths

        Returns:
        -   image_label_list: list of 2-tuples, each 2-tuple is comprised of a relative image path
                and a relative label path
    """
    assert split in ['train', 'val', 'test']
    if not os.path.isfile(data_list):
        raise (RuntimeError("Image list file do not exist: " + data_list + "\n"))
    image_label_list = []
    list_read = open(data_list).readlines()
    print("Totally {} samples in {} set.".format(len(list_read), split))

    for line in list_read:
        line = line.strip()
        line_split = line.split(' ')
        if split == 'test':
            if len(line_split) != 1:
                raise (RuntimeError("Image list file read line error : " + line + "\n"))
            image_name = os.path.join(data_root, line_split[0])
            label_name = image_name  # just set place holder for label_name, not for use
        else:
            if len(line_split) != 2:
                raise (RuntimeError("Image list file read line error : " + line + "\n"))
            image_name = os.path.join(data_root, line_split[0])
            label_name = os.path.join(data_root, line_split[1])
        '''
        following check costs some time
        if is_image_file(image_name) and is_image_file(label_name) and os.path.isfile(image_name) and os.path.isfile(label_name):
            item = (image_name, label_name)
            image_label_list.append(item)
        else:
            raise (RuntimeError("Image list file line error : " + line + "\n"))
        '''
        item = (image_name, label_name)
        image_label_list.append(item)
    print("Checking image&label pair {} list done!".format(split))

    return image_label_list


MSEG_DATASETS = [
    'ade20k-150-relabeled',
    'bdd-relabeled',
    'cityscapes-19-relabeled',
    'coco-panoptic-133-relabeled',
    'idd-39-relabeled',
    'mapillary-public65-relabeled',
    'sunrgbd-37-relabeled'
]

MSEG_DATASETS_INFO = dict(
    data_root={dataset: infos[dataset].dataroot for dataset in MSEG_DATASETS},
    train_list={dataset: infos[dataset].trainlist for dataset in MSEG_DATASETS}
)



class transform_args:
    def __init__(self):
        pass

    train_h = 1024
    train_w = 1024
    scale_min = 0.5  # minimum random scale
    scale_max = 2.0  # maximum random scale
    short_size = 1080  # image resolution is 1080p at training
    rotate_min = -10  # minimum random rotate
    rotate_max = 10  # maximum random rotate
    ignore_label = 255
    use_naive_taxonomy = False


@DATASETS.register_module()
class MsegUniversalDataset(Dataset):
    CLASSES = get_universal_class_names()
    PALETTE = None
    def __init__(self, split: str = 'train', dataset_name=None, data_root=None):
        """
            Args:
            -   split: string representing split of data set to use, must be either 'train','val','test'
            -   data_root: path to where data lives, and where relative image paths are relative to
            -   data_list: path to .txt file with relative image paths
            -   transform: Pytorch torchvision transform

            Returns:
            -   None
        """
        self.split = split
        assert dataset_name in MSEG_DATASETS
        transform_args.dataset_name = dataset_name
        self.transform = get_dataset_split_transform(transform_args, split='train')

        data_root = MSEG_DATASETS_INFO['data_root'][dataset_name]
        data_list = MSEG_DATASETS_INFO['train_list'][dataset_name]
        self.data_list = make_dataset(split, data_root, data_list)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        """ """
        image_path, label_path = self.data_list[index]
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # BGR 3 channel ndarray wiht shape H * W * 3
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # convert cv2 read image from BGR order to RGB order
        image = np.float32(image)
        img_metas = dict()
        img_metas['filename'] = os.path.basename(image_path)
        img_metas['ori_filename'] = os.path.basename(image_path)

        img_metas['ori_shape'] = image.shape
        # Set initial values for default meta_keys
        img_metas['pad_shape'] = image.shape
        img_metas['scale_factor'] = 1.0
        num_channels = 1 if len(image.shape) < 3 else image.shape[2]
        img_metas['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)

        label = imageio.imread(label_path).squeeze().astype(np.uint8)  # # GRAY 1 channel ndarray with shape H * W

        if image.shape[0] != label.shape[0] or image.shape[1] != label.shape[1]:
            raise (RuntimeError("Image & label shape mismatch: " + image_path + " " + label_path + "\n"))

        if self.transform is not None:
            if self.split != 'test':
                image, label = self.transform(image, label)
            else:
                # use dummy label in transform, since label unknown for test
                image, label = self.transform(image, image[:, :, 0])

        img_metas['img_shape'] = image.shape

        label = label.unsqueeze(1)

        return {'img': image, 'gt_semantic_seg': label, 'img_metas': DC(img_metas, cpu_only=True)}
