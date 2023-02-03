import os
import os.path as osp
from functools import reduce

import mmcv
import numpy as np
from mmcv.utils import print_log
from terminaltables import AsciiTable
from torch.utils.data import Dataset

from mmseg.core import eval_metrics
from mmseg.utils import get_root_logger
from .builder import DATASETS
from .pipelines import Compose
import pandas as pd
from PIL import Image

@DATASETS.register_module()
class RVC_SEG_Dataset(Dataset):
    """Custom dataset for semantic segmentation. An example of file structure
    is as followed.

    .. code-block:: none

        ├── data
        │   ├── my_dataset
        │   │   ├── img_dir
        │   │   │   ├── train
        │   │   │   │   ├── xxx{img_suffix}
        │   │   │   │   ├── yyy{img_suffix}
        │   │   │   │   ├── zzz{img_suffix}
        │   │   │   ├── val
        │   │   ├── ann_dir
        │   │   │   ├── train
        │   │   │   │   ├── xxx{seg_map_suffix}
        │   │   │   │   ├── yyy{seg_map_suffix}
        │   │   │   │   ├── zzz{seg_map_suffix}
        │   │   │   ├── val

    The img/gt_semantic_seg pair of CustomDataset should be of the same
    except suffix. A valid img/gt_semantic_seg filename pair should be like
    ``xxx{img_suffix}`` and ``xxx{seg_map_suffix}`` (extension is also included
    in the suffix). If split is given, then ``xxx`` is specified in txt file.
    Otherwise, all files in ``img_dir/``and ``ann_dir`` will be loaded.
    Please refer to ``docs/tutorials/new_dataset.md`` for more details.


    Args:
        pipeline (list[dict]): Processing pipeline
        img_dir (str): Path to image directory
        img_suffix (str): Suffix of images. Default: '.jpg'
        ann_dir (str, optional): Path to annotation directory. Default: None
        seg_map_suffix (str): Suffix of segmentation maps. Default: '.png'
        split (str, optional): Split txt file. If split is specified, only
            file with suffix in the splits will be loaded. Otherwise, all
            images in img_dir/ann_dir will be loaded. Default: None
        data_root (str, optional): Data root for img_dir/ann_dir. Default:
            None.
        test_mode (bool): If test_mode=True, gt wouldn't be loaded.
        ignore_index (int): The label index to be ignored. Default: 255
        reduce_zero_label (bool): Whether to mark label zero as ignored.
            Default: False
        classes (str | Sequence[str], optional): Specify classes to load.
            If is None, ``cls.CLASSES`` will be used. Default: None.
        palette (Sequence[Sequence[int]]] | np.ndarray | None):
            The palette of segmentation map. If None is given, and
            self.PALETTE is None, random palette will be generated.
            Default: None
    """

    CLASSES = ('airplane',
               'animal',
               'apple',
               'arcade_machine',
               'armchair',
               'awning',
               'backpack',
               'ball',
               'banana',
               'banner',
               'barrel',
               'barrier',
               'baseball_bat',
               'baseball_glove',
               'basket',
               'bathtub',
               'bear',
               'bed',
               'bench',
               'bicycle',
               'bicyclist',
               'bike_lane',
               'bike_rack',
               'billboard',
               'billiard_table',
               'bird',
               'blanket',
               'boat',
               'book',
               'bookcase',
               'booth',
               'bottle',
               'bowl',
               'box',
               'brickwall',
               'bridge',
               'broccoli',
               'building',
               'bulletin_board',
               'bus',
               'cabinetry',
               'cake',
               'canopy',
               'car',
               'car_mount',
               'caravan',
               'cardboard',
               'carrot',
               'cat',
               'catch_basin',
               'ceiling',
               'chair',
               'chandelier',
               'chest_of_drawers',
               'clock',
               'clothing',
               'coffee_table',
               'computer',
               'computer_keyboard',
               'computer_monitor',
               'conveyer_belt',
               'couch',
               'counter',
               'counter_table',
               'countertop',
               'cow',
               'cup',
               'curb',
               'curb_cut',
               'curtain',
               'cushion',
               'desk',
               'dirt',
               'dishwasher',
               'dog',
               'donut',
               'door',
               'double_door',
               'drinking_glass',
               'dynamic',
               'ego',
               'electric_fan',
               'elephant',
               'escalator',
               'exhaust_hood',
               'fence',
               'fire_hydrant',
               'fireplace',
               'flag',
               'flat_vegetation_soil',
               'floor',
               'flower',
               'flowerpot',
               'food',
               'fork',
               'fountain',
               'frisbee',
               'fruit',
               'furniture',
               'giraffe',
               'grandstand',
               'gravel',
               'ground',
               'guard_rail',
               'hair_dryer',
               'handbag',
               'handcart',
               'handrail',
               'hill',
               'horse',
               'hot_dog',
               'house',
               'hut',
               'infant_bed',
               'junction_box',
               'kitchen_island',
               'kite',
               'knife',
               'lamp',
               'land_vehicle',
               'laptop',
               'light_bulb',
               'light_source',
               'luggage_and_bags',
               'mailbox',
               'manhole',
               'microwave',
               'minibike',
               'mirror',
               'mobile_phone',
               'motorcycle',
               'motorcyclist',
               'mountain',
               'mouse',
               'net',
               'orange',
               'ottoman',
               'oven',
               'palm_tree',
               'paper',
               'parking_meter',
               'parking_space',
               'pathway',
               'pedestal',
               'pedestrian_area',
               'person',
               'phone_booth',
               'pickup_truck',
               'picture_frame',
               'pier',
               'pillar',
               'pillow',
               'pizza',
               'plain_crosswalk',
               'plant',
               'plate',
               'platform',
               'playing_field',
               'pole',
               'poster',
               'pothole',
               'potted_plant',
               'projection_screen',
               'radiator',
               'rail_track',
               'refrigerator',
               'remote',
               'rider',
               'river',
               'road',
               'road_shoulder',
               'road_surface_marking',
               'rock',
               'roof',
               'rug',
               'runway',
               'sand',
               'sandwich',
               'scissors',
               'screen_door',
               'sculpture',
               'sea',
               'seating',
               'serving_tray',
               'sheep',
               'shelf',
               'ship',
               'shower_curtain',
               'showerhead',
               'sidewalk',
               'signboard',
               'sink',
               'skateboard',
               'ski',
               'sky',
               'skyscraper',
               'snow',
               'snowboard',
               'soil',
               'spoon',
               'stairs',
               'static',
               'stone_wall',
               'stool',
               'stop_sign',
               'stove',
               'streetlight',
               'striped_crosswalk',
               'suitcase',
               'surfboard',
               'surveillance_camera',
               'swimming_pool',
               'swivel_chair',
               'table',
               'tank',
               'teddy_bear',
               'television_set',
               'tennis_racket',
               'tent',
               'tie',
               'tiles',
               'toaster',
               'toilet',
               'toothbrush',
               'towel',
               'tower',
               'toy',
               'traffic_light',
               'traffic_sign_backside',
               'traffic_sign_frame',
               'traffic_sign_front',
               'trailer',
               'train',
               'trash',
               'tree',
               'truck',
               'tunnel',
               'umbrella',
               'van',
               'vase',
               'vegetation',
               'video_display',
               'vitrine',
               'wall',
               'wall_bracket',
               'wardrobe',
               'washing_machine',
               'waste_container',
               'water',
               'waterfall',
               'window',
               'window_blind',
               'wooden_floor',
               'wooden_wall',
               'zebra',
               'unlabeled')

    PALETTE = None

    def __init__(self,
                 pipeline,
                 img_dir,
                 img_suffix='.jpg',
                 ann_dir=None,
                 seg_map_suffix='.png',
                 split=None,
                 data_root=None,
                 test_mode=False,
                 ignore_index=255,
                 remap_rvc_dataset=None,
                 rvc_seg_mapping_tsv=None,
                 reduce_zero_label=False,
                 classes=None,
                 palette=None):
        self.pipeline = Compose(pipeline)
        self.img_dir = img_dir
        self.img_suffix = img_suffix
        self.ann_dir = ann_dir
        self.seg_map_suffix = seg_map_suffix
        self.split = split
        self.data_root = data_root
        self.test_mode = test_mode
        self.ignore_index = ignore_index

        self.reduce_zero_label = reduce_zero_label
        self.label_map = None
        self.CLASSES, self.PALETTE = self.get_classes_and_palette(
            classes, palette)

        self.CITYSCAPES_CLASSES = ('road', 'sidewalk', 'building', 'wall', 'fence', '{pole, polegroup}',
               'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky',
               'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
               'bicycle')

        if remap_rvc_dataset is not None:
            assert rvc_seg_mapping_tsv is not None
            mapping_all = pd.read_csv(rvc_seg_mapping_tsv)
            if remap_rvc_dataset.startswith('cityscapes'):
                ignore_indexes = np.where(~np.isin(np.array(mapping_all['cityscapes-34']), self.CITYSCAPES_CLASSES))[0]
            else:
                mapping = mapping_all[remap_rvc_dataset]
                label_map = {}
                ignore_indexes = np.where(np.isnan(np.array(mapping)))[0]
                # ignore_indexes = np.where(np.array(mapping == 'unlabeled'))[0]
            self.excluded_indexes = ignore_indexes
            valid_classes_num = 256 - len(self.excluded_indexes)
            print(f"{valid_classes_num} classes are evaluated.")
            # for i in range(256):
            #     if i in ignore_indexes:
            #         label_map[i] = 255
            #     else:
            #         label_map[i] = i
            # self.label_map = label_map

        # join paths if data_root is specified
        if self.data_root is not None:
            if not osp.isabs(self.img_dir):
                self.img_dir = osp.join(self.data_root, self.img_dir)
            if not (self.ann_dir is None or osp.isabs(self.ann_dir)):
                self.ann_dir = osp.join(self.data_root, self.ann_dir)
            if not (self.split is None or osp.isabs(self.split)):
                self.split = osp.join(self.data_root, self.split)

        # load annotations
        self.img_infos = self.load_annotations(self.img_dir, self.img_suffix,
                                               self.ann_dir,
                                               self.seg_map_suffix, self.split)

    def __len__(self):
        """Total number of samples of data."""
        return len(self.img_infos)

    def load_annotations(self, img_dir, img_suffix, ann_dir, seg_map_suffix,
                         split):
        """Load annotation from directory.

        Args:re
            img_dir (str): Path to image directory
            img_suffix (str): Suffix of images.
            ann_dir (str|None): Path to annotation directory.
            seg_map_suffix (str|None): Suffix of segmentation maps.
            split (str|None): Split txt file. If split is specified, only file
                with suffix in the splits will be loaded. Otherwise, all images
                in img_dir/ann_dir will be loaded. Default: None

        Returns:
            list[dict]: All image info of dataset.
        """

        img_infos = []
        if split is not None:
            with open(split) as f:
                for line in f:
                    img_name = line.strip()
                    img_info = dict(filename=img_name + img_suffix)
                    if ann_dir is not None:
                        seg_map = img_name + seg_map_suffix
                        img_info['ann'] = dict(seg_map=seg_map)
                    img_infos.append(img_info)
        else:

            for img in mmcv.scandir(img_dir, img_suffix, recursive=True):
                img_info = dict(filename=img)
                if ann_dir is not None:
                    seg_map = img.replace(img_suffix, seg_map_suffix)
                    img_info['ann'] = dict(seg_map=seg_map)
                img_infos.append(img_info)

        print_log(f'Loaded {len(img_infos)} images', logger=get_root_logger())
        return img_infos

    def get_ann_info(self, idx):
        """Get annotation by index.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """

        return self.img_infos[idx]['ann']

    def pre_pipeline(self, results):
        """Prepare results dict for pipeline."""
        results['seg_fields'] = []
        results['img_prefix'] = self.img_dir
        results['seg_prefix'] = self.ann_dir
        if self.custom_classes:
            results['label_map'] = self.label_map

    def __getitem__(self, idx):
        """Get training/test data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training/test data (with annotation if `test_mode` is set
                False).
        """

        if self.test_mode:
            return self.prepare_test_img(idx)
        else:
            return self.prepare_train_img(idx)

    def prepare_train_img(self, idx):
        """Get training data and annotations after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training data and annotation after pipeline with new keys
                introduced by pipeline.
        """

        img_info = self.img_infos[idx]
        ann_info = self.get_ann_info(idx)

        results = dict(img_info=img_info, ann_info=ann_info)
        self.pre_pipeline(results)
        return self.pipeline(results)

    def prepare_test_img(self, idx):
        """Get testing data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Testing data after pipeline with new keys intorduced by
                piepline.
        """

        img_info = self.img_infos[idx]
        results = dict(img_info=img_info)
        self.pre_pipeline(results)
        return self.pipeline(results)

    def results2img(self, results, imgfile_prefix, to_label_id, indices=None):
        """Write the segmentation results to images.
        Args:
            results (list[ndarray]): Testing results of the
                dataset.
            imgfile_prefix (str): The filename prefix of the png files.
                If the prefix is "somepath/xxx",
                the png files will be named "somepath/xxx.png".
            to_label_id (bool): whether convert output to label_id for
                submission.
            indices (list[int], optional): Indices of input results, if not
                set, all the indices of the dataset will be used.
                Default: None.
        Returns:
            list[str: str]: result txt files which contains corresponding
            semantic segmentation images.
        """
        if indices is None:
            indices = list(range(len(self)))

        mmcv.mkdir_or_exist(imgfile_prefix)
        result_files = []
        for result, idx in zip(results, indices):

            filename = self.img_infos[idx]['filename']
            basename = osp.splitext(osp.basename(filename))[0]
            if 'scannet' in filename:
                folder_prefix = osp.dirname(filename).split('/')[-2]
                basename = folder_prefix + '_' + basename
            png_filename = osp.join(imgfile_prefix, f'{basename}.png')


            output = Image.fromarray(result.astype(np.uint8))
            output.save(png_filename)
            result_files.append(png_filename)

        return result_files

    def format_results(self,
                       results,
                       imgfile_prefix,
                       to_label_id=True,
                       indices=None):
        """Format the results into dir (standard format for ade20k evaluation).
        Args:
            results (list): Testing results of the dataset.
            imgfile_prefix (str | None): The prefix of images files. It
                includes the file path and the prefix of filename, e.g.,
                "a/b/prefix".
            to_label_id (bool): whether convert output to label_id for
                submission. Default: False
            indices (list[int], optional): Indices of input results, if not
                set, all the indices of the dataset will be used.
                Default: None.
        Returns:
            tuple: (result_files, tmp_dir), result_files is a list containing
               the image paths, tmp_dir is the temporal directory created
                for saving json/png files when img_prefix is not specified.
        """

        if indices is None:
            indices = list(range(len(self)))

        assert isinstance(results, list), 'results must be a list.'
        assert isinstance(indices, list), 'indices must be a list.'

        result_files = self.results2img(results, imgfile_prefix, to_label_id,
                                        indices)
        return

    def get_gt_seg_maps(self, efficient_test=False):
        """Get ground truth segmentation maps for evaluation."""
        gt_seg_maps = []
        for img_info in self.img_infos:
            seg_map = osp.join(self.ann_dir, img_info['ann']['seg_map'])
            if efficient_test:
                gt_seg_map = seg_map
            else:
                gt_seg_map = mmcv.imread(
                    seg_map, flag='unchanged', backend='pillow')
            gt_seg_maps.append(gt_seg_map)
        return gt_seg_maps

    def get_classes_and_palette(self, classes=None, palette=None):
        """Get class names of current dataset.

        Args:
            classes (Sequence[str] | str | None): If classes is None, use
                default CLASSES defined by builtin dataset. If classes is a
                string, take it as a file name. The file contains the name of
                classes where each line contains one class name. If classes is
                a tuple or list, override the CLASSES defined by the dataset.
            palette (Sequence[Sequence[int]]] | np.ndarray | None):
                The palette of segmentation map. If None is given, random
                palette will be generated. Default: None
        """
        if classes is None:
            self.custom_classes = False
            return self.CLASSES, self.PALETTE

        self.custom_classes = True
        if isinstance(classes, str):
            # take it as a file path
            class_names = mmcv.list_from_file(classes)
        elif isinstance(classes, (tuple, list)):
            class_names = classes
        else:
            raise ValueError(f'Unsupported type {type(classes)} of classes.')

        if self.CLASSES:
            if not set(classes).issubset(self.CLASSES):
                raise ValueError('classes is not a subset of CLASSES.')

            # dictionary, its keys are the old label ids and its values
            # are the new label ids.
            # used for changing pixel labels in load_annotations.
            self.label_map = {}
            for i, c in enumerate(self.CLASSES):
                if c not in class_names:
                    self.label_map[i] = -1
                else:
                    self.label_map[i] = classes.index(c)

        palette = self.get_palette_for_custom_classes(class_names, palette)

        return class_names, palette

    def get_palette_for_custom_classes(self, class_names, palette=None):

        if self.label_map is not None:
            # return subset of palette
            palette = []
            for old_id, new_id in sorted(
                    self.label_map.items(), key=lambda x: x[1]):
                if new_id != -1:
                    palette.append(self.PALETTE[old_id])
            palette = type(self.PALETTE)(palette)

        elif palette is None:
            if self.PALETTE is None:
                palette = np.random.randint(0, 255, size=(len(class_names), 3))
            else:
                palette = self.PALETTE

        return palette

    def evaluate(self,
                 results,
                 metric='mIoU',
                 logger=None,
                 efficient_test=False,
                 **kwargs):
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated. 'mIoU' and
                'mDice' are supported.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.

        Returns:
            dict[str, float]: Default metrics.
        """

        if isinstance(metric, str):
            metric = [metric]
        allowed_metrics = ['mIoU', 'mDice']
        if not set(metric).issubset(set(allowed_metrics)):
            raise KeyError('metric {} is not supported'.format(metric))
        eval_results = {}
        gt_seg_maps = self.get_gt_seg_maps(efficient_test)
        if self.CLASSES is None:
            num_classes = len(
                reduce(np.union1d, [np.unique(_) for _ in gt_seg_maps]))
        else:
            num_classes = len(self.CLASSES)
        # print('Total classes:', num_classes)
        # print(self.label_map)
        ret_metrics = eval_metrics(
            results,
            gt_seg_maps,
            num_classes,
            self.ignore_index,
            metric,
            label_map=self.label_map,
            reduce_zero_label=self.reduce_zero_label)
        class_table_data = [['Class'] + [m[1:] for m in metric] + ['Acc']]
        if self.CLASSES is None:
            class_names = tuple(range(num_classes))
        else:
            class_names = self.CLASSES
        ret_metrics_round = [
            np.round(ret_metric * 100, 2) for ret_metric in ret_metrics
        ]
        for i in range(1, len(ret_metrics)):
            ret_metrics[i][self.excluded_indexes] = np.nan
            ret_metrics_round[i][self.excluded_indexes] = np.nan
        for i in range(num_classes):
            class_table_data.append([class_names[i]] +
                                    [m[i] for m in ret_metrics_round[2:]] +
                                    [ret_metrics_round[1][i]])
        summary_table_data = [['Scope'] +
                              ['m' + head
                               for head in class_table_data[0][1:]] + ['aAcc']]

        for ret_metric in ret_metrics:
            print(ret_metric)

        ret_metrics_mean = [
            np.round(np.nanmean(ret_metric) * 100, 2)
            for ret_metric in ret_metrics
        ]
        summary_table_data.append(['global'] + ret_metrics_mean[2:] +
                                  [ret_metrics_mean[1]] +
                                  [ret_metrics_mean[0]])
        print_log('per class results:', logger)
        table = AsciiTable(class_table_data)
        print_log('\n' + table.table, logger=logger)
        print_log('Summary:', logger)
        table = AsciiTable(summary_table_data)
        print_log('\n' + table.table, logger=logger)

        for i in range(1, len(summary_table_data[0])):
            eval_results[summary_table_data[0]
            [i]] = summary_table_data[1][i] / 100.0
        if mmcv.is_list_of(results, str):
            for file_name in results:
                os.remove(file_name)
        return eval_results
