import os
import argparse

from PIL import Image
import numpy as np
import pandas as pd
from math import nan, isnan

parser = argparse.ArgumentParser()

parser.add_argument('--map_csv', default='ss_mapping_uint8_new.csv', type=str)
parser.add_argument('--test_set', default=None, type=str)
parser.add_argument('--data_dir', default='/saccadenet/jfxiao/codes/FAN-OOD/segmentation/work_dirs/rvc_seg_test_ALL-KITTI+BDD+IDD_80k/uint8_results', type=str)
parser.add_argument('--target_dir', default='/saccadenet/jfxiao/codes/FAN-OOD/segmentation/work_dirs/submit_rvc_seg_test_ALL-KITTI+BDD+IDD_80k', type=str)
args = parser.parse_args()
test_set_map = {
    'ade20k-151':1,
    'cityscapes-34':3,
    'mapillary-public66':5,
    'scannet-41':6,
    'viper-rvc-32':7,
    'wilddash2-rvc-39':8
}
mapping_all = pd.read_csv(args.map_csv)
all_classes = list(mapping_all.iloc[:, 0])

def create_directory(path):
    if not os.path.isdir(path):
        os.makedirs(path)
    return path

def map_test(test_set):
    print(f'remap {test_set} start')
    data_dir = create_directory(f'{args.data_dir}/{test_set}/')
    target_dir = create_directory(f'{args.target_dir}/{test_set}/')
    
    test_set_target = test_set_map[test_set]
    target_classes = list(mapping_all.iloc[:, test_set_target])
    #unlabeled = 255 if test_set=='cityscapes-34' else 0
    unlabeled = 0
    target_classes = [unlabeled if isnan(i) else i for i in target_classes]
    
    img_list = os.listdir(data_dir)

    for img_name in img_list:
        
        img_PIL = Image.open(data_dir + img_name)
        img_PIL = np.array(img_PIL)
        h,w = img_PIL.shape
        img_remapped = np.zeros((h,w), dtype=np.uint8)

        for i in range(len(target_classes)):
            mask = img_PIL == i
            img_remapped[mask] = np.array(target_classes[i], dtype=np.uint8)
            
        img_remapped = Image.fromarray(img_remapped)
        img_remapped.save(target_dir + img_name)

    print(f'remap {test_set} end')


if __name__ == '__main__':
    test_sets = os.listdir(args.data_dir)
    
    if args.test_set==None:
        for test_set in test_sets:
            map_test(test_set)
    else:
        map_test(args.test_set)

