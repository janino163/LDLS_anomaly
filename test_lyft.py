import numpy as np
from pathlib import Path
import skimage
import sys
import os
import os.path as osp
from lidar_segmentation.segmentation import LidarSegmentation
from lidar_segmentation.utils import load_image
from mask_rcnn.mask_rcnn import MaskRCNNDetector
from lidar_segmentation.plotting import plot_segmentation_result
# from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud, Box
import matplotlib.pyplot as plt
from tqdm import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf
from lyft_dataset_sdk.lyftdataset import LyftDataset
import logging;
logging.disable(logging.WARNING)
import warnings
warnings.filterwarnings('ignore')
def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)
    
def display_args(args):
    eprint("========== ldls info ==========")
    eprint("host: {}".format(os.getenv('HOSTNAME')))
    eprint(OmegaConf.to_yaml(args))
    eprint("=======================================")

def save_masks(args, name, mask_ped, mask_car):
    if args.resave:
        np.savez(osp.join(args.data_paths.ldls_mask_path, f"{name}"), pedestrian=mask_ped, car=mask_car)
    else:
        if not osp.exists(osp.join(args.data_paths.ldls_mask_path, f"{name}")):
            np.savez(osp.join(args.data_paths.ldls_mask_path, f"{name}"), pedestrian=mask_ped, car=mask_car)
            
            
@hydra.main(config_path="configs/", config_name="test_lyft.yaml")
def main(args: DictConfig):
    display_args(args)
    visualize = False
    nusc = LyftDataset(data_path='/home/yy785/datasets/lyft_original/v1.01-train/', json_path='/home/yy785/datasets/lyft_original/v1.01-train/v1.01-train/', verbose=False)
    #Pick sensor
    camera_channel = 'CAM_FRONT'
    #run image detector
    detector = MaskRCNNDetector()
    with open(args.sample_path, "r") as f:
        sample_tokens = f.read().splitlines()
    
    for sample_token in tqdm(sample_tokens[17300:]):
#         print(f'scene number {scene_num} out of {len(nusc.scene)}')
#         first_sample_token = my_scene['first_sample_token']
        sample = nusc.get('sample', sample_token)
#         step = 0
        
#         while sample['next'] != '':
        base_name = sample['token']
        detections = detector.detect_nusc(sample, nusc, camera_channel)
        detection = detections[0]

        lidarseg = LidarSegmentation(data_type='nusc')
        results = lidarseg.run_nusc(detection, sample, nusc, camera_channel, max_iters=20)
#         if step % 10 == 0:
        if visualize:
            detection.visualize_nusc(sample, nusc, camera_channel)
            plot_segmentation_result(results, label_type='class', name=f'{basename}.html')

        global_lidar = results.get_global_points(nusc)
        class_labels = results.class_labels()

        mask_ped = np.ones(global_lidar.shape[0], dtype=bool)
        mask_ped = np.logical_and(mask_ped, class_labels == 1)



        mask_car = np.ones(global_lidar.shape[0], dtype=bool)
        mask_car = np.logical_and(mask_car, class_labels == 3)

        filename = f'{base_name}'
        save_masks(args, filename, mask_ped=mask_ped, mask_car=mask_car)



#         sample = nusc.get('sample', sample['next'])
#         step = step + 1


if __name__=="__main__":
    main()