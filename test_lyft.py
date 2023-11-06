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
# import logging;
# logging.disable(logging.WARNING)
# import warnings
# warnings.filterwarnings('ignore')

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)
    
def display_args(args):
    eprint("========== ldls info ==========")
    eprint("host: {}".format(os.getenv('HOSTNAME')))
    eprint(OmegaConf.to_yaml(args))
    eprint("=======================================")

def save_masks(args, name, mask_ped, mask_car):
#     if args.resave:
    np.savez(osp.join(args.data_paths.ldls_full_mask_path, f"{name}"), pedestrian=mask_ped, car=mask_car)
#     else:
#         if not osp.exists(osp.join(args.data_paths.ldls_mask_path, f"{name}")):
#             np.savez(osp.join(args.data_paths.ldls_mask_path, f"{name}"), pedestrian=mask_ped, car=mask_car)
            
            
@hydra.main(version_base='1.1', config_path="configs/", config_name="test_lyft.yaml")
def main(args: DictConfig):
    display_args(args)
    visualize = False
    nusc = LyftDataset(data_path='/home/yy785/datasets/lyft_original/v1.01-train/', json_path='/home/yy785/datasets/lyft_original/v1.01-train/v1.01-train/', verbose=False)
    #Pick sensor
#     camera_channel = 'CAM_FRONT'
    camera_channels = ['CAM_BACK_RIGHT','CAM_FRONT_RIGHT','CAM_FRONT_LEFT','CAM_BACK_LEFT','CAM_FRONT','CAM_BACK']
    #run image detector
    detector = MaskRCNNDetector()
#     sam_path = osp.join(args.data_paths.ldls_sample_path, "all_samples.txt")
    sam_path = args.sample_path
    scenes = nusc.scene

    
    with open(sam_path, "r") as f:
        sample_tokens = f.read().splitlines()
    
    for sample_token in tqdm(sample_tokens):
        
        
        sample = nusc.get('sample', sample_token)
        base_name = sample['token']
        mask_ped = None
        mask_car = None
        
        if osp.exists(osp.join(args.data_paths.ldls_full_mask_path, f"{base_name}.npz")):
            continue
        
        for camera_channel in camera_channels:

            detections = detector.detect_nusc(sample, nusc, camera_channel)
            detection = detections[0]

            lidarseg = LidarSegmentation(data_type='nusc')
            results = lidarseg.run_nusc(detection, sample, nusc, camera_channel, max_iters=100)

            if visualize:
                detection.visualize_nusc(sample, nusc, camera_channel)
                plot_segmentation_result(results, label_type='class', name=f'{base_name}.html')

            class_labels = results.class_labels()

            if type(mask_ped) is np.ndarray:
                mask_ped[results.in_camera_view] = np.logical_or(mask_ped[results.in_camera_view], class_labels == 1)
            else:
                mask_ped = np.zeros(results.in_camera_view.shape[0], dtype=bool)
                mask_ped[results.in_camera_view] = np.logical_or(mask_ped[results.in_camera_view], class_labels == 1)

            if type(mask_car) is np.ndarray:
                mask_car[results.in_camera_view] = np.logical_or(mask_car[results.in_camera_view], class_labels == 3)
            else:
                mask_car = np.zeros(results.in_camera_view.shape[0], dtype=bool)
                mask_car[results.in_camera_view] = np.logical_or(mask_car[results.in_camera_view], class_labels == 3)


        filename = f'{base_name}'
        save_masks(args, filename, mask_ped=mask_ped, mask_car=mask_car)

if __name__=="__main__":
    main()