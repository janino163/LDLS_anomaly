
from ithaca365.ithaca365 import Ithaca365

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
from nuscenes.utils.data_classes import LidarPointCloud, Box
import matplotlib.pyplot as plt
from tqdm import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf
import pandas as pd

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)
    
def display_args(args):
    eprint("========== ldls info ==========")
    eprint("host: {}".format(os.getenv('HOSTNAME')))
    eprint(OmegaConf.to_yaml(args))
    eprint("=======================================")

def save_masks(args, name, mask_ped, mask_car):
    np.savez(osp.join(args.data_paths.ldls_full_mask_path, f"{name}"), pedestrian=mask_ped, car=mask_car)

            
            
@hydra.main(version_base='1.1', config_path="configs/", config_name="test_ithaca365.yaml")
def main(args: DictConfig):
    display_args(args)
    visualize = True
    nusc = Ithaca365(version='v1.1', dataroot='/share/campbell/Skynet/nuScene_format/v1.1', verbose=True)
    camera_channels = ['cam0', 'cam2']
    detector = MaskRCNNDetector()
    sam_path = args.sample_path
#     scenes = nusc.scene
    
#     with open(sam_path, "r") as f:
#         sample_tokens = f.read().splitlines()
        
    data = pd.read_csv(sam_path, header=None)
#     data = data.reset_index()
    
    sample_data_tokens = data.values.tolist()
    sample_data_tokens = [['ccedcc590b9548beafba8c19fbe444d4','f75ed782fd441b5acf401a94f49087c5','b4b380d8f61fa3089c24c05424855e9f']]
    for sample_data_token in tqdm(sample_data_tokens):
        
        lidar_sample_data_token = sample_data_token[0]
        cam_sample_data_tokens = sample_data_token[1:]
        base_name = lidar_sample_data_token
        mask_ped = None
        mask_car = None
#         if osp.exists(osp.join(args.data_paths.ldls_full_mask_path, f"{base_name}.npz")):
#             continue
        for camera_channel in cam_sample_data_tokens:
            
            detections = detector.detect_ithaca365(camera_channel, nusc)
            detection = detections[0]
            
            lidarseg = LidarSegmentation(data_type='nusc')
            results = lidarseg.run_ithaca365(detection, lidar_sample_data_token, nusc, camera_channel, max_iters=20, device=0, save_all=False)
            if visualize:
                detection.visualize_ithaca365(camera_channel, nusc)
                plot_segmentation_result(results, label_type='class', name=f'{base_name}.html')
                visualize = False
            class_labels = results.class_labels()
            print(class_labels.shape)
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