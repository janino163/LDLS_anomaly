from tracking.track import format_sample_result, AB3DMOT
import numpy as np
from pathlib import Path
import sys
import os
import os.path as osp
from tqdm import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf
from ithaca365.ithaca365 import Ithaca365
from pyquaternion import Quaternion
import json
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
import pandas as pd
NUSCENES_TRACKING_NAMES = [
  'car',
  'pedestrian'
]
# NUSCENES_TRACKING_NAMES = [
#   'car'
# ]

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)
    
def display_args(args):
    eprint("========== ldls info ==========")
    eprint("host: {}".format(os.getenv('HOSTNAME')))
    eprint(OmegaConf.to_yaml(args))
    eprint("=======================================")

# def get_pose(sample, nusc):
#     pointsensor_token = sample['data']['LIDAR_TOP']
#     pointsensor = nusc.get('sample_data', pointsensor_token)
#     poserecord = nusc.get('ego_pose', pointsensor['ego_pose_token'])
#     return np.array(poserecord['translation'])
def box_iou_format(boxes):
    """
    Args:
        boxes: list of N box objects
    Returns:
        Numpy array of shape (N, 5) representing N boxes, where each box is represented as
            [x, y, z, width, length] in meters.
    """
    formated_boxes = []
    for box in boxes:
        xyz = box.center
        wlh = box.wlh
        formated_boxes.append([xyz[0], xyz[1], xyz[2], wlh[0], wlh[1]])
    return np.array(formated_boxes)
def nms_bev(ori_boxes, iou_threshold):
    """
    Applies non-maximum suppression (NMS) to a set of NuScenes boxes in birds-eye-view.

    Args:
        boxes: list of N box objects
        scores: Numpy array of shape (N,) representing the confidence score of each box.
        iou_threshold: Float representing the IoU threshold for NMS.

    Returns:
        selected_indices: List of indices representing the selected boxes after NMS.
    """
    boxes = box_iou_format(ori_boxes)
    scores = np.ones(boxes.shape[0])
    # Sort the boxes by their scores in descending order
    sorted_indices = np.argsort(scores)[::-1]
    
    # which boxes to keep
    selected_indices = []
    
        
    while len(sorted_indices) > 0:
        # Select the box with the highest score
        best_index = sorted_indices[0]
        selected_indices.append(best_index)

        # Compute the IoU between the best box and all other boxes
        best_box = boxes[best_index]
        other_boxes = boxes[sorted_indices[1:]]
        iou = calculate_iou_bev(best_box, other_boxes)
        
        # Remove boxes with high IoU
        high_iou_indices = np.where(iou >= iou_threshold)[0]
        sorted_indices = np.delete(sorted_indices, high_iou_indices + 1)
        # remove best box
        sorted_indices = np.delete(sorted_indices, 0)

    return selected_indices

def calculate_iou_bev(box, other_boxes):
    """
    Calculates the IoU between a NuScenes box in birds-eye-view and a set of other boxes.

    Args:
        box: Numpy array of shape (5,) representing the box as [x, y, z, width, length] in meters.
        other_boxes: Numpy array of shape (M, 5) representing M other boxes in the same format.

    Returns:
        iou: Numpy array of shape (M,) representing the IoU between the box and each other box.
    """

    # Convert the box to [x_min, y_min, x_max, y_max] format
    x_min = box[0] - box[3] / 2
    y_min = box[1] - box[4] / 2
    x_max = box[0] + box[3] / 2
    y_max = box[1] + box[4] / 2
    box_coords = np.array([x_min, y_min, x_max, y_max])

    # Convert the other boxes to [x_min, y_min, x_max, y_max] format
    other_x_min = other_boxes[:, 0] - other_boxes[:, 3] / 2
    other_y_min = other_boxes[:, 1] - other_boxes[:, 4] / 2
    other_x_max = other_boxes[:, 0] + other_boxes[:, 3] / 2
    other_y_max = other_boxes[:, 1] + other_boxes[:, 4] / 2
    other_coords = np.stack([other_x_min, other_y_min, other_x_max, other_y_max], axis=1)

    # Compute the intersection and union areas
    intersection = np.maximum(0, np.minimum(box_coords[2:], other_coords[:, 2:]) - np.maximum(box_coords[:2], other_coords[:, :2]))
    intersection_area = intersection[:, 0] * intersection[:, 1]
    box_area = box[3] * box[4]
    other_areas = other_boxes[:, 3] * other_boxes[:, 4]
    union_area = box_area + other_areas - intersection_area

    # Compute the IoU
    iou = intersection_area / union_area

    return iou




@hydra.main(version_base='1.1', config_path="configs/", config_name="test_ithaca365_scene_fixed.yaml")
def main(args: DictConfig):
    display_args(args)
    nusc = Ithaca365(version='v1.1', dataroot='/share/campbell/Skynet/nuScene_format/v1.1', verbose=True)
    track_dir = args.data_paths.ldls_track_path
#     track_dir = '/share/campbell/ithaca365_token'
#     for f in listdir(track_dir):
    
    track_file_paths = [f for f in listdir(track_dir) if isfile(join(track_dir, f))]
#     print(len(track_file_paths))
#     exit()
    for track_file_path in track_file_paths:
#         print(track_file_path)
#         exit()
        results = {}
        
        fake_scene_token = track_file_path.split('.')[0]
        
        track_result_path = f'{fake_scene_token}.csv'
        track_file_path = osp.join(track_dir, track_file_path)

#         if os.path.getsize(track_file_path) == 0:
#             print('continue')
#             continue
        data = pd.read_csv(track_file_path, header=None)
        track_data_samples = data.values.tolist()
        
        # loop through samples
        mot_trackers = {tracking_name: AB3DMOT(args.covariance_id, tracking_name=tracking_name, use_angular_velocity=args.use_angular_velocity, tracking_nuscenes=True) for tracking_name in NUSCENES_TRACKING_NAMES}

        for track_data_sample in track_data_samples:
#             sample = nusc.get('sample', sample_token)
            lidar_data_token = track_data_sample[0]
#             print(lidar_data_token)
#             exit()
#             print(track_data_sample)
#             sample_data_record = nusc.get("sample_data", sample['data']['LIDAR_TOP'])
#             ego_pose = nusc.get("ego_pose", sample_data_record['ego_pose_token'])
#             print(f'ego_pose: {ego_pose}')
            # get boxes generated by ldls for sample
#             boxes_path = osp.join(args.data_paths.ldls_full_box_path, f"{lidar_data_token}.npz")
            sample_data = nusc.get('sample_data', lidar_data_token)
            base_name = sample_data['filename'].split('/')[-1].split('.')[0]
#             print(sample_data)
            timestamp = nusc.get('sample_data', lidar_data_token)['timestamp']
    
            boxes_path = osp.join(args.data_paths.ithaca365_boxes, f"{base_name}.npz") # named by time
#             exit()
#             print(boxes_path)
#             print(os.path.isfile(boxes_path))
#             print(os.path.exists(boxes_path))
#             print(np.load(boxes_path))
            
            if not os.path.isfile(boxes_path):
                print(f'break: {boxes_path}')
                break
                
            with np.load(boxes_path, allow_pickle=True) as f:
#                 print('load boxes')
                pedestrian_boxes = f['pedestrian'].tolist()
                ped_valid = nms_bev(pedestrian_boxes, iou_threshold=0.5)
                
                car_boxes = f['car'].tolist()
                car_valid = nms_bev(car_boxes, iou_threshold=0.5)
                
                boxes = []
                
                ped_boxes = [pedestrian_boxes[i] for i in ped_valid]
                boxes.extend(ped_boxes)
                
                c_boxes = [car_boxes[i] for i in car_valid]
                boxes.extend(c_boxes)
                
            
            results[lidar_data_token] = []
            dets = {tracking_name: [] for tracking_name in NUSCENES_TRACKING_NAMES}
            info = {tracking_name: [] for tracking_name in NUSCENES_TRACKING_NAMES}
#             trans = get_pose(sample, nusc)

            for box in boxes: # boxes are in global coordinate
                q = Quaternion(box.orientation)
                
                angle = q.angle if q.axis[2] > 0 else -q.angle
                #[h, w, l, x, y, z, rot_y]
                detection = np.array([
                  box.wlh[2], box.wlh[0], box.wlh[1], 
                  box.center[0],  box.center[1], box.center[2],
                  angle])

                information = np.array([box.score])
                dets[box.name].append(detection)
                info[box.name].append(information)
            
            dets_all = {tracking_name: {'dets': np.array(dets[tracking_name]), 'info': np.array(info[tracking_name])} for tracking_name in NUSCENES_TRACKING_NAMES}
            for tracking_name in NUSCENES_TRACKING_NAMES:
                if dets_all[tracking_name]['dets'].shape[0] > 0:
                    trackers = mot_trackers[tracking_name].update(dets_all[tracking_name], args.match_distance, args.match_threshold, args.match_algorithm, fake_scene_token)
                    # (N, 9)
                    # (h, w, l, x, y, z, rot_y), tracking_id, tracking_score
                    for i in range(trackers.shape[0]):
                        sample_result = format_sample_result(lidar_data_token, tracking_name, trackers[i])
                        results[lidar_data_token].append(sample_result)
        # finished tracking all scenes, write output data
        output_data = {'meta': [], 'results': results}
        
        with open(os.path.join(args.data_paths.ithaca365_trajectories,fake_scene_token), 'w') as outfile:
            json.dump(output_data, outfile)
        

if __name__=="__main__":
    main()