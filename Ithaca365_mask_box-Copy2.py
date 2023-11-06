
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
from ithaca365.utils.data_classes import LidarPointCloud, Box
from ithaca365.utils.geometry_utils import view_points
import matplotlib.pyplot as plt
from tqdm import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf
import pandas as pd
from PIL import Image
from pyquaternion import Quaternion
import open3d as o3d
from sklearn.cluster import DBSCAN, KMeans
import matplotlib.pyplot as plt
from ithaca365.utils.geometry_utils import view_points, box_in_image, BoxVisibility, transform_matrix, transform_points
from pyquaternion import Quaternion
from scipy.spatial import Delaunay, cKDTree
from ithaca365.utils.data_io import load_velo_scan
from scipy.spatial.transform import Rotation as R
import types
from multiprocessing import Pool

def get_point_persistency_score(sample_data_token, nusc, num_histories=5, ranges=(-70, 70), 
                                every_x_meter=5., max_neighbor_dist=0.3, num_workers=-1):
    # load current lidar data
    sample_lidar = load_velo_scan(nusc.get_sample_data_path(sample_data_token))  # lidar frame

    # get past traversals in the same location
    # we need to ensure that it is time valid (from times before current scans)
    # and have accurate history (localization is good)
    past_traversals = get_other_traversals(sample_data_token, nusc, num_history=num_histories,
        ranges=ranges, every_x_meter=every_x_meter, time_valid=False, accurate_history_only=True)

    # get number of history points within a ball-radius of each current point
    # we use a KD Tree to save this count
    trees = {}
    for past_sample_data_token, past_ptc in past_traversals.items():
        trees[past_sample_data_token] = cKDTree(past_ptc[:, :3])
    # counting numbers of neighbors across past traversals
    neighbor_count = {}
    for past_sample_data_token in trees.keys():
        neighbor_count[past_sample_data_token] = trees[past_sample_data_token].query_ball_point(
            sample_lidar[:, :3], r=max_neighbor_dist,
            return_length=True,
            workers=num_workers)
    point_counts = np.stack(list(neighbor_count.values())).T

    # computing entropy, which is the point persistency score
    N = point_counts.shape[1]
    P = point_counts / (np.expand_dims(point_counts.sum(axis=1), -1) + 1e-8)
    H = (-P * np.log(P + 1e-8)).sum(axis=1) / np.log(N)
    return H

def get_other_traversals(sample_data_token, nusc, num_history=5,
        ranges=(-70, 70), every_x_meter=5, time_valid=False, accurate_history_only=True):
    sample_data = nusc.get('sample_data', sample_data_token)
    assert sample_data['channel'] == 'LIDAR_TOP'
#     assert sample_data['location_token'] != ""
    ego_pose = nusc.get('ego_pose', sample_data['ego_pose_token'])
    ego2world = transform_matrix(ego_pose['translation'],
                                 Quaternion(ego_pose['rotation']),)
    world2ego = np.linalg.inv(ego2world)
    cs_record = nusc.get('calibrated_sensor', sample_data['calibrated_sensor_token'])
    lidar2ego = transform_matrix(cs_record['translation'],
                                 Quaternion(cs_record['rotation']),)
    ego2lidar = np.linalg.inv(lidar2ego)
    
    # TODO: CHANGE
    token_list = query_by_lidar(sample_data, nusc)
    # remove self
#     token_list.remove(sample_data['token'])
#     if time_valid:
#         token_list = [tkn for tkn in token_list
#                       if nusc.get("sample_data", tkn)['timestamp'] < sample_data['timestamp']]
#     if accurate_history_only:
#         token_list = [tkn for tkn in token_list
#                       if nusc.get("sample_data", tkn)['bestpos']['field.pos_type.type'] >= 56.]
#     token_list = list(filter(lambda x: not self.is_ignored_history(x), token_list))
    history_scans = {}
    for i in range(min(num_history, len(token_list))):
        sample_data_token = token_list[i]
        _current_lidar = load_velo_scan(nusc.get_sample_data_path(sample_data_token))
        _current_lidar = nusc.remove_ego_points(_current_lidar)
        dense_ptc = nusc.get_lidar_within_range(
            sample_data_token, ranges=ranges, every_x_meter=every_x_meter)[0]
        _current_sd = nusc.get('sample_data', sample_data_token)
        _current_ego_pose = nusc.get('ego_pose', _current_sd['ego_pose_token'])
        _current_ego2world = transform_matrix(
            _current_ego_pose['translation'],
            Quaternion(_current_ego_pose['rotation']),)
        dense_ptc = np.concatenate([_current_lidar] + dense_ptc)
        dense_ptc[:, :3] = transform_points(dense_ptc[:, :3], ego2lidar @
                         world2ego @ _current_ego2world @ lidar2ego)
        history_scans[sample_data_token] = dense_ptc.astype(np.float32)
    return history_scans

def query_by_lidar(sample_data, nusc):
    lidar_token = sample_data['token']
    target_pose = nusc.get('ego_pose', sample_data['ego_pose_token'])
    target_T = np.array(target_pose['translation'])
    target_R = np.array(target_pose['rotation'])
    # get other traversals
    # first reverse through sample data to get nearest location token to target location
    canidate_sd = sample_data.copy()
    prev = True
    while True:
        if canidate_sd['prev'] == "":
            prev = False            
        if canidate_sd['location_token'] != "":
            canidate_location = canidate_sd['location_token']
            break 
        if prev:
            canidate_sd = nusc.get('sample_data', canidate_sd['prev'])
        else:
            canidate_sd = nusc.get('sample_data', canidate_sd['next'])
            
        
    # get sample data token based on canidate location
    token_list = nusc.query_by_location_and_channel(
            canidate_sd['location_token'], 'LIDAR_TOP',
            sorted_by='hgt_stdev', increasing_order=True)

    # remove self and inaccurate tokens
    token_list.remove(canidate_sd['token'])
    token_list = [tkn for tkn in token_list
        if nusc.get("sample_data", tkn)['bestpos']['field.pos_type.type'] >= 56.]
    token_list = list(filter(lambda x: not nusc.is_ignored_history(x), token_list))
    target_token_list = []
    for token in token_list:
        # scan forwards if prev=True through each token to get nearest lidar to target
        # scan backwards if prev=False through each token to get nearest lidar to target
        
        curr_min = np.inf
        test_sd = nusc.get('sample_data', token)
        closet_sd_token = token
        while True:
            test_pose = nusc.get("ego_pose", test_sd["ego_pose_token"])
            test_T = np.array(test_pose['translation'])
            test_R = np.array(test_pose['rotation'])
            distance = np.linalg.norm(test_T - target_T)
            if distance > curr_min:
                target_token_list.append(closet_sd_token)
                break
            curr_min = distance
            if prev:
                test_sd = nusc.get('sample_data', test_sd["next"])
            else:
                test_sd = nusc.get('sample_data', test_sd["prev"])
                
            closet_sd_token = test_sd['token']
    return target_token_list

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)
    
def display_args(args):
    eprint("========== ldls info ==========")
    eprint("host: {}".format(os.getenv('HOSTNAME')))
    eprint(OmegaConf.to_yaml(args))
    eprint("=======================================")

def save_masks(args, name, mask_ped, mask_car, obj_mask, p2):
    np.savez(osp.join(args.data_paths.ithaca365_masks, f"{name}"), pedestrian=mask_ped, car=mask_car, obj=obj_mask, p2=p2)

def project_to_image(lidar_data_token, nusc, camera_data_token):
    cam = nusc.get('sample_data', camera_data_token)
    pointsensor = nusc.get('sample_data', lidar_data_token)
    try:
        root = nusc.dataroot
    except AttributeError:
        root = nusc.data_path
    pcl_path = osp.join(root, pointsensor['filename'])
    pc = LidarPointCloud.from_file(pcl_path)
    lidar = pc.points
    im = Image.open(osp.join(root, cam['filename']))

    # First step: transform the pointcloud to the ego vehicle frame for the timestamp of the sweep.
    cs_record = nusc.get('calibrated_sensor', pointsensor['calibrated_sensor_token'])
    pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix)
    pc.translate(np.array(cs_record['translation']))

    # Second step: transform from ego to the global frame.
    poserecord = nusc.get('ego_pose', pointsensor['ego_pose_token'])
    pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix)
    pc.translate(np.array(poserecord['translation']))

    # Third step: transform from global into the ego vehicle frame for the timestamp of the image.
    poserecord = nusc.get('ego_pose', cam['ego_pose_token'])
    pc.translate(-np.array(poserecord['translation']))
    pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix.T)

    # Fourth step: transform from ego into the camera.
    cs_record = nusc.get('calibrated_sensor', cam['calibrated_sensor_token'])
    pc.translate(-np.array(cs_record['translation']))
    pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix.T)

    # Fifth step: actually take a "picture" of the point cloud.
    # Grab the depths (camera frame z axis points away from the camera).
    depths = pc.points[2, :]
    cam_points = view_points(pc.points[:3, :], np.array(cs_record['camera_intrinsic']), normalize=True)
    return cam_points, lidar, depths, im

def mask_points(detection, lidar_cam, lidar, depths, im, object_mask):
    # return list of masks
    ped_seg_mask = np.zeros((lidar.shape[-1])).astype(bool)
    car_seg_mask = np.zeros((lidar.shape[-1])).astype(bool)
    
    min_dist = 1
    fov_mask = np.ones(depths.shape[0], dtype=bool)
    fov_mask = np.logical_and(fov_mask, depths > min_dist)
    fov_mask = np.logical_and(fov_mask, lidar_cam[0, :] > 1)
    fov_mask = np.logical_and(fov_mask, lidar_cam[0, :] < im.size[0]-1)
    fov_mask = np.logical_and(fov_mask, lidar_cam[1, :] > 1)
    fov_mask = np.logical_and(fov_mask, lidar_cam[1, :] < im.size[1]-1)
    
    fov_mask = np.logical_and(fov_mask, object_mask)
    
    car_clusters = []
    ped_clusters = []
    
    masks = detection.masks
    scores = detection.scores
    class_ids = detection.class_ids
    
    lidar = lidar[:3, fov_mask]
    lidar_cam = lidar_cam[:3, fov_mask]
    for i in range(masks.shape[-1]):
        mask = masks[:,:,i]
        ind = mask[lidar_cam[1, :].astype(np.int32), lidar_cam[0, :].astype(np.int32)]
        if class_ids[i] == 1:
            ped_clusters.append(lidar[:3, ind])
            ped_seg_mask[fov_mask] = np.logical_or(ped_seg_mask[fov_mask], ind)
            
        elif class_ids[i] == 3:
            car_clusters.append(lidar[:3, ind])
            car_seg_mask[fov_mask] = np.logical_or(car_seg_mask[fov_mask], ind)
            

    return car_seg_mask, car_seg_mask, ped_clusters, car_clusters

def get_objs(clusters, category):
    objs = []
    for lidar in clusters:
        if lidar.shape[1] < 10:
            continue
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(lidar[0:3,:].T)
        labels = np.array(pcd.cluster_dbscan(eps=1, min_points=5))
        unique_labels = np.unique(labels)
        # remove outliers from unique labels
        unique_labels = np.delete(unique_labels, np.where(unique_labels==-1))
        cluster_num_points = []
        for label in unique_labels:
            cluster_num_points.append(lidar[0:3, labels == label].shape[1])
        
        try:
            max_value = max(cluster_num_points)
        except ValueError:
            continue
        max_index = cluster_num_points.index(max_value)
        max_lidar = lidar[0:3, labels == unique_labels[max_index]]
        objs.append(get_obj(max_lidar, category))
    return objs
        
def get_obj(cluster, category):
    
    object_extents = {'car':{"width": 1.92,
                            "length": 4.62,
                            "height": 1.69,
                            "max_extent": 5.8,
                            "min_extent": 0.05},
                  'pedestrian':{"width": 0.79,
                            "length": 0.9,
                            "height": 1.88,
                            "max_extent": 1.3,
                            "min_extent": 0.05}}
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cluster.T)
    bbox = o3d.geometry.OrientedBoundingBox()
    bbox = bbox.create_from_points(pcd.points)
    
    bbox_rotation = bbox.R.copy()
    extents = bbox.extent
    l = extents[0]
    w = extents[1]
    h = extents[2]
    r = R.from_matrix(bbox_rotation)
    euler_angles = r.as_euler('zyx', degrees=True)
    euler_angles = [0]
    
    
    obj = types.SimpleNamespace()
    obj.t = np.array(bbox.center)
    
    obj.l = l if l > object_extents[category]["length"] else object_extents[category]["length"]
    obj.w = w if w > object_extents[category]["width"] else object_extents[category]["width"]
    obj.h = h if h > object_extents[category]["height"] else object_extents[category]["height"]
    obj.ry = euler_angles[0]
    obj.volume = obj.l*obj.w*obj.h
    return obj

def box_kiiti_to_ithaca365(obj, pointsensor_token, name, nusc):
    center = (float(obj.t[0]), float(obj.t[1]), float(obj.t[2]))
    wlh = (float(obj.w), float(obj.l), float(obj.h))
    yaw_camera = obj.ry
    quat_box = Quaternion(axis=(0, 0, 1), angle=yaw_camera)
    box = Box([0.0, 0.0, 0.0], wlh, quat_box, name=name)
    box.translate(center)
    box.score = np.nan
    
    # Set dummy velocity.
    box.velocity = np.array((0.0, 0.0, 0.0))
    
    pointsensor = nusc.get('sample_data', pointsensor_token)

    # First step: transform the pointcloud to the ego vehicle frame for the timestamp of the sweep. 
    cs_record = nusc.get('calibrated_sensor', pointsensor['calibrated_sensor_token'])
    box.rotate(Quaternion(cs_record['rotation']))
    box.translate(np.array(cs_record['translation']))

    # Second step: transform from ego to the global frame.
    poserecord = nusc.get('ego_pose', pointsensor['ego_pose_token'])
    box.rotate(Quaternion(poserecord['rotation']))
    box.translate(np.array(poserecord['translation']))
    return box

def run(sample_data_token, nusc, detector):
    lidar_sample_data_token = sample_data_token[0]
    pointsensor = nusc.get('sample_data', lidar_sample_data_token)
    base_name = pointsensor['filename'].split('/')[-1].split('.')[0]
    cam_sample_data_tokens = sample_data_token[1:]
#         base_name = lidar_sample_data_token
    car_seg_mask = None
    ped_seg_mask = None
    car_boxes = []
    ped_boxes = []
#         if osp.exists(osp.join(args.data_paths.ldls_full_mask_path, f"{base_name}.npz")):
#             continue
#         if os.path.exists(osp.join(args.data_paths.ithaca365_masks, f"{base_name}")):
#             continue
    H = get_point_persistency_score(lidar_sample_data_token, nusc, num_histories=5, ranges=(0, 70))
    object_mask = H < 0.5
    for camera_channel in cam_sample_data_tokens:
        detections = detector.detect_ithaca365(camera_channel, nusc)
        detection = detections[0]
        lidar_cam, lidar, depths, im = project_to_image(lidar_sample_data_token, nusc, camera_channel)

        temp_car_seg_mask, temp_ped_seg_mask, ped_cluster, car_cluster = mask_points(detection, lidar_cam, lidar, depths, im, object_mask)
        car_objs = get_objs(car_cluster, 'car')
        for car in car_objs:
            car_boxes.append(box_kiiti_to_ithaca365(car, lidar_sample_data_token, 'car', nusc))

        ped_objs = get_objs(ped_cluster, 'pedestrian')
        for pedestrian in ped_objs:
            ped_boxes.append(box_kiiti_to_ithaca365(pedestrian, lidar_sample_data_token, 'pedestrian', nusc))

        if car_seg_mask is None:
            car_seg_mask = temp_car_seg_mask
        else:
            car_seg_mask = np.logical_or(car_seg_mask, temp_car_seg_mask)

        if ped_seg_mask is None:
            ped_seg_mask = temp_ped_seg_mask
        else:
            ped_seg_mask = np.logical_or(ped_seg_mask, temp_ped_seg_mask)


    filename = f'{base_name}'
    # save Mask-rccn + object mask, pure object mask, p2 score
    save_masks(args, filename, mask_ped=ped_seg_mask, mask_car=car_seg_mask, obj_mask=object_mask, p2=H)
    # save box object
    np.savez(osp.join(args.data_paths.ithaca365_boxes, f"{base_name}"), pedestrian=ped_boxes, car=car_boxes)
    
@hydra.main(version_base='1.1', config_path="configs/", config_name="test_ithaca365.yaml")
def main(args: DictConfig):
    display_args(args)
    nusc = Ithaca365(version='v1.2', dataroot='/share/campbell/Skynet/nuScene_format/v1.2', verbose=True)
    detector = MaskRCNNDetector()
    
    detector = MaskRCNNDetector()
    sam_path = args.sample_path
        
    data = pd.read_csv(sam_path, header=None)
    
    sample_data_tokens = data.values.tolist()
#     sample_data_tokens = [['ccedcc590b9548beafba8c19fbe444d4','f75ed782fd441b5acf401a94f49087c5','b4b380d8f61fa3089c24c05424855e9f']]
#     sample_data_tokens = sample_data_tokens[55000:82500]
    sample_data_tokens = sample_data_tokens[109360:114360]

    for sample_data_token in tqdm(sample_data_tokens):
        lidar_sample_data_token = sample_data_token[0]
        pointsensor = nusc.get('sample_data', lidar_sample_data_token)
        base_name = pointsensor['filename'].split('/')[-1].split('.')[0]
        cam_sample_data_tokens = sample_data_token[1:]
    #         base_name = lidar_sample_data_token
        car_seg_mask = None
        ped_seg_mask = None
        car_boxes = []
        ped_boxes = []

        if os.path.exists(osp.join(args.data_paths.ithaca365_masks, f"{base_name}.npz")):
            continue
        H = get_point_persistency_score(lidar_sample_data_token, nusc, num_histories=8, ranges=(0, 70))
        object_mask = H < 0.5
        for camera_channel in cam_sample_data_tokens:
            detections = detector.detect_ithaca365(camera_channel, nusc)
            detection = detections[0]
            lidar_cam, lidar, depths, im = project_to_image(lidar_sample_data_token, nusc, camera_channel)

            temp_car_seg_mask, temp_ped_seg_mask, ped_cluster, car_cluster = mask_points(detection, lidar_cam, lidar, depths, im, object_mask)
            car_objs = get_objs(car_cluster, 'car')
            for car in car_objs:
                car_boxes.append(box_kiiti_to_ithaca365(car, lidar_sample_data_token, 'car', nusc))

            ped_objs = get_objs(ped_cluster, 'pedestrian')
            for pedestrian in ped_objs:
                ped_boxes.append(box_kiiti_to_ithaca365(pedestrian, lidar_sample_data_token, 'pedestrian', nusc))

            if car_seg_mask is None:
                car_seg_mask = temp_car_seg_mask
            else:
                car_seg_mask = np.logical_or(car_seg_mask, temp_car_seg_mask)

            if ped_seg_mask is None:
                ped_seg_mask = temp_ped_seg_mask
            else:
                ped_seg_mask = np.logical_or(ped_seg_mask, temp_ped_seg_mask)


        filename = f'{base_name}'
        # save Mask-rccn + object mask, pure object mask, p2 score
        save_masks(args, filename, mask_ped=ped_seg_mask, mask_car=car_seg_mask, obj_mask=object_mask, p2=H)
        # save box object
        np.savez(osp.join(args.data_paths.ithaca365_boxes, f"{base_name}"), pedestrian=ped_boxes, car=car_boxes)

        
        
if __name__=="__main__":
    main()