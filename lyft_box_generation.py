import numpy as np
from pathlib import Path
import skimage
import sys
import os
import os.path as osp
import matplotlib.pyplot as plt
from tqdm import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf
from lyft_dataset_sdk.lyftdataset import LyftDataset
from utils.pointcloud_utils import get_obj
# from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import BoxVisibility, transform_matrix
from nuscenes.utils.data_classes import LidarPointCloud, Box
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation as R
from nuscenes.utils.geometry_utils import view_points
from PIL import Image
from sklearn.cluster import DBSCAN
from matplotlib.patches import Rectangle
import matplotlib as mpl
def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)
    
def display_args(args):
    eprint("========== ldls info ==========")
    eprint("host: {}".format(os.getenv('HOSTNAME')))
    eprint(OmegaConf.to_yaml(args))
    eprint("=======================================")

def box_kiiti_to_lyft(obj, transforms, name):
    kitti_to_nu_lidar_inv = Quaternion(axis=(0, 0, 1), angle=np.pi).inverse
    kitti_to_nu_lidar = Quaternion(axis=(0, 0, 1), angle=np.pi)
#     center = obj.t.reshape(3, 1)
    center = (float(obj.t[0]), float(obj.t[1]), float(obj.t[2]))
    wlh = (float(obj.w), float(obj.l), float(obj.h))
    yaw_camera = obj.ry
    
    
    # Optional: Filter classes.
#     if filter_classes is not None and name not in filter_classes:
#     continue

    # The Box class coord system is oriented the same way as as KITTI LIDAR: x forward, y left, z up.
    # For orientation confer: http://www.cvlibs.net/datasets/kitti/setup.php.

    # 1: Create box in Box coordinate system with center at origin.
    # The second quaternion in yaw_box transforms the coordinate frame from the object frame
    # to KITTI camera frame. The equivalent cannot be naively done afterwards, as it's a rotation
    # around the local object coordinate frame, rather than the camera frame.
    quat_box = Quaternion(axis=(0, 1, 0), angle=yaw_camera) * Quaternion(axis=(1, 0, 0), angle=np.pi/2)
    box = Box([0.0, 0.0, 0.0], wlh, quat_box, name=name)

    # 2: Translate: KITTI defines the box center as the bottom center of the vehicle. We use true center,
    # so we need to add half height in negative y direction, (since y points downwards), to adjust. The
    # center is already given in camera coord system.
    box.translate(center + np.array([0, -wlh[2] / 2, 0]))
    
    # 3: Transform to KITTI LIDAR coord system. First transform from rectified camera to camera, then
    # camera to KITTI lidar.
    box.rotate(Quaternion(matrix=transforms['r0_rect']).inverse)

    box.translate(-transforms['velo_to_cam_T'])
    
    box.rotate(Quaternion(matrix=transforms['velo_to_cam_R']).inverse)
   
    # 4: Transform to nuScenes LIDAR coord system.
    box.rotate(kitti_to_nu_lidar)

    # Set score or NaN.
    box.score = np.nan

    # Set dummy velocity.
    box.velocity = np.array((0.0, 0.0, 0.0))

    # Optional: Filter by max_dist
#     if max_dist is not None:
#     dist = np.sqrt(np.sum(box.center[:2] ** 2))
#     if dist > max_dist:
#         continue

#     boxes.append(box)

    return box

def get_lidar_boxes_nusc(pc, in_view, mask, kitti_transforms, name, fit_method='min_zx_area_fit'):
    boxes = []
    lidar = pc.points.T
    l = lidar[in_view,:][mask,:]
    if l.shape[0] < 5:
        # too few points of objects where found
        return boxes
    
    clustering = DBSCAN(eps=1, min_samples=2).fit(l)
    labels = clustering.labels_
    unique_labels = list(set(clustering.labels_))
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]

    for i, label in enumerate(unique_labels):
        if label == -1:
            continue
        ptc = l[labels == label]

        if len(ptc) < 5:
            continue
        obj = get_obj(ptc, lidar, fit_method=fit_method)
        boxes.append(box_kiiti_to_lyft(obj, kitti_transforms, name))
    return boxes

@hydra.main(version_base='1.1', config_path="configs/", config_name="test_lyft.yaml")
def main(args: DictConfig):
    display_args(args)
    visualize = False
    nusc = LyftDataset(data_path='/home/yy785/datasets/lyft_original/v1.01-train/', json_path='/home/yy785/datasets/lyft_original/v1.01-train/v1.01-train/', verbose=False)
    #Pick sensor
    camera_channel = 'CAM_FRONT'
    
#     kitti_to_nu_lidar = Quaternion(axis=(0, 0, 1), angle=np.pi)
#     kitti_to_nu_lidar_inv = self.kitti_to_nu_lidar.inverse
    with open(args.sample_path, "r") as f:
        sample_tokens = f.read().splitlines()
    
    for sample_token in tqdm(sample_tokens):
        sample = nusc.get("sample", sample_token)
        index_ = 0
#         fig, ax = plt.subplots()
            
        # Transforms
        pointsensor_token = sample["data"]['LIDAR_TOP']
        camera_token = sample["data"][camera_channel]
        cam = nusc.get("sample_data", camera_token)

        try:
            root = nusc.dataroot
        except AttributeError:
            root = nusc.data_path
        im = Image.open(osp.join(root, cam['filename']))

        pointsensor = nusc.get("sample_data", pointsensor_token)
        pcl_path = osp.join(root, pointsensor['filename'])
#         pc_ori = LidarPointCloud.from_file(pcl_path)

        pc = LidarPointCloud.from_file(pcl_path)
        # First step: transform the point-cloud to the ego vehicle frame for the timestamp of the sweep.
        cs_record = nusc.get("calibrated_sensor", pointsensor["calibrated_sensor_token"])
        pc.rotate(Quaternion(cs_record["rotation"]).rotation_matrix)
        pc.translate(np.array(cs_record["translation"]))

        # Second step: transform to the global frame.
        poserecord = nusc.get("ego_pose", pointsensor["ego_pose_token"])
        pc.rotate(Quaternion(poserecord["rotation"]).rotation_matrix)
        pc.translate(np.array(poserecord["translation"]))

        # Third step: transform into the ego vehicle frame for the timestamp of the image.
        poserecord = nusc.get("ego_pose", cam["ego_pose_token"])
        pc.translate(-np.array(poserecord["translation"]))
        pc.rotate(Quaternion(poserecord["rotation"]).rotation_matrix.T)

        # Fourth step: transform into the camera.
        cs_record = nusc.get("calibrated_sensor", cam["calibrated_sensor_token"])
        pc.translate(-np.array(cs_record["translation"]))
        pc.rotate(Quaternion(cs_record["rotation"]).rotation_matrix.T)

        depths = pc.points[2, :]

        points = view_points(pc.points[:3, :], np.array(cs_record['camera_intrinsic']), normalize=True)

        # Remove points that are either outside or behind the camera. Leave a margin of 1 pixel for aesthetic reasons.
        # Also make sure points are at least 1m in front of the camera to avoid seeing the lidar points on the camera
        # casing for non-keyframes which are slightly out of sync.
        min_dist = 1.0
        mask = np.ones(depths.shape[0], dtype=bool)
        mask = np.logical_and(mask, depths > min_dist)
        mask = np.logical_and(mask, points[0, :] > 1)
        mask = np.logical_and(mask, points[0, :] < im.size[0] - 1)
        mask = np.logical_and(mask, points[1, :] > 1)
        mask = np.logical_and(mask, points[1, :] < im.size[1] - 1)
        in_view  = mask

        # Create calibration file.

        # Note: cam uses same conventions in KITTI and nuScenes.
        # Retrieve sensor records.
        sd_record_cam = nusc.get('sample_data', camera_token)
        sd_record_lid = nusc.get('sample_data', pointsensor_token)
        cs_record_cam = nusc.get('calibrated_sensor', sd_record_cam['calibrated_sensor_token'])
        cs_record_lid = nusc.get('calibrated_sensor', sd_record_lid['calibrated_sensor_token'])

        lid_to_ego = transform_matrix(cs_record_lid['translation'], Quaternion(cs_record_lid['rotation']),
                                      inverse=False)
        ego_to_cam = transform_matrix(cs_record_cam['translation'], Quaternion(cs_record_cam['rotation']),
                                      inverse=True)
        velo_to_cam = np.dot(ego_to_cam, lid_to_ego)

        # Convert from KITTI to nuScenes LIDAR coordinates, where we apply velo_to_cam.
        kitti_to_nu_lidar = Quaternion(axis=(0, 0, 1), angle=np.pi)
        velo_to_cam_kitti = np.dot(velo_to_cam, kitti_to_nu_lidar.transformation_matrix)

        # Currently not used.
        imu_to_velo_kitti = np.zeros((3, 4))  # Dummy values.
        r0_rect = Quaternion(axis=[1, 0, 0], angle=0)  # Dummy values.

        # Projection matrix.
        p_left_kitti = np.zeros((3, 4))
        p_left_kitti[:3, :3] = cs_record_cam['camera_intrinsic']  # Cameras are always rectified.

        # Create KITTI style transforms.
        velo_to_cam_rot = velo_to_cam_kitti[:3, :3]
        velo_to_cam_trans = velo_to_cam_kitti[:3, 3]
        assert (velo_to_cam_rot.round(0) == np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]])).all()
        assert (velo_to_cam_trans[1:3] < 0).all()
        kitti_transforms = dict()
#             kitti_transforms['P0'] = np.zeros((3, 4))  # Dummy values.
#             kitti_transforms['P1'] = np.zeros((3, 4))  # Dummy values.
#             kitti_transforms['P2'] = p_left_kitti  # Left camera transform.
#             kitti_transforms['P3'] = np.zeros((3, 4))  # Dummy values.
        kitti_transforms['r0_rect'] = r0_rect.rotation_matrix  # Cameras are already rectified.
#             kitti_transforms['Tr_velo_to_cam'] = np.hstack((velo_to_cam_rot, velo_to_cam_trans.reshape(3, 1)))
        kitti_transforms['velo_to_cam_R'] = velo_to_cam_rot
        kitti_transforms['velo_to_cam_T'] = velo_to_cam_trans
#             kitti_transforms['imu_to_velo'] = imu_to_velo_kitti
        ##############

        ldls_mask_path = osp.join(args.data_paths.ldls_mask_path, f"{sample['token']}.npz")
        with np.load(ldls_mask_path) as data:
            car_mask = data['car']
            ped_mask = data['pedestrian']
            
        car_boxes = get_lidar_boxes_nusc(pc, in_view, car_mask, kitti_transforms, name='car', fit_method='closeness_to_edge')
        pedestrian_boxes = get_lidar_boxes_nusc(pc, in_view, ped_mask, kitti_transforms, name='pedestrian', fit_method='closeness_to_edge')
        np.savez(osp.join(args.data_paths.ldls_box_path, f"{sample_token}"), pedestrian=pedestrian_boxes, car=car_boxes)
        
        
        
        
if __name__=="__main__":
    main()