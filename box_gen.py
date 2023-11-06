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
# from sklearn.cluster import DBSCAN
from matplotlib.patches import Rectangle
import matplotlib as mpl
# from learning3d.models import PCN
from PCN_PyTorch.models import PCN
from PCN_PyTorch.visualization import plot_pcd_one_view
import plotly.graph_objects as go
import torch
ptc_layout_config={
    'title': {
        'text': 'test vis LiDAR',
        'font': {
            'size': 20,
            'color': 'rgb(150,150,150)',
        },
        'xanchor': 'left',
        'yanchor': 'top'},
    'paper_bgcolor': 'rgb(0,0,0)',
    'width' : 800,
    'height' : 800,
    'margin' : {
        'l': 20,
        'r': 20,
        'b': 20,
        't': 20
    },
    'legend': {
        'font':{
            'size':20,
            'color': 'rgb(150,150,150)',
        },
        'itemsizing': 'constant'
    },
    "hoverlabel": {
        "namelength": -1,
    },
    'showlegend': False,
    'scene': {
#           'aspectmode': 'manual',
#           'aspectratio': {'x': 1, 'y': 1, 'z': 1},
          'camera': {'eye': {'x': 0, 'y': 0, 'z': 0.5}},
          'xaxis': {'color': 'rgb(150,150,150)',
                    'dtick': 10,
                    'gridcolor': 'rgb(100,100,100)',
                    'range': [-40, 40],
                    'showbackground': False,
                    'showgrid': False,
                    'showline': False,
                    'showticklabels': False,
                    'tickmode': 'linear',
                    'tickprefix': 'x:'},
          'yaxis': {'color': 'rgb(150,150,150)',
                    'dtick': 10,
                    'gridcolor': 'rgb(100,100,100)',
                    'range': [-40, 40],
                    'showbackground': False,
                    'showgrid': False,
                    'showline': False,
                    'showticklabels': False,
                    'tickmode': 'linear',
                    'tickprefix': 'y:'},
          'zaxis': {'color': 'rgb(150,150,150)',
                    'dtick': 10,
                    'gridcolor': 'rgb(100,100,100)',
                    'range': [-15, 15],
                    'showbackground': False,
                    'showgrid': False,
                    'showline': False,
                    'showticklabels': False,
                    'tickmode': 'linear',
                    'tickprefix': 'z:'}},
}

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)
    

def display_args(args):
    eprint("========== ldls info ==========")
    eprint("host: {}".format(os.getenv('HOSTNAME')))
    eprint(OmegaConf.to_yaml(args))
    eprint("=======================================")

def box_kiiti_to_lyft(obj, sample, name, nusc):
    center = (float(obj.t[0]), float(obj.t[1]), float(obj.t[2]))
    wlh = (float(obj.w), float(obj.l), float(obj.h))
    yaw_camera = obj.ry
    quat_box = Quaternion(axis=(0, 0, 1), angle=yaw_camera)
    box = Box([0.0, 0.0, 0.0], wlh, quat_box, name=name)
    box.translate(center)
    box.score = np.nan
    
    # Set dummy velocity.
    box.velocity = np.array((0.0, 0.0, 0.0))
    
    pointsensor_token = sample['data']['LIDAR_TOP']
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

def get_lidar(ptc, name='LiDAR', size=0.8):
    return [go.Scatter3d(
        x=ptc[:,0],
        y=ptc[:,1],
        z=ptc[:,2],
        mode='markers',
        marker_size=size,
        name=name)]

def display_plotly(input_pc, output, i):
	input_pc_ = get_lidar(input_pc, name='partial', size=0.8)
	output_ = get_lidar(output, name='complete', size=0.8)
	fig = go.Figure(data= input_pc_ + output_, layout=ptc_layout_config)
# 	fig = go.Figure(data= input_pc_, layout=ptc_layout_config)

	fig.write_html(f"ldls_test_{i}.html")
    
def normalize_pc(points):
    centroid = np.mean(points, axis=0)
    points = points - centroid
    furthest_distance = 4.73
    points = points / furthest_distance
    return points, centroid, furthest_distance

def plot_box_global(box, sample, nusc, ax=None):
    pointsensor_token = sample['data']['LIDAR_TOP']
    pointsensor = nusc.get('sample_data', pointsensor_token)

    # First step: transform the pointcloud to the ego vehicle frame for the timestamp of the sweep. 
    cs_record = nusc.get('calibrated_sensor', pointsensor['calibrated_sensor_token'])
    box.rotate(Quaternion(cs_record['rotation']))
    box.translate(np.array(cs_record['translation']))

    # Second step: transform from ego to the global frame.
    poserecord = nusc.get('ego_pose', pointsensor['ego_pose_token'])
    box.rotate(Quaternion(poserecord['rotation']))
    box.translate(np.array(poserecord['translation']))
    box.render(axis=ax)
    
    return box

def get_lidar_boxes_nusc(pc, mask, name, fit_method='min_zx_area_fit', sample=None, nusc=None, vis=False):
    
    if vis:
        fig, ax = plt.subplots()
    
    boxes = []
    lidar = pc.points.T
    l = lidar[mask,:]
    if l.shape[0] < 5:
        # too few points of objects where found
        return boxes
    
    clustering = DBSCAN(eps=0.3, min_samples=30, n_jobs=4).fit(l)
    labels = clustering.labels_
    unique_labels = list(set(clustering.labels_))

    if vis:
        ax.scatter(l[:,0], l[:, 1], s=0.1)
        ax.axis('equal')
        plt.savefig(f'lidar.png')
        ax.cla()
    for i, label in enumerate(unique_labels):
        if label == -1:
            continue
        ptc = l[labels == label]
#         plt.plot(ptc[:,0], ptc[:,1])
        if ptc.shape[0] < 5:
            continue
            
#             display_plotly(ptc, output, i)
#             plot_pcd_one_view(os.path.join("", '{:03d}.png'.format(i)), [input_ptc[0].detach().cpu().numpy(), fine[0].detach().cpu().numpy()], ['Input', 'Output'], xlim=(-0.35, 0.35), ylim=(-0.35, 0.35), zlim=(-0.35, 0.35))
        obj = get_obj(ptc, lidar, fit_method=fit_method)
        bbox = box_kiiti_to_lyft(obj, sample, name, nusc)
        
        if vis:
            bbox.render(axis=ax) 
        
        boxes.append(bbox)
    if vis:
        ax.scatter(l[:,0], l[:, 1], s=0.1)
        ax.axis('equal')
        plt.savefig('custom_boxes.png')
        
    return boxes


def test_dbscan(pc, mask, name, fit_method='min_zx_area_fit', sample=None, nusc=None, vis=False):
    import itertools
    
    lidar = pc.points.T
    l = lidar[mask,:]
    if l.shape[0] < 5:
        # too few points of objects where found
        return boxes
    eps = [0.6, 0.7, 0.8, 0.9]
    min_samples = [1, 3, 5, 7, 9, 11]
    fig, ax = plt.subplots()
    ax.scatter(l[:,0], l[:, 1], s=0.1)
    ax.axis('equal')
    plt.savefig(f'lidar.png')
    ax.cla()
    for e, sam in itertools.product(eps, min_samples):
        boxes = []
        
        clustering = DBSCAN(eps=e, min_samples=sam).fit(l[:, 0:2])
        labels = clustering.labels_
        unique_labels = list(set(clustering.labels_))


        for i, label in enumerate(unique_labels):
            if label == -1:
                continue
            ptc = l[labels == label]
            if ptc.shape[0] < 5:
                continue

            obj = get_obj(ptc, lidar, fit_method=fit_method)
            bbox = box_kiiti_to_lyft(obj, sample, name, nusc)

            if vis:
                bbox.render(axis=ax)


            boxes.append(bbox)
        if vis:
            ax.scatter(l[:,0], l[:, 1], s=0.1)
            ax.axis('equal')
            plt.savefig(f'custom_boxes_eps_{e}_samples_{sam}.png')
            ax.cla()
        


@hydra.main(version_base='1.1', config_path="configs/", config_name="test_lyft.yaml")
def main(args: DictConfig):
    display_args(args)
    visualize = False
    nusc = LyftDataset(data_path='/home/yy785/datasets/lyft_original/v1.01-train/', json_path='/home/yy785/datasets/lyft_original/v1.01-train/v1.01-train/', verbose=False)
    sam_path = args.sample_path
    
    with open(sam_path, "r") as f:
        sample_tokens = f.read().splitlines()
    
    for sample_token in tqdm(sample_tokens):

        sample = nusc.get("sample", sample_token)
            
        # Transforms
        pointsensor_token = sample["data"]['LIDAR_TOP']

        try:
            root = nusc.dataroot
        except AttributeError:
            root = nusc.data_path
        pointsensor = nusc.get("sample_data", pointsensor_token)
        pcl_path = osp.join(root, pointsensor['filename'])
        
        pc_ori = LidarPointCloud.from_file(pcl_path)

        ldls_mask_path = osp.join(args.data_paths.ldls_full_mask_path, f"{sample['token']}.npz")
        with np.load(ldls_mask_path) as data:
            car_mask = data['car']
            ped_mask = data['pedestrian']
            
        car_boxes = get_lidar_boxes_nusc(pc_ori, car_mask, name='car', fit_method='PCA', sample=sample, nusc=nusc, vis=visualize)
        pedestrian_boxes = get_lidar_boxes_nusc(pc_ori, car_mask, name='pedestrian', fit_method='PCA', sample=sample, nusc=nusc, vis=visualize)
        
        np.savez(osp.join(args.data_paths.ldls_full_box_path, f"{sample_token}"), pedestrian=pedestrian_boxes, car=car_boxes)
        
#         test_dbscan(pc_ori, car_mask, name='car', fit_method='PCA', sample=sample, nusc=nusc, vis=visualize)
        
        if visualize:
            fig, ax = plt.subplots()
            _, boxes, _ = nusc.get_sample_data(sample['data']['LIDAR_TOP'])
            for box in boxes:
                if box.name == "car":
                    box.render(axis=ax)
            plt.savefig('gt_global_boxes.png')
            exit()

        
        
        
        
if __name__=="__main__":
    main()