from ithaca365.ithaca365 import Ithaca365
from ithaca365.utils.data_io import load_velo_scan
import numpy as np
import os
import os.path as osp
import plotly.graph_objects as go
from ithaca365.utils.data_classes import LidarPointCloud, RadarPointCloud, Box
import random
import hydra
from omegaconf import DictConfig, OmegaConf

ptc_layout_config={
    'title': {
        'text': '',
        'font': {
            'size': 20,
            'color': 'rgb(150,150,150)',
        },
        'xanchor': 'left',
        'yanchor': 'top'},
    'paper_bgcolor': 'rgb(255,255,255)',
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
          'aspectmode': 'manual',
          'aspectratio': {'x': 0.75, 'y': 0.75, 'z': 0.05},
          'camera': {'eye': {'x': 0, 'y': 0, 'z': 0.5}},
          'xaxis': {'color': 'rgb(150,150,150)',
                    'dtick': 10,
                    'gridcolor': 'rgb(100,100,100)',
                    'range': [-150, 150],
                    'showbackground': False,
                    'showgrid': False,
                    'showline': False,
                    'showticklabels': True,
                    'tickmode': 'linear',
                    'tickprefix': 'x:'},
          'yaxis': {'color': 'rgb(150,150,150)',
                    'dtick': 10,
                    'gridcolor': 'rgb(100,100,100)',
                    'range': [-150, 150],
                    'showbackground': False,
                    'showgrid': False,
                    'showline': False,
                    'showticklabels': False,
                    'tickmode': 'linear',
                    'tickprefix': 'y:'},
          'zaxis': {'color': 'rgb(150,150,150)',
                    'dtick': 10,
                    'gridcolor': 'rgb(100,100,100)',
                    'range': [-10, 10],
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

def get_lidar(ptc, name='LiDAR', size=0.8):
    return [go.Scatter3d(
        x=ptc[:,0],
        y=ptc[:,1],
        z=ptc[:,2],
        mode='markers',
        marker_size=size,
        name=name)]

def subsample(points, nbr=200000):
    return points[np.random.permutation(points.shape[0])[:nbr]]

def get_linemarks(obj):
#     _, corners = compute_box_3d(obj, calib.P)
    corners = obj.corners().T
    mid_front = (corners[0] + corners[1]) / 2
    mid_left = (corners[0] + corners[3]) / 2
    mid_right = (corners[1] + corners[2]) / 2
    corners = np.vstack(
        (corners, np.vstack([mid_front, mid_left, mid_right])))
    idx = [0,8,9,10,8,1,2,3,0,4,5,1,5,6,2,6,7,3,7,4]
    return corners[idx, :]

def get_bbox(obj, calib, name='bbox'
             , color='yellow'):
    markers = get_linemarks(obj, calib)
    return go.Scatter3d(
        mode='lines',
        x=markers[:, 0],
        y=markers[:, 1],
        z=markers[:, 2],
        line=dict(color=color, width=3),
        name=name)

def get_lidar(ptc, name='LiDAR', size=0.8):
    return [go.Scatter3d(
        x=ptc[:,0],
        y=ptc[:,1],
        z=ptc[:,2],
        mode='markers',
        marker_size=size,
        name=name)]

def showvelo(lidar, calib, labels=None, predictions=None, classes=['car', 'pedestrian', 'truck', 'van'], size=0.8):
    gt_bboxes = [] if labels is None else [get_bbox(obj, calib, name='gt_bbox', color='lightgreen') for obj in labels if obj.name in classes]
    pred_bboxes = [] if predictions is None else [get_bbox(obj, calib, name='pred_bbox', color='red') for obj in predictions if obj.name in classes]
    fig = go.Figure(data=get_lidar(lidar, size=size) +
                    gt_bboxes + pred_bboxes, layout=ptc_layout_config)
    fig.show()
    return fig

def get_custom_annotations(lidar_data_token: str, results_base: str):
    results_path = osp.join(results_base, lidar_data_token) # read in tracking for scene
    results = json.load(open(results_path))
    results = results['results']

    annotations = []
    for result in results[lidar_data_token]:
        annotation = {}
        annotation.update({'sample_token': result['sample_token']})
        annotation.update({'instance_token': result['tracking_id']})
        annotation.update({'translation': result['translation']})
        annotation.update({'size': result['size']})
        annotation.update({'rotation': result['rotation']})
        if result['tracking_name'] == 'car':
            annotation.update({'category_name': 'vehicle.car'})
        elif result['tracking_name'] == 'pedestrian':
            annotation.update({'category_name': 'human.pedestrian.adult'})
            
        # Dummy values
        annotation.update({'num_lidar_pts': 5})
        annotations.append(annotation)
        
    return annotations

def get_box(record: dict) -> Box:
        """
        Instantiates a Box class from a sample annotation record.
        :param sample_annotation_token: Unique sample_annotation identifier.
        """
        return Box(record['translation'], record['size'], Quaternion(record['rotation']),
                   name=record['category_name'], token=record['token'])
    
    
@hydra.main(version_base='1.1', config_path="configs/", config_name="test_ithaca365.yaml")
def main(args: DictConfig):
    # load dataset parser
    nusc = Ithaca365(version='v1.1', dataroot='/share/campbell/Skynet/nuScene_format/v1.1', verbose=True)
    # load in a random sample data from box dir
    annotation_file = random.choice(os.listdir(args.data_paths.ldls_full_box_path))
    
    # get sample token from selction
    sample_data_token = annotation_file.split('.')[0]
    
    # load in transforms
    sd_record = nusc.get('sample_data', sample_data_token)
    cs_record = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
    sensor_record = nusc.get('sensor', cs_record['sensor_token'])
    pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])
    
    # get annotations in frame
    boxes_path = osp.join(args.data_paths.ldls_full_box_path, f"{sample_data_token}.npz")
    with np.load(boxes_path, allow_pickle=True) as f:
#                 pedestrian_boxes = f['pedestrian']
        boxes = f['car'].tolist()
    print(sample_data_token)
    print(len(boxes))
    for box in boxes:
        #box in global coordinate
        print(dir(box))
        exit()
        # Move box to ego vehicle coord system.
        box.translate(-np.array(pose_record['translation']))
        box.rotate(Quaternion(pose_record['rotation']).inverse)

        #  Move box to sensor coord system.
        box.translate(-np.array(cs_record['translation']))
        box.rotate(Quaternion(cs_record['rotation']).inverse)
    
    
if __name__=="__main__":
    main()