import numpy as np
from pathlib import Path
import skimage

from lidar_segmentation.segmentation import LidarSegmentation
from lidar_segmentation.utils import load_image
from mask_rcnn.mask_rcnn import MaskRCNNDetector
from lidar_segmentation.plotting import plot_segmentation_result
from nuscenes.nuscenes import NuScenes
import matplotlib.pyplot as plt

nusc = NuScenes(version='v1.0-mini', dataroot='/home/jan268/datasets/nuscenes', verbose=False)
my_scene = nusc.scene[1]
first_sample_token = my_scene['first_sample_token']
sample = nusc.get('sample', first_sample_token)

#Pick sensor
camera_channel = 'CAM_FRONT'

#run image detector
detector = MaskRCNNDetector()
detections = detector.detect_nusc(sample, nusc, camera_channel)
x = []
y = []
step = 0
while sample['next'] != '':

    for detection in detections:
    #     detection.visualize_nusc(sample, nusc, camera_channel)

        lidarseg = LidarSegmentation(data_type='nusc')
        results = lidarseg.run_nusc(detection, sample, nusc, camera_channel, max_iters=20)
        
        plot_segmentation_result(results, label_type='class', name=f'step_{step}.html')
        
        global_lidar = results.get_global_points(nusc)
        class_labels = results.class_labels()

        mask = np.ones(global_lidar.shape[0], dtype=bool)
        mask = np.logical_and(mask, class_labels == 1)

        pedestrians = global_lidar[mask, :]
        x.extend(pedestrians[:, 0])
        y.extend(pedestrians[:, 1])
        
    sample = nusc.get('sample', sample['next'])
    step = step + 1
plt.hist2d(x, y, bins=200)
plt.savefig('global_pedestrians.png')
    # Show points colored by class label
