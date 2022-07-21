import numpy as np
from pathlib import Path
import skimage

# from lidar_segmentation.detections import MaskRCNNDetections
from lidar_segmentation.segmentation import LidarSegmentation
# from lidar_segmentation.kitti_utils import load_kitti_lidar_data, load_kitti_object_calib
from lidar_segmentation.utils import load_image
from mask_rcnn.mask_rcnn import MaskRCNNDetector
from lidar_segmentation.plotting import plot_segmentation_result
from ithaca365.ithaca365 import Ithaca365

# Define file paths
# calib_path = Path("data/") / "kitti_demo" / "calib" / "000571.txt"
# image_path = Path("data/") / "kitti_demo" / "image_2" / "000571.png"
# lidar_path = Path("data/") / "kitti_demo" / "velodyne" / "000571.bin"

# Load calibration data
# projection = load_kitti_object_calib(calib_path)
dataset = Ithaca365(version="v1.1", dataroot="/share/campbell/Skynet/nuScene_format/data", verbose=True)
sample = None

# Load image
# image = load_image(image_path)

# Load lidar
# lidar = load_kitti_lidar_data(lidar_path, load_reflectance=False)

#run image detector
detector = MaskRCNNDetector()
detections = detector.detect_nusc(sample)
detections.visualize_nusc(sample)



# lidarseg = LidarSegmentation(data_type='ithaca365', projection=dataset)

# Be sure to set save_all=False when running segmentation
# If set to true, returns label diffusion results at each iteration in the results
# This is useful for analysis or visualizing the diffusion, but slow.

# results = lidarseg.run(lidar, detections, max_iters=1, save_all=False)

# Show points colored by class label
# plot_segmentation_result(results, label_type='class')
# plot_segmentation_result(results, label_type='class')