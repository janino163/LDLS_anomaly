"""

Module for performing object segmentation and initial labeling of images.

Reference:
Uses the Mask-RCNN detector from https://github.com/matterport/Mask_RCNN

"""

import os
import os.path as osp
import sys

from lidar_segmentation.detections import MaskRCNNDetections
import torch
import torchvision
from torchvision.transforms import transforms as transforms
import numpy as np
from skimage.io import imread

# Root directory of the project
ROOT_DIR = os.path.abspath(".")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
# from mrcnn import utils
# import mrcnn.model as modellib

# Import COCO config
# sys.path.append(os.path.join(ROOT_DIR, "coco/"))  # To find local version

# from mask_rcnn import coco


COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]


class MaskRCNNDetector(object):
    def __init__(self, threshold=0.9):
        self.threshold = threshold
        # Directory to save logs and trained model
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True, progress=True, 
                                                               num_classes=91)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        # load the modle on to the computation device and set to eval mode
        model.to(self.device).eval()

        # transform to convert the image to tensor
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

        self.model = model

    def detect(self, images, verbose=0):
        """
        Run Mask-RCNN to detect objects.
        Input can be one image, or a list of images
        
        Parameters
        ----------
        images: numpy.ndarray or list
        verbose: int
            0 or 1.

        Returns
        -------
        MaskRCNNDetections, or list of MaskRCNNDetections objects
            

        """
        detect_multiple = type(images) == list
        if not detect_multiple:
            images = [images]
        all_detections = []
        for image in images:
            mask, roi, id_, score = self.get_prediction(image)
            roi = np.array(roi).reshape((-1,4))
            id_ = np.array(id_)
            score = np.array(score)
            
            all_detections.extend([MaskRCNNDetections(shape=image.shape,
                                             rois=roi,
                                             masks=mask,
                                             class_ids=id_,
                                             scores=score)])
        if not detect_multiple:
            return all_detections[0]
        else:
            return all_detections

    def detect_nusc(self, samples, nusc, sensor, verbose=0):
        images = []
        multiple_samples = type(samples) == list
        if not multiple_samples:
            samples = [samples]
        for sample in samples:
            cam = nusc.get('sample_data', sample['data'][sensor])
            
            try:
                root = nusc.dataroot
            except AttributeError:
                root = nusc.data_path
            images.append(imread(osp.join(root, cam['filename'])))
            
        return self.detect(images, verbose)
        
    def get_prediction(self, image, min_score=0.7):
        image = self.transform(image)
        image = torch.unsqueeze(image, dim=0)
        image = image.to(self.device)
        pred = self.model(image)

        pred_score = list(pred[0]['scores'].detach().cpu().numpy())    
        masks = (pred[0]['masks']>0.5).squeeze().detach().cpu().numpy()

        if masks.ndim < 3:
            masks = masks[np.newaxis, :, :]

        masks = np.swapaxes(masks,0,2)
        masks = np.swapaxes(masks,0,1)
        
        
        class_list = list(pred[0]['labels'].detach().cpu().numpy())
        box_list = list(pred[0]['boxes'].detach().cpu().numpy())
        
        pred_class = [i for i in class_list]
        pred_boxes = [[i[1], i[0], i[3], i[2]] for i in box_list]
        
        return masks, pred_boxes, pred_class, pred_score