import numpy as np
from skimage.morphology import binary_erosion, binary_dilation
# from mrcnn import visualize
from colorsys import hsv_to_rgb
import torch
import torchvision
from torchvision.transforms import transforms as transforms
from skimage.measure import find_contours
import matplotlib.pyplot as plt
from matplotlib import patches,  lines
from matplotlib.patches import Polygon
from skimage.io import imread
import os.path as osp
# CLASS_NAMES = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
#                'bus', 'train', 'truck', 'boat', 'traffic light',
#                'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
#                'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
#                'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
#                'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
#                'kite', 'baseball bat', 'baseball glove', 'skateboard',
#                'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
#                'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
#                'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
#                'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
#                'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
#                'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
#                'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
#                'teddy bear', 'hair drier', 'toothbrush']
CLASS_NAMES = [
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
class Detections(object):
    """
    Stores object detections for an image.
    """
    def __init__(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def create_label_image(self, mask_shrink=0.5, mask_dilate=1.5):
        """
        Outputs a label image, i.e. a (n_rows by n_cols) matrix whose entries
        are 0 (for background pixels) or integer numbers.
        
        So if label_image[i,j] == 4, then pixel (i,j) is part of instance 4
        
        Returns
        -------
        numpy.ndarray

        """
        raise NotImplementedError


class MaskRCNNDetections(Detections):

    def __init__(self, shape, rois, masks, class_ids, scores):
        self.shape = shape
        self.rois = rois
        
        self.masks = masks
        self.class_ids = class_ids # stored as ints
        self.scores = scores

    def __len__(self):
        return self.masks.shape[2]

    @property
    def class_names(self):
        return [CLASS_NAMES[i] for i in self.class_ids]

    @classmethod
    def load_file(cls, filename):
        """
        Load MaskRCNN detections from a zipped Numpy file
        Parameters
        ----------
        file

        Returns
        -------

        """
        filename = str(filename)
        if not filename.endswith(".npz"):
            filename += ".npz"
        with open(filename, "rb") as loadfile:
            npzfile = np.load(loadfile)
            detections = cls(shape=tuple(npzfile["shape"]), rois=npzfile["rois"],
                             masks=npzfile["masks"],
                             class_ids=npzfile["class_ids"],
                             scores=npzfile["scores"])
        return detections

    def to_file(self, filename):
        """
        Save to a zipped Numpy file
        
        Parameters
        ----------
        filename

        Returns
        -------
        None

        """
        with open(filename, "wb") as savefile:
            np.savez_compressed(savefile, shape=np.array(self.shape), rois=self.rois,
                     masks=self.masks, class_ids=self.class_ids,
                     scores=self.scores)

    def visualize(self, image):
        # Make dictionary of class colors
        hues = np.random.rand(len(CLASS_NAMES))
        s = 0.6
        v = 0.8

        # Separate out hues that actually appear in the image
        classes_in_image = list(set(self.class_ids))
        class_hues = np.linspace(0, 1.0, num=len(classes_in_image) + 1)[:-1]
        # Randomize hues but keep them separated and between 0 and 1.0
        class_hues = np.mod(class_hues + np.random.rand(), 1.0)
        np.random.shuffle(class_hues)
        for i, c in enumerate(classes_in_image):
            hues[c] = class_hues[i]

        class_colors = np.array([hsv_to_rgb(h, s, v) for h in hues]) * 255
        class_colors[0, :] = [175, 175, 175]  # set background color to grey

        # Visualize results

        self.display_instances(image, self.rois, self.masks, self.class_ids,
                                    CLASS_NAMES, self.scores,
                                    colors=[class_colors[i, :] / 255. for i in
                                            self.class_ids])
    def visualize_nusc(self, sample, nusc, sensor):
        cam = nusc.get('sample_data', sample['data'][sensor])
        try:
            root = nusc.dataroot
        except AttributeError:
            root = nusc.data_path
        image = imread(osp.join(root, cam['filename']))
        self.visualize(image)
        
    def visualize_ithaca365(self, sample, nusc):
        cam = nusc.get('sample_data', sample)
        try:
            root = nusc.dataroot
        except AttributeError:
            root = nusc.data_path
        image = imread(osp.join(root, cam['filename']))
        self.visualize(image) 
        
    def apply_mask(self, image, mask, color, alpha=0.5):
        """Apply the given mask to the image.
        """
        for c in range(3):
            image[:, :, c] = np.where(mask == 1,
                                      image[:, :, c] *
                                      (1 - alpha) + alpha * color[c] * 255,
                                      image[:, :, c])
        return image

    def display_instances(self, image, boxes, masks, class_ids, class_names,
                      scores=None, title="",
                      figsize=(16, 16), ax=None,
                      show_mask=True, show_bbox=True,
                      colors=None, captions=None):
        """
        boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
        masks: [height, width, num_instances]
        class_ids: [num_instances]
        class_names: list of class names of the dataset
        scores: (optional) confidence scores for each box
        title: (optional) Figure title
        show_mask, show_bbox: To show masks and bounding boxes or not
        figsize: (optional) the size of the image
        colors: (optional) An array or colors to use with each object
        captions: (optional) A list of strings to use as captions for each object
        """
#         print(f'rois initial shape 2: {self.rois.shape}')
        # Number of instances
        N = boxes.shape[0]
        
        if not N:
            print("\n*** No instances to display *** \n")
        else:
            assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

        # If no axis is passed, create one and automatically call show()
        auto_save = True
        if not ax:
            _, ax = plt.subplots(1, figsize=figsize)
            auto_save = True

        # Generate random colors
        colors = colors

        # Show area outside image boundaries.
        height, width = image.shape[:2]
        ax.set_ylim(height + 10, -10)
        ax.set_xlim(-10, width + 10)
        ax.axis('off')
        ax.set_title(title)

        masked_image = image.astype(np.uint32).copy()
        for i in range(N):
            color = colors[i]

            # Bounding box
            if not np.any(boxes[i]):
                # Skip this instance. Has no bbox. Likely lost in image cropping.
                continue
            y1, x1, y2, x2 = boxes[i]
            if show_bbox:
                p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                                    alpha=0.7, linestyle="dashed",
                                    edgecolor=color, facecolor='none')
                ax.add_patch(p)

            # Label
            if not captions:
                class_id = class_ids[i]
                score = scores[i] if scores is not None else None
                label = class_names[class_id]
                caption = "{} {:.3f}".format(label, score) if score else label
            else:
                caption = captions[i]
            ax.text(x1, y1 + 8, caption,
                    color='w', size=11, backgroundcolor="none")

            # Mask
            mask = masks[:, :, i]
            if show_mask:
                masked_image = self.apply_mask(masked_image, mask, color)

            # Mask Polygon
            # Pad to ensure proper polygons for masks that touch image edges.
            padded_mask = np.zeros(
                (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
            padded_mask[1:-1, 1:-1] = mask
            contours = find_contours(padded_mask, 0.5)
            for verts in contours:
                # Subtract the padding and flip (y, x) to (x, y)
                verts = np.fliplr(verts) - 1
                p = Polygon(verts, facecolor="none", edgecolor=color)
                ax.add_patch(p)
        
        ax.imshow(masked_image.astype(np.uint8))
        if auto_save:
            plt.savefig('/home/jan268/temp/image_detections.png')
        
        
    def get_background(self):
        bg_mask = np.logical_not(np.logical_or.reduce(self.masks, axis=2))
        return bg_mask

