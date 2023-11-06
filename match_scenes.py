import numpy as np
from pathlib import Path
import skimage
import sys
import os
import os.path as osp
from tqdm import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf
from lyft_dataset_sdk.lyftdataset import LyftDataset
import json
import matplotlib.pyplot as plt
from pyquaternion import Quaternion
from PIL import Image
def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)
    
def display_args(args):
    eprint("========== ldls info ==========")
    eprint("host: {}".format(os.getenv('HOSTNAME')))
    eprint(OmegaConf.to_yaml(args))
    eprint("=======================================")

def crop_image(image: np.array, x_px: int, y_px: int, axes_limit_px: int) -> np.array:
    x_min = int(x_px - axes_limit_px)
    x_max = int(x_px + axes_limit_px)
    y_min = int(y_px - axes_limit_px)
    y_max = int(y_px + axes_limit_px)

    cropped_image = image[y_min:y_max, x_min:x_max]

    return cropped_image
        
def get_pose(sample, nusc):
    pointsensor_token = sample['data']['LIDAR_TOP']
    pointsensor = nusc.get('sample_data', pointsensor_token)
    poserecord = nusc.get('ego_pose', pointsensor['ego_pose_token'])
    trans = poserecord['translation']
    return [trans[0], trans[1]], poserecord

@hydra.main(version_base='1.1', config_path="configs/", config_name="test_lyft.yaml")
def main(args: DictConfig):
    dist_thresn = 5
    display_args(args)
    nusc = LyftDataset(data_path='/home/yy785/datasets/lyft_original/v1.01-train/', json_path='/home/yy785/datasets/lyft_original/v1.01-train/v1.01-train/', verbose=False)
    scenes = nusc.scene
    matched_scenes = []
    matched_dict = {}
    fig, ax = plt.subplots()
    for i, scene in enumerate(tqdm(scenes)):
        
        sample_token = scene['first_sample_token']
        first_sample = sample_token
        scene_token = scene['token']
        if scene_token in matched_scenes:
            continue
            
        
        matched_scenes.append(scene_token)
        poses = []
        matched_dict.update({scene_token: []})
        matched_dict[scene_token].append(scene_token)
        # get pose for target scene
        while True:
            sample = nusc.get('sample', sample_token)
            trans, _ = get_pose(sample, nusc)
            poses.append(trans)
            if sample['next'] == '': 
                break
        
            sample_token = sample['next']
        
        poses = np.array(poses)
        ax.plot(poses[:,0], poses[:,1])
        
        start_ = i + 1
        for curr_scene in scenes[start_:]:
            curr_scene_token = curr_scene['token']
            if curr_scene_token in matched_scenes:
                continue
            curr_sample_token = curr_scene['first_sample_token']
            curr_pose = []
            plot = False
            while True:
                curr_sample = nusc.get('sample', curr_sample_token)
                trans, _ = get_pose(curr_sample, nusc)
                dist = np.linalg.norm(poses - trans, axis=1)
                curr_pose.append(trans)
                if any(dist < 5):
                    if not plot:
                        matched_scenes.append(curr_scene_token)
                        matched_dict[scene_token].append(curr_scene_token)
                        plot = True
                    
                    
                if curr_sample['next'] == '':
                    if plot:
                        curr_pose = np.array(curr_pose)
                        ax.plot(curr_pose[:,0], curr_pose[:,1])
                    break
                curr_sample_token = curr_sample['next']

        if len(matched_dict[scene_token]) < 5:
            del matched_dict[scene_token]
            continue
        
        
#         log = nusc.get("log", scene["log_token"])
#         map_ = nusc.get("map", log["map_token"])
#         map_mask = map_["mask"]
#         axes_limit = 40
#         pixel_coords = map_mask.to_pixel_coords(pose["translation"][0], pose["translation"][1])

#         scaled_limit_px = int(axes_limit * (1.0 / map_mask.resolution))
#         mask_raster = map_mask.mask()

#         cropped = crop_image(mask_raster, pixel_coords[0], pixel_coords[1], int(scaled_limit_px * np.sqrt(2)))

#         ypr_rad = Quaternion(pose["rotation"]).yaw_pitch_roll
#         yaw_deg = -np.rad2deg(ypr_rad[0])

#         rotated_cropped = np.array(Image.fromarray(cropped).rotate(yaw_deg))
#         ego_centric_map = crop_image(
#             rotated_cropped, rotated_cropped.shape[1] / 2, rotated_cropped.shape[0] / 2, scaled_limit_px
#         )
#         ax.imshow(
#             ego_centric_map, extent=[-axes_limit, axes_limit, -axes_limit, axes_limit], cmap="gray", vmin=0, vmax=150
#         )
        plt.savefig(f'{len(matched_dict[scene_token])}_{scene_token}.png')
        plt.cla()
    file_path = osp.join(args.data_paths.matched_scenes_path, 'matched_scenes.json')
    json.dump(matched_dict, open(file_path, 'w'))
        
        
if __name__=="__main__":
    main()