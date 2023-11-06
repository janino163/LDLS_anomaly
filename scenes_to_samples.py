import numpy as np
from pathlib import Path
import skimage
import sys
import os
import os.path as osp
from tqdm import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf
from ithaca365.ithaca365 import Ithaca365
import csv
import json
def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)
    
def display_args(args):
    eprint("========== ldls info ==========")
    eprint("host: {}".format(os.getenv('HOSTNAME')))
    eprint(OmegaConf.to_yaml(args))
    eprint("=======================================")


@hydra.main(version_base='1.1', config_path="configs/", config_name="test_ithaca365_scene.yaml")
def main(args: DictConfig):
    display_args(args)
    nusc = Ithaca365(version='v1.1', dataroot='/share/campbell/Skynet/nuScene_format/v1.1', verbose=True)
    scenes = nusc.scene
    sample_list = []
            
#     test_scenes = ['6e5e4e1dc38cc7ddfd70c46815821dcc']
    matched_dict = {'train': {}, 'test': {}}
    for test_scene in scenes:
#         print(test_scene)
        matched_dict['train'].update({test_scene['token']: []})
        scene = nusc.get('scene', test_scene['token'])
        first_sample = nusc.get('sample', scene['first_sample_token'])
        
        track_list = []
        cam_data_0 = nusc.get('sample_data', first_sample['data']['cam0'])
        cam_data_2 = nusc.get('sample_data', first_sample['data']['cam2'])
        lidar_data = nusc.get('sample_data', first_sample['data']['LIDAR_TOP'])
        track_name = lidar_data['token']
        matched_dict['train'][test_scene['token']].append(track_name)
        while True:
            track_list.append([lidar_data['token'], cam_data_0['token'], cam_data_2['token']])
            sample_list.append([lidar_data['token'], cam_data_0['token'], cam_data_2['token']])
            
            if lidar_data['next'] == "":
                break
                
            cam_data_0 = nusc.get('sample_data', cam_data_0['next'])
            cam_data_2 = nusc.get('sample_data', cam_data_2['next'])
            lidar_data = nusc.get('sample_data', lidar_data['next'])
            
        track_file_path = osp.join(args.data_paths.ldls_track_path, track_name)
        if not os.path.exists(track_file_path):
            os.makedirs(track_file_path)

        with open(f"{track_file_path}.csv", "w") as f:
            writer = csv.writer(f)
            writer.writerows(track_list)
            
    sample_file_path = osp.join(args.data_paths.ldls_sample_path, "all_samples")
    if not os.path.exists(sample_file_path):
        os.makedirs(sample_file_path)
    print(f"{sample_file_path}.csv")
    with open(f"{sample_file_path}.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerows(sample_list)
                
    file_path = osp.join(args.data_paths.matched_scenes_path, 'matched_scenes_ithaca365.json')
    if not os.path.exists(args.data_paths.matched_scenes_path):
        os.makedirs(args.data_paths.matched_scenes_path)
    json.dump(matched_dict, open(file_path, 'w'))

    
        
if __name__=="__main__":
    main()