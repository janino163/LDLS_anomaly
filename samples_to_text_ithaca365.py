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


@hydra.main(version_base='1.1', config_path="configs/", config_name="test_ithaca365.yaml")
def main(args: DictConfig):
    display_args(args)
    nusc = Ithaca365(version='v1.2', dataroot='/share/campbell/Skynet/nuScene_format/v1.2', verbose=True)
    scenes = nusc.scene
    sample_list = []
    
    my_locations = [40, 2345, 2385, 2427, 2478, 2513]
    matched_dict = {'train': {}, 'test': {}}
    for location_index in my_locations:
        my_location = nusc.location[location_index]
        cam_0_data_tokens = nusc.query_by_location_and_channel(my_location['token'], 'cam0')
        cam_2_data_tokens = nusc.query_by_location_and_channel(my_location['token'], 'cam2')
        lidar_data_tokens = nusc.query_by_location_and_channel(my_location['token'], 'LIDAR_TOP')
        matched_dict['train'].update({str(location_index): []})
#         matched_dict[scene_token].append(scene_token)
        
        for traversal_cam_0, traversal_cam_2, traversal_lidar in zip(cam_0_data_tokens, cam_2_data_tokens, lidar_data_tokens):
            track_name = traversal_lidar
            track_list = []
            
            cam_data_0 = nusc.get('sample_data', traversal_cam_0)
            cam_data_2 = nusc.get('sample_data', traversal_cam_2)
            lidar_data = nusc.get('sample_data', traversal_lidar)
            matched_dict['train'][str(location_index)].append(track_name)
            if 'location' in cam_data_0['filename']:
                continue
#                 while True:
#                     track_list.append([lidar_data['token'], cam_data_0['token'], cam_data_2['token']])
#                     sample_list.append([lidar_data['token'], cam_data_0['token'], cam_data_2['token']])
#                     if cam_data_0['next'] == "":
#                         break
#                     cam_data_0 = nusc.get('sample_data', cam_data_0['next'])
#                     cam_data_2 = nusc.get('sample_data', cam_data_2['next'])
#                     lidar_data = nusc.get('sample_data', lidar_data['next'])
            else:
                for i in range(400):
                    track_list.append([lidar_data['token'], cam_data_0['token'], cam_data_2['token']])
                    sample_list.append([lidar_data['token'], cam_data_0['token'], cam_data_2['token']])

                    cam_data_0 = nusc.get('sample_data', cam_data_0['next'])
                    cam_data_2 = nusc.get('sample_data', cam_data_2['next'])
                    lidar_data = nusc.get('sample_data', lidar_data['next'])
                    
            track_file_path = osp.join(args.data_paths.ldls_track_path, track_name)

            with open(f"{track_file_path}.csv", "w") as f:
                writer = csv.writer(f)
                writer.writerows(track_list)

            
    test_scenes = ['1acb9939c6b2111a29b6f74301fa236b', '9ad8d68ea12e5e65f549045c221f2ac1',
                  '65b784284d278fdcc98ccf693a52f8da', '7cbcf5245ef56e61e65da8071b146f8f']
    for test_scene in test_scenes:
        matched_dict['test'].update({test_scene: []})
        
        scene = nusc.get('scene', test_scene)
        first_sample = nusc.get('sample', scene['first_sample_token'])
        
        track_list = []
        cam_data_0 = nusc.get('sample_data', first_sample['data']['cam0'])
        cam_data_2 = nusc.get('sample_data', first_sample['data']['cam2'])
        lidar_data = nusc.get('sample_data', first_sample['data']['LIDAR_TOP'])
        track_name = lidar_data['token']
        matched_dict['test'][test_scene].append(track_name)
        while True:
            track_list.append([lidar_data['token'], cam_data_0['token'], cam_data_2['token']])
            sample_list.append([lidar_data['token'], cam_data_0['token'], cam_data_2['token']])
            
            if lidar_data['next'] == "":
                break
                
            cam_data_0 = nusc.get('sample_data', cam_data_0['next'])
            cam_data_2 = nusc.get('sample_data', cam_data_2['next'])
            lidar_data = nusc.get('sample_data', lidar_data['next'])
            
        track_file_path = osp.join(args.data_paths.ldls_track_path, track_name)
        with open(f"{track_file_path}.csv", "w") as f:
            writer = csv.writer(f)
            writer.writerows(track_list)
            
    sample_file_path = osp.join(args.data_paths.ldls_sample_path, "all_samples")
    
    with open(f"{sample_file_path}.csv", "w") as f:
                writer = csv.writer(f)
                writer.writerows(sample_list)
                
    file_path = osp.join(args.data_paths.matched_scenes_path, 'matched_scenes_ithaca365.json')
    json.dump(matched_dict, open(file_path, 'w'))

    
        
if __name__=="__main__":
    main()