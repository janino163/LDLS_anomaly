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
from os import listdir
from os.path import isfile, join
import pandas as pd
def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)
    
def display_args(args):
    eprint("========== ldls info ==========")
    eprint("host: {}".format(os.getenv('HOSTNAME')))
    eprint(OmegaConf.to_yaml(args))
    eprint("=======================================")
    

@hydra.main(version_base='1.1', config_path="configs/", config_name="test_ithaca365_scene_fixed.yaml")
def main(args: DictConfig):
    display_args(args)
    nusc = Ithaca365(version='v1.1', dataroot='/share/campbell/Skynet/nuScene_format/v1.1', verbose=True)
    scenes = nusc.scene
    sample_list = []
    
    matched_dict = {'train': {}, 'test': {}, 'val': {}}
    
    # read in train, test, val filenames in /share/campbell/ithaca365_token
    track_dir = '/share/campbell/ithaca365_token'
    track_file_paths = [f for f in listdir(track_dir) if isfile(join(track_dir, f))]
    
    train_track_files = [i for i in track_file_paths if "train" in i]
    train_track_names = '\t'.join(train_track_files)
    
    test_track_files = [i for i in track_file_paths if "test" in i]
    test_track_names = '\t'.join(test_track_files)
    
#     train_track_files = [i for i in track_file_paths if "val" in i]
#     val_track_names = '\t'.join(val_file_paths)
    sample_to_track = dict()
    for data_class in ['train', 'test']:
        for test_scene in scenes:
            matched_dict[data_class].update({test_scene['token']: []})
            filename = f'/share/campbell/ithaca365_token/{test_scene["name"]}_{data_class}.csv'
            type_ = data_class
#             if test_scene['name'] in train_track_names:
#                 matched_dict['train'].update({test_scene['token']: []})
#                 filename = f'/share/campbell/ithaca365_token/{test_scene["name"]}_train.csv'
#                 type_ = 'train'
#             elif test_scene['name'] in test_track_names:
#                 matched_dict['test'].update({test_scene['token']: []})
#                 filename = f'/share/campbell/ithaca365_token/{test_scene["name"]}_test.csv'
#                 type_ = 'test'
#             else:
#                 raise NotImplementedError
    #             matched_dict['val'].update({test_scene['token']: []})
    #             filename = f'/share/campbell/ithaca365_token/{test_scene['name']}_test.csv'




            scene = nusc.get('scene', test_scene['token'])
            first_sample = nusc.get('sample', scene['first_sample_token'])

            track_list = []
            cam_data_0 = nusc.get('sample_data', first_sample['data']['cam0'])
            cam_data_2 = nusc.get('sample_data', first_sample['data']['cam2'])
            lidar_data = nusc.get('sample_data', first_sample['data']['LIDAR_TOP'])

            # loop through samples in scene until we reach the first sample data in /share/campbell/ithaca365_token files
            data = pd.read_csv(filename, header=None)
            track_data_samples = data.values.tolist()
            target_lidar_sample = track_data_samples[0][0]
            last_lidar_sample = track_data_samples[-1][0]
            while lidar_data['token'] != target_lidar_sample:
                cam_data_0 = nusc.get('sample_data', cam_data_0['next'])
                cam_data_2 = nusc.get('sample_data', cam_data_2['next'])
                lidar_data = nusc.get('sample_data', lidar_data['next'])

            track_name = lidar_data['token']
            sample_to_track.update({track_name: filename})

            matched_dict[type_][test_scene['token']].append(track_name)

            while True:
                track_list.append([lidar_data['token'], cam_data_0['token'], cam_data_2['token']])
                sample_list.append([lidar_data['token'], cam_data_0['token'], cam_data_2['token']])

                if lidar_data['next'] == "":
                    break
                if lidar_data['token'] == last_lidar_sample:
                    print(lidar_data['token'] == last_lidar_sample)
                    break

                cam_data_0 = nusc.get('sample_data', cam_data_0['next'])
                cam_data_2 = nusc.get('sample_data', cam_data_2['next'])
                lidar_data = nusc.get('sample_data', lidar_data['next'])

            track_file_path = osp.join(args.data_paths.ldls_track_path, track_name)
    #         if not os.path.exists(track_file_path):
    #             os.makedirs(track_file_path)

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
    
    np.save("/home/jan268/youya/date_dictionary.npy", sample_to_track)
    
    
if __name__=="__main__":
    main()