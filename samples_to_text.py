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

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)
    
def display_args(args):
    eprint("========== ldls info ==========")
    eprint("host: {}".format(os.getenv('HOSTNAME')))
    eprint(OmegaConf.to_yaml(args))
    eprint("=======================================")


@hydra.main(version_base='1.1', config_path="configs/", config_name="test_lyft.yaml")
def main(args: DictConfig):
    display_args(args)
    nusc = LyftDataset(data_path='/home/yy785/datasets/lyft_original/v1.01-train/', json_path='/home/yy785/datasets/lyft_original/v1.01-train/v1.01-train/', verbose=False)
    scenes = nusc.scene
    sample_list = []
    for scene in scenes:
        
        sample_token = scene['first_sample_token']
        first_sample = sample_token
        track_list = []
        time = []
        track_name = scene['token']
        while True:
            sample = nusc.get('sample', sample_token)
            sample_list.append(sample['token'])
            track_list.append(sample['token'])
            time.append(sample['timestamp'])
            
            if sample['next'] == '':
                track_file_path = osp.join(args.data_paths.ldls_track_path, track_name)
                
                with open(f'{track_file_path}.txt', 'w') as f:
                    f.write('\n'.join(track_list))
                break

            sample_token = sample['next']
        
        
        sample_file_path = osp.join(args.data_paths.ldls_sample_path, "all_samples")
        with open(f'{sample_file_path}.txt', 'w') as f:
                    f.write('\n'.join(sample_list))
        
if __name__=="__main__":
    main()