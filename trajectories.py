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

class Frame(object):
    def __init__(self, x, y, vx, vy):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        
class Trajectory():
    def __init__(self, id_, sample_token, translation, rotation, size, velocity, name, scene_token):
        self.id = id_
        self.sample_token = sample_token
        self.translation = translation # x, y, z
        self.rotation = rotation
        self.size = size
        self.velocity = velocity # vx, vy
        self.scene_token = scene_token
        self.name = name
    def get_translation(self):
        return np.array(self.translation).reshape((-1,3))
    def get_velocity(self):
        return np.array(self.velocity).reshape((-1,2))
    def get_id(self):
        return self.id
    def append(self, translation, rotation, size, velocity):
        self.translation.extend(translation)
        self.rotation.extend(rotation)
        self.size.extend(size)
        self.velocity.extend(velocity)
        

class TrajectoryManager():
    def __init__(self):
        self.trajectories = {}
        
    def add(self, id_, sample_token, translation, rotation, size, velocity, name, scene_token):
        self.trajectories.update({str(id_): Trajectory(id_, sample_token, translation, rotation, size, velocity, name, scene_token)})
        
    def get_ids(self):
        return list(self.trajectories.keys())
    
    def update(self, id_, translation, rotation, size, velocity):
        self.trajectories[str(id_)].append(translation, rotation, size, velocity)
        
    def get_trajectories(self):
        return self.trajectories
    
    
def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)
    
def display_args(args):
    eprint("========== ldls info ==========")
    eprint("host: {}".format(os.getenv('HOSTNAME')))
    eprint(OmegaConf.to_yaml(args))
    eprint("=======================================")
    
@hydra.main(version_base='1.1', config_path="configs/", config_name="test_lyft.yaml")
def main(args: DictConfig):
    file_path = osp.join(args.data_paths.matched_scenes_path, 'matched_scenes.json')
    matches = json.load(open(file_path))
    keys = list(matches.keys())
    results_base = '/home/jan268/repo/LDLS_anomaly/outputs/2022-07-25/02-33-12'
    results_base = '/home/jan268/repo/LDLS_anomaly/outputs/2022-09-03/00-47-12'
    
    
    for key in keys:
        for match in matches[key]:
            mt = TrajectoryManager()
            results_path = osp.join(results_base, key) # read in tracking for scene
            results = json.load(open(results_path))
            results = results['results']
            samples = list(results.keys()) # get samples in scene
            for sample in samples:
                tracks = results[sample]
                for track in tracks:
                    id_ = track['tracking_id']
                    sample_token = track['sample_token']
                    rotation = track['rotation']
                    translation = track['translation']
                    assert len(translation) == 3
                    size = track['size']
                    velocity = track['velocity']
                    name = track['tracking_name']
                    if id_ not in mt.get_ids():
                        mt.add(id_, sample_token, translation, rotation, size, velocity, name, match)
                    else:
                        mt.update(id_, translation, rotation, size, velocity)
                    
                
            break
        break
    x_store = []
    y_store = []
#     vx = []
#     vy = []
#     strore_frames = {}
    frames = []
    
    for key, value in mt.trajectories.items():
        trans = value.get_translation()
        vel = value.get_velocity()
        x_store = []
        y_store = []
        for t, v in zip(trans, vel):
            x = np.array([t[0]])
            y = np.array([t[1]])
            vx = np.array([v[0]])
            vy = np.array([v[1]])
            frames.append(Frame(x, y, vx, vy))
            x_store.extend(x)
            y_store.extend(y)
            
        plt.plot(x_store, y_store)
    np.save("frames_one_scene.npy", frames)
    
#         plt.plot(trans[:,0], trans[:,1])
#     plt.quiver(x,y,vx,vy)   
    plt.savefig('test.png')
    
    
if __name__=="__main__":
    main()