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
import pandas as pd

def main():
    df = pd.read_csv('./split_loc3.csv')
    


if __name__ == "__main__":
    main()