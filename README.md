# Installation

``git clone -b clean-up git@github.com:janino163/LDLS_anomaly.git``

Installing dependencies using conda is recommended:

``cd LDLS_anomaly``

``conda create -n ldls_anomaly python=3.8``

``conda activate ldls_anomaly``

``conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch``

``pip install -r requirements.txt``

Install Ithaca365-devkit

``git clone git@github.com:cxy1997/ithaca365-devkit.git``

``cd ithaca365-devkit``

``python setup.py develop``

# Generating 3D box labels

## set configs

configs to update: configs/test_ithaca365.yaml

update the path string: **sample_path**

this is a csv file where each row timestamped matched data sample tokens: [point_sensor_token, cam0_sensor_token, cam1_sensor_token]

you should have a row for each frame you want labeled.

configs to update: configs/data_paths/default_ithaca365.yaml

update

configs to update: configs/test_ithaca365.yaml

update the path string: **ithaca365_boxes**

this is where the 3d boxes are saved

update the path string: **ithaca365_masks**

this is where intermediate data products are stored to make reruns faster

## run script

The entry script can be run with the following command.

``python corrected_box_gen_ithaca365.py``