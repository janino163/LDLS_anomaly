# author: Vinit Sarode (vinitsarode5@gmail.com) 03/23/2020

import argparse
import os
import sys
import logging
import numpy
import numpy as np
import torch
import torch.utils.data
import torchvision
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm
import plotly.graph_objects as go
# Only if the files are in example folder.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR[-8:] == 'examples':
	sys.path.append(os.path.join(BASE_DIR, os.pardir))
	os.chdir(os.path.join(BASE_DIR, os.pardir))
	
from learning3d.models import PCN
from learning3d.losses import ChamferDistanceLoss
from learning3d.data_utils import ClassificationData, ModelNet40Data
from learning3d.data_utils.shapenet import ShapeNet
ptc_layout_config={
    'title': {
        'text': 'test vis LiDAR',
        'font': {
            'size': 20,
            'color': 'rgb(150,150,150)',
        },
        'xanchor': 'left',
        'yanchor': 'top'},
    'paper_bgcolor': 'rgb(0,0,0)',
    'width' : 800,
    'height' : 800,
    'margin' : {
        'l': 20,
        'r': 20,
        'b': 20,
        't': 20
    },
    'legend': {
        'font':{
            'size':20,
            'color': 'rgb(150,150,150)',
        },
        'itemsizing': 'constant'
    },
    "hoverlabel": {
        "namelength": -1,
    },
    'showlegend': False,
    'scene': {
          'aspectmode': 'manual',
          'aspectratio': {'x': 0.75, 'y': 0.75, 'z': 0.05},
          'camera': {'eye': {'x': 0, 'y': 0, 'z': 0.5}},
          'xaxis': {'color': 'rgb(150,150,150)',
                    'dtick': 10,
                    'gridcolor': 'rgb(100,100,100)',
                    'range': [-150, 150],
                    'showbackground': False,
                    'showgrid': True,
                    'showline': False,
                    'showticklabels': True,
                    'tickmode': 'linear',
                    'tickprefix': 'x:'},
          'yaxis': {'color': 'rgb(150,150,150)',
                    'dtick': 10,
                    'gridcolor': 'rgb(100,100,100)',
                    'range': [-150, 150],
                    'showbackground': False,
                    'showgrid': True,
                    'showline': False,
                    'showticklabels': True,
                    'tickmode': 'linear',
                    'tickprefix': 'y:'},
          'zaxis': {'color': 'rgb(150,150,150)',
                    'dtick': 10,
                    'gridcolor': 'rgb(100,100,100)',
                    'range': [-15, 15],
                    'showbackground': False,
                    'showgrid': True,
                    'showline': False,
                    'showticklabels': True,
                    'tickmode': 'linear',
                    'tickprefix': 'z:'}},
}
def _init_(args):
	if not os.path.exists('checkpoints'):
		os.makedirs('checkpoints')
	if not os.path.exists('checkpoints/' + args.exp_name):
		os.makedirs('checkpoints/' + args.exp_name)
	if not os.path.exists('checkpoints/' + args.exp_name + '/' + 'models'):
		os.makedirs('checkpoints/' + args.exp_name + '/' + 'models')
	os.system('cp train_pcn.py checkpoints' + '/' + args.exp_name + '/' + 'main.py.backup')


class IOStream:
	def __init__(self, path):
		self.f = open(path, 'a')

	def cprint(self, text):
		print(text)
		self.f.write(text + '\n')
		self.f.flush()

	def close(self):
		self.f.close()

def test_one_epoch(device, model, test_loader):
	model.eval()
	test_loss = 0.0
	pred  = 0.0
	count = 0
	for i, data in enumerate(tqdm(test_loader)):
		partial_pc, complete_pc = data
        
		partial_pc = partial_pc.to(device)
		complete_pc = complete_pc.to(device)
		
		output = model(partial_pc)
		loss_val = ChamferDistanceLoss()(complete_pc, output['coarse_output']) + ChamferDistanceLoss()(complete_pc, output['fine_output'])

		test_loss += loss_val.item()
		count += 1

	test_loss = float(test_loss)/count
	return test_loss

def test(args, model, test_loader, textio):
	test_loss = test_one_epoch(args.device, model, test_loader)
	textio.cprint('Validation Loss: %f'%(test_loss))
def get_lidar(ptc, name='LiDAR', size=0.8):
    return [go.Scatter3d(
        x=ptc[:,0],
        y=ptc[:,1],
        z=ptc[:,2],
        mode='markers',
        marker_size=size,
        name=name)]
def display_plotly(input_pc, output, i=0):
	input_pc_ = get_lidar(input_pc, name='partial', size=0.8)
	output_ = get_lidar(output, name='complete', size=0.8)
	fig = go.Figure(data= input_pc_ + output_, layout=ptc_layout_config)
	fig.write_html(f"ldls_test_{i}.html")
    
def train_one_epoch(device, model, train_loader, optimizer):
	model.train()
	train_loss = 0.0
	pred  = 0.0
	count = 0
	for i, data in enumerate(tqdm(train_loader)):
		if i <= 50:
			alpha = 0.01
		elif 50 < i and i <= 100:
			alpha= 0.1
		elif 100 < i and i <= 150:
			alpha= 0.5
		elif 150 < i and i <= 200:
			alpha = 1

		partial_pc, complete_pc = data
		optimizer.zero_grad()
# 		display_plotly(partial_pc[0], complete_pc[0], i=0)
        
		partial_pc = partial_pc.to(device)
		complete_pc = complete_pc.to(device)
        

		output = model(partial_pc)
		loss_val = ChamferDistanceLoss()(complete_pc, output['coarse_output']) + alpha*ChamferDistanceLoss()(complete_pc, output['fine_output'])

		# backward + optimize
		
		loss_val.backward()
		optimizer.step()

		train_loss += loss_val.item()
		count += 1

	train_loss = float(train_loss)/count
	return train_loss

def train(args, model, train_loader, test_loader, boardio, textio, checkpoint):
	learnable_params = filter(lambda p: p.requires_grad, model.parameters())
	if args.optimizer == 'Adam':
		optimizer = torch.optim.Adam(learnable_params)
	else:
		optimizer = torch.optim.SGD(learnable_params, lr=0.1)

	if checkpoint is not None:
		min_loss = checkpoint['min_loss']
		optimizer.load_state_dict(checkpoint['optimizer'])

	best_test_loss = np.inf

	for epoch in range(args.start_epoch, args.epochs):
		train_loss = train_one_epoch(args.device, model, train_loader, optimizer)
		test_loss = test_one_epoch(args.device, model, test_loader)

		if test_loss<best_test_loss:
			best_test_loss = test_loss
			snap = {'epoch': epoch + 1,
					'model': model.state_dict(),
					'min_loss': best_test_loss,
					'optimizer' : optimizer.state_dict(),}
			torch.save(snap, 'checkpoints/%s/models/best_model_snap.t7' % (args.exp_name))
			torch.save(model.state_dict(), 'checkpoints/%s/models/best_model.t7' % (args.exp_name))

		torch.save(snap, 'checkpoints/%s/models/model_snap.t7' % (args.exp_name))
		torch.save(model.state_dict(), 'checkpoints/%s/models/model.t7' % (args.exp_name))
		
		boardio.add_scalar('Train_Loss', train_loss, epoch+1)
		boardio.add_scalar('Test_Loss', test_loss, epoch+1)
		boardio.add_scalar('Best_Test_Loss', best_test_loss, epoch+1)

		textio.cprint('EPOCH:: %d, Traininig Loss: %f, Testing Loss: %f, Best Loss: %f'%(epoch+1, train_loss, test_loss, best_test_loss))

def options():
	parser = argparse.ArgumentParser(description='Point Completion Network')
	parser.add_argument('--exp_name', type=str, default='exp_pcn', metavar='N',
						help='Name of the experiment')
	parser.add_argument('--dataset_path', type=str, default='ModelNet40',
						metavar='PATH', help='path to the input dataset') # like '/path/to/ModelNet40'
	parser.add_argument('--eval', type=bool, default=False, help='Train or Evaluate the network.')

	# settings for input data
	parser.add_argument('--dataset_type', default='modelnet', choices=['modelnet', 'shapenet2'],
						metavar='DATASET', help='dataset type (default: modelnet)')
	parser.add_argument('--num_points', default=1024, type=int,
						metavar='N', help='points in point-cloud (default: 1024)')

	# settings for PCN
	parser.add_argument('--emb_dims', default=1024, type=int,
						metavar='K', help='dim. of the feature vector (default: 1024)')
	parser.add_argument('--detailed_output', default=True, type=bool,
						help='Coarse + Fine Output')

	# settings for on training
	parser.add_argument('--seed', type=int, default=1234)
	parser.add_argument('-j', '--workers', default=4, type=int,
						metavar='N', help='number of data loading workers (default: 4)')
	parser.add_argument('-b', '--batch_size', default=32, type=int,
						metavar='N', help='mini-batch size (default: 32)')
	parser.add_argument('--epochs', default=200, type=int,
						metavar='N', help='number of total epochs to run')
	parser.add_argument('--start_epoch', default=0, type=int,
						metavar='N', help='manual epoch number (useful on restarts)')
	parser.add_argument('--optimizer', default='Adam', choices=['Adam', 'SGD'],
						metavar='METHOD', help='name of an optimizer (default: Adam)')
	parser.add_argument('--resume', default='', type=str,
						metavar='PATH', help='path to latest checkpoint (default: null (no-use))')
	parser.add_argument('--pretrained', default='', type=str,
						metavar='PATH', help='path to pretrained model file (default: null (no-use))')
	parser.add_argument('--device', default='cuda:0', type=str,
						metavar='DEVICE', help='use CUDA if available')

	args = parser.parse_args()
	return args

def main():
	args = options()
	args.dataset_path = os.path.join(os.getcwd(), os.pardir, os.pardir, 'ModelNet40', 'ModelNet40')

	torch.backends.cudnn.deterministic = True
	torch.manual_seed(args.seed)
	torch.cuda.manual_seed_all(args.seed)
	np.random.seed(args.seed)

	boardio = SummaryWriter(log_dir='checkpoints/' + args.exp_name)
	_init_(args)

	textio = IOStream('checkpoints/' + args.exp_name + '/run.log')
	textio.cprint(str(args))

	trainset = ShapeNet(dataroot="/home/jan268/repo/LDLS_anomaly/learning3d/data/shapenet/PCN/", split="train", category="car")
	testset = ShapeNet(dataroot="/home/jan268/repo/LDLS_anomaly/learning3d/data/shapenet/PCN/", split="test", category="car")
    
# 	trainset = ClassificationData(ModelNet40Data(train=True))
# 	testset = ClassificationData(ModelNet40Data(train=False))
	train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=args.workers)
	test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.workers)

	if not torch.cuda.is_available():
		args.device = 'cpu'
	args.device = torch.device(args.device)

	# Create PointNet Model.
	model = PCN(emb_dims=args.emb_dims, detailed_output=args.detailed_output)

	checkpoint = None
	if args.resume:
		assert os.path.isfile(args.resume)
		checkpoint = torch.load(args.resume)
		args.start_epoch = checkpoint['epoch']
		model.load_state_dict(checkpoint['model'])

	if args.pretrained:
		assert os.path.isfile(args.pretrained)
		model.load_state_dict(torch.load(args.pretrained, map_location='cpu'))
	model.to(args.device)

	if args.eval:
		test(args, model, test_loader, textio)
	else:
		train(args, model, train_loader, test_loader, boardio, textio, checkpoint)

if __name__ == '__main__':
	main()