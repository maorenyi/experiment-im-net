import os
import time
import math
import random
import numpy as np
import h5py

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable

from utils import *
from model import *
if torch.cuda.is_available():
	device = torch.device('cuda')
else:
	device = torch.device('cpu')

# !!! data
data = h5py.File('data/all_vox256_img/all_vox256_img_mini_train.hdf5','r')
data_points = (data['points_'+str(16)][:].astype(np.float32)+0.5)/256-0.5
data_values = data['values_'+str(16)][:].astype(np.float32)
data_voxels = data['voxels'][:]
#reshape to NCHW
data_voxels = np.reshape(data_voxels, [-1, 1, 64, 64, 64]).astype(np.float32)

# !!! model 
ef_dim = 32
gf_dim = 128
z_dim = 256
point_dim = 3
z_num = 4
model  = im_network(ef_dim, gf_dim, z_dim, point_dim, z_num).to(device)

shape_num = len(data_voxels)
batch_index_list = np.arange(shape_num)
shape_batch_size = 16
point_batch_size = 16*16*16
optimizer = torch.optim.Adam([{'params':model.parameters(),},\
	#{'params': G.parameters()},\
		],\
			lr=0.000001, betas=(0.5, 0.999))
checkpoint_path = "im_network.pth"
if os.path.exists(checkpoint_path):
	model.load_state_dict(torch.load(checkpoint_path))
	print(" [*] Load SUCCESS")
else:
	model.generator = torch.load("generator.pt")

for epoch in range(0, 2000):
		model.train()
		np.random.shuffle(batch_index_list)
		avg_loss_sp = 0
		avg_num = 0
		for idx in range(shape_num//shape_batch_size):
			dxb = batch_index_list[idx*shape_batch_size:(idx+1)*shape_batch_size]
			batch_voxels = data_voxels[dxb].astype(np.float32)
			point_coord = data_points[dxb]
			point_value = data_values[dxb]
			batch_voxels = torch.from_numpy(batch_voxels)
			point_coord = torch.from_numpy(point_coord)
			shape = point_coord.shape
			point_coord = point_coord.unsqueeze(1)
			point_coord = point_coord.expand(-1,z_num,-1,-1).contiguous().view(-1,shape[-2],shape[-1])
			point_value = torch.from_numpy(point_value)
			
			batch_voxels = batch_voxels.to(device)
			point_coord = point_coord.to(device)
			point_value = point_value.to(device)
			_, net_out = model(batch_voxels,None,point_coord)
			errSP= loss(net_out,point_value)
			errSP.backward()
			optimizer.step()

			avg_loss_sp += errSP.item()
			avg_num += 1
		print("loss_sp: %.6f, net_out: %.6f" % (avg_loss_sp/avg_num, net_out.max().item()))
	