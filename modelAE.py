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

import mcubes

from utils import *
from model import *
#pytorch 1.2.0 implementation

class IM_AE(object):
	def __init__(self, config):
		#progressive training
		#1-- (16, 16*16*16)
		#2-- (32, 16*16*16)
		#3-- (64, 16*16*16*4)
		self.sample_vox_size = config.sample_vox_size
		if self.sample_vox_size==16:
			self.load_point_batch_size = 16*16*16
			self.point_batch_size = 16*16*16
			self.shape_batch_size = 32
		elif self.sample_vox_size==32:
			self.load_point_batch_size = 16*16*16
			self.point_batch_size = 16*16*16
			self.shape_batch_size = 32
		elif self.sample_vox_size==64:
			self.load_point_batch_size = 16*16*16*4
			self.point_batch_size = 16*16*16
			self.shape_batch_size = 32
		self.input_size = 64 #input voxel grid size

		#batch_size
		self.shape_batch_size = 32
		#!!! z_num definition
		self.ef_dim = 32
		self.gf_dim = 128
		self.z_dim = 256
		self.point_dim = 3
		self.z_num = 4
		
		data_hdf5_name = "data/all_vox256_img/all_vox256_img_mini_train.hdf5"
		if os.path.exists(data_hdf5_name):
			data_dict = h5py.File(data_hdf5_name, 'r')
			self.data_points = (data_dict['points_'+str(self.sample_vox_size)][:].astype(np.float32)+0.5)/256-0.5
			self.data_values = data_dict['values_'+str(self.sample_vox_size)][:].astype(np.float32)
			self.data_voxels = data_dict['voxels'][:]
			#reshape to NCHW
			self.data_voxels = np.reshape(self.data_voxels, [-1,1,self.input_size,self.input_size,self.input_size])
		else:
			print("error: cannot load "+data_hdf5_name)
			exit(0)


		if torch.cuda.is_available():
			self.device = torch.device('cuda')
			torch.backends.cudnn.benchmark = True
		else:
			self.device = torch.device('cpu')

		#build model
		self.im_network = im_network(self.ef_dim, self.gf_dim, self.z_dim, self.point_dim, self.z_num)
		self.im_network.to(self.device)
		#print params
		#for param_tensor in self.im_network.state_dict():
		#	print(param_tensor, "\t", self.im_network.state_dict()[param_tensor].size())
		# !!! optimizer
		self.optimizer = torch.optim.Adam(self.im_network.parameters(), lr=config.learning_rate, betas=(config.beta1, 0.999))
		#pytorch does not have a checkpoint manager
		#have to define it myself to manage max num of checkpoints to keep
		self.max_to_keep = 2
		#loss

		#keep everything a power of 2
		self.cell_grid_size = 4
		self.frame_grid_size = 64
		self.real_size = self.cell_grid_size*self.frame_grid_size #=256, output point-value voxel grid size in testing
		self.test_size = 32 #related to testing batch_size, adjust according to gpu memory size
		self.test_point_batch_size = self.test_size*self.test_size*self.test_size #do not change
		##region 
		#get coords for training
		dima = self.test_size
		dim = self.frame_grid_size
		self.aux_x = np.zeros([dima,dima,dima],np.uint8)
		self.aux_y = np.zeros([dima,dima,dima],np.uint8)
		self.aux_z = np.zeros([dima,dima,dima],np.uint8)
		multiplier = int(dim/dima)
		multiplier2 = multiplier*multiplier
		multiplier3 = multiplier*multiplier*multiplier
		for i in range(dima):
			for j in range(dima):
				for k in range(dima):
					self.aux_x[i,j,k] = i*multiplier
					self.aux_y[i,j,k] = j*multiplier
					self.aux_z[i,j,k] = k*multiplier
		self.coords = np.zeros([multiplier3,dima,dima,dima,3],np.float32)
		for i in range(multiplier):
			for j in range(multiplier):
				for k in range(multiplier):
					self.coords[i*multiplier2+j*multiplier+k,:,:,:,0] = self.aux_x+i
					self.coords[i*multiplier2+j*multiplier+k,:,:,:,1] = self.aux_y+j
					self.coords[i*multiplier2+j*multiplier+k,:,:,:,2] = self.aux_z+k
		self.coords = (self.coords.astype(np.float32)+0.5)/dim-0.5
		self.coords = np.reshape(self.coords,[multiplier3,self.test_point_batch_size,3])
		self.coords = torch.from_numpy(self.coords)
		self.coords = self.coords.to(self.device)
		
		self.sampling_threshold = 0.5 #final marching cubes threshold
		##endregion
	@property
	def model_dir(self):
		return "{}_ae_{}".format(self.dataset_name, self.input_size)

	def train(self, config):
		# !!! load previous checkpoint
		checkpoint_path = "im_network.pth"
		if os.path.exists(checkpoint_path):
			self.im_network.load_state_dict(torch.load(checkpoint_path))
			print(" [*] Load SUCCESS")
		else:
			self.im_network.generator = torch.load("generator.pt")
			
		shape_num = len(self.data_voxels)
		batch_index_list = np.arange(shape_num)
		
		print("\n\n----------net summary----------")
		print("training samples   ", shape_num)
		print("-------------------------------\n\n")
		
		start_time = time.time()
		training_epoch = 150
		batch_num = int(shape_num/self.shape_batch_size)
		point_batch_num = int(self.load_point_batch_size/self.point_batch_size)
		for epoch in range(0, training_epoch):
			self.im_network.train()
			np.random.shuffle(batch_index_list)
			avg_loss_sp = 0
			avg_num = 0
			for idx in range(batch_num):
				dxb = batch_index_list[idx*self.shape_batch_size:(idx+1)*self.shape_batch_size]
				batch_voxels = self.data_voxels[dxb].astype(np.float32)
				if point_batch_num==1:
					point_coord = self.data_points[dxb]
					point_value = self.data_values[dxb]
				else:
					which_batch = np.random.randint(point_batch_num)
					point_coord = self.data_points[dxb,which_batch*self.point_batch_size:(which_batch+1)*self.point_batch_size]
					point_value = self.data_values[dxb,which_batch*self.point_batch_size:(which_batch+1)*self.point_batch_size]

				batch_voxels = torch.from_numpy(batch_voxels)
				point_coord = torch.from_numpy(point_coord)
				shape = point_coord.shape
				point_coord = point_coord.unsqueeze(1)
				point_coord = point_coord.expand(-1,self.z_num,-1,-1).contiguous().view(-1,shape[-2],shape[-1])
				point_value = torch.from_numpy(point_value)

				batch_voxels = batch_voxels.to(self.device)
				point_coord = point_coord.to(self.device)
				point_value = point_value.to(self.device)

				self.im_network.zero_grad()
				_, net_out = self.im_network(batch_voxels, None, point_coord, is_training=True)
				errSP = loss(net_out, point_value)

				errSP.backward()
				self.optimizer.step()

				avg_loss_sp += errSP.item()
				avg_num += 1
			print(str(self.sample_vox_size)+" Epoch: [%2d/%2d] time: %4.4f, loss_sp: %.6f" % (epoch, training_epoch, time.time() - start_time, avg_loss_sp/avg_num))
			if epoch%10==1:
				self.test_1(config,"train_"+str(self.sample_vox_size)+"_"+str(epoch))
			# !!! model save
			# // if epoch%100 == 99:
			# //	torch.save(self.im_network.state_dict(),checkpoint_path)
	
	def test_1(self, config, name):
		multiplier = int(self.frame_grid_size/self.test_size)
		multiplier2 = multiplier*multiplier
		self.im_network.eval()
		t = np.random.randint(len(self.data_voxels))
		model_float = np.zeros([self.frame_grid_size+2,self.frame_grid_size+2,self.frame_grid_size+2],np.float32)
		batch_voxels = self.data_voxels[t:t+1].astype(np.float32)
		batch_voxels = torch.from_numpy(batch_voxels)
		batch_voxels = batch_voxels.to(self.device)
		z_vectors, _ = self.im_network(batch_voxels, None, None, is_training=False)
		for (num_part,z_vector) in enumerate(z_vectors):
			for i in range(multiplier):
				for j in range(multiplier):
					for k in range(multiplier):
						minib = i*multiplier2+j*multiplier+k
						point_coord = self.coords[minib:minib+1]
						_, net_out = self.im_network(None, z_vector, point_coord, is_training=False)
						#net_out = torch.clamp(net_out, min=0, max=1)
						model_float[self.aux_x+i+1,self.aux_y+j+1,self.aux_z+k+1] = np.reshape(net_out.detach().cpu().numpy(), [self.test_size,self.test_size,self.test_size])
		
			vertices, triangles = mcubes.marching_cubes(model_float, self.sampling_threshold)
			vertices = (vertices.astype(np.float32)-0.5)/self.frame_grid_size-0.5
			#output ply sum
			if not os.path.exists(config.sample_dir):
				for i in range(self.z_num):
					os.makedirs(config.sample_dir+"/"+str(i))
			write_ply_triangle(config.sample_dir+"/"+str(num_part)+"/"+name+"_"+str(num_part)+".ply", vertices, triangles)
		print("[sample]")