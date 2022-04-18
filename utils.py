import math
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

def write_ply_point(name, vertices):
	fout = open(name, 'w')
	fout.write("ply\n")
	fout.write("format ascii 1.0\n")
	fout.write("element vertex "+str(len(vertices))+"\n")
	fout.write("property float x\n")
	fout.write("property float y\n")
	fout.write("property float z\n")
	fout.write("end_header\n")
	for ii in range(len(vertices)):
		fout.write(str(vertices[ii,0])+" "+str(vertices[ii,1])+" "+str(vertices[ii,2])+"\n")
	fout.close()


def write_ply_point_normal(name, vertices, normals=None):
	fout = open(name, 'w')
	fout.write("ply\n")
	fout.write("format ascii 1.0\n")
	fout.write("element vertex "+str(len(vertices))+"\n")
	fout.write("property float x\n")
	fout.write("property float y\n")
	fout.write("property float z\n")
	fout.write("property float nx\n")
	fout.write("property float ny\n")
	fout.write("property float nz\n")
	fout.write("end_header\n")
	if normals is None:
		for ii in range(len(vertices)):
			fout.write(str(vertices[ii,0])+" "+str(vertices[ii,1])+" "+str(vertices[ii,2])+" "+str(vertices[ii,3])+" "+str(vertices[ii,4])+" "+str(vertices[ii,5])+"\n")
	else:
		for ii in range(len(vertices)):
			fout.write(str(vertices[ii,0])+" "+str(vertices[ii,1])+" "+str(vertices[ii,2])+" "+str(normals[ii,0])+" "+str(normals[ii,1])+" "+str(normals[ii,2])+"\n")
	fout.close()


def write_ply_triangle(name, vertices, triangles):
	fout = open(name, 'w')
	fout.write("ply\n")
	fout.write("format ascii 1.0\n")
	fout.write("element vertex "+str(len(vertices))+"\n")
	fout.write("property float x\n")
	fout.write("property float y\n")
	fout.write("property float z\n")
	fout.write("element face "+str(len(triangles))+"\n")
	fout.write("property list uchar int vertex_index\n")
	fout.write("end_header\n")
	for ii in range(len(vertices)):
		fout.write(str(vertices[ii,0])+" "+str(vertices[ii,1])+" "+str(vertices[ii,2])+"\n")
	for ii in range(len(triangles)):
		fout.write("3 "+str(triangles[ii,0])+" "+str(triangles[ii,1])+" "+str(triangles[ii,2])+"\n")
	fout.close()


def sample_points_triangle(vertices, triangles, num_of_points):
	epsilon = 1e-6
	triangle_area_list = np.zeros([len(triangles)],np.float32)
	triangle_normal_list = np.zeros([len(triangles),3],np.float32)
	for i in range(len(triangles)):
		#area = |u x v|/2 = |u||v|sin(uv)/2
		a,b,c = vertices[triangles[i,1]]-vertices[triangles[i,0]]
		x,y,z = vertices[triangles[i,2]]-vertices[triangles[i,0]]
		ti = b*z-c*y
		tj = c*x-a*z
		tk = a*y-b*x
		area2 = math.sqrt(ti*ti+tj*tj+tk*tk)
		if area2<epsilon:
			triangle_area_list[i] = 0
			triangle_normal_list[i,0] = 0
			triangle_normal_list[i,1] = 0
			triangle_normal_list[i,2] = 0
		else:
			triangle_area_list[i] = area2
			triangle_normal_list[i,0] = ti/area2
			triangle_normal_list[i,1] = tj/area2
			triangle_normal_list[i,2] = tk/area2
	
	triangle_area_sum = np.sum(triangle_area_list)
	sample_prob_list = (num_of_points/triangle_area_sum)*triangle_area_list

	triangle_index_list = np.arange(len(triangles))

	point_normal_list = np.zeros([num_of_points,6],np.float32)
	count = 0
	watchdog = 0

	while(count<num_of_points):
		np.random.shuffle(triangle_index_list)
		watchdog += 1
		if watchdog>100:
			print("infinite loop here!")
			return point_normal_list
		for i in range(len(triangle_index_list)):
			if count>=num_of_points: break
			dxb = triangle_index_list[i]
			prob = sample_prob_list[dxb]
			prob_i = int(prob)
			prob_f = prob-prob_i
			if np.random.random()<prob_f:
				prob_i += 1
			normal_direction = triangle_normal_list[dxb]
			u = vertices[triangles[dxb,1]]-vertices[triangles[dxb,0]]
			v = vertices[triangles[dxb,2]]-vertices[triangles[dxb,0]]
			base = vertices[triangles[dxb,0]]
			for j in range(prob_i):
				#sample a point here:
				u_x = np.random.random()
				v_y = np.random.random()
				if u_x+v_y>=1:
					u_x = 1-u_x
					v_y = 1-v_y
				ppp = u*u_x+v*v_y+base
				
				point_normal_list[count,:3] = ppp
				point_normal_list[count,3:] = normal_direction
				count += 1
				if count>=num_of_points: break

	return point_normal_list

def gumbel_softmax(logits, dim, temperature=1):
    """
    ST-gumple-softmax w/o random gumbel samplings
    input: [*, n_class]
    return: flatten --> [*, n_class] an one-hot vector
    """
    y = F.softmax(logits / temperature, dim=dim)

    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    beta = 3
    y_hard = (y_hard - beta*y).detach() + beta*y
    return y_hard

def loss(net_out, point_value):
	#netout (batch_size*z_num,points_size,1)
	#point (batch_size,points_size,1)
	net_out = net_out.view(point_value.shape[0],net_out.shape[0]//point_value.shape[0],
						point_value.shape[-2],point_value.shape[-1])#(batch_size,z_num,points_size,1)
	net_out = net_out.permute(0,2,1,3).squeeze()#(batch_size,points_size,z_num)
	#print(net_out.max())
	point_value = point_value.squeeze()#(batch_size,point_size)
	out_point_value = 1 - point_value.clone()#(batch_size,point_size)
	
	out_loss = torch.sum(net_out**2,dim =2)#(batch_size,point_size)
	one_hot = gumbel_softmax(net_out,2)##(batch_size,points_size,z_num)
	in_loss = torch.sum((net_out-one_hot)**2,dim=2)#(batch_size,point_size)

	z_num = one_hot.shape[-1]
	one_hot = one_hot.view(-1,z_num)#(batch_size*point_size,z_num)
	one_hot_Uniform_sum = torch.sum(one_hot,dim = 0)/torch.sum(one_hot)#(1,z_num)
	part_avg_loss = torch.mean((one_hot_Uniform_sum- 1/z_num)**2)
	#print(in_loss.shape)
	re_loss = torch.mean(point_value*in_loss + out_point_value*out_loss) + 10*part_avg_loss
	#re_loss = torch.sum(net_out)
	return re_loss
