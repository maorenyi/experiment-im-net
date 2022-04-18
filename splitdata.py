# -*- coding: utf-8 -*-

import h5py
import numpy as np

f = h5py.File('./data/all_vox256_img/all_vox256_img_mini_train.hdf5','r')
print(f['values_16'][1,2000:3000])
f.close()