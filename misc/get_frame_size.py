import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import json
import cv2
import os
import h5py
import process_anno
from tqdm import tqdm
import numpy as np

if __name__ == '__main__':
	path_ = '/saat/Charades_for_SAAT/Frames_for_temp' #  /saat/Charades_for_SAAT/Charades_frames # make dir
	anno = os.listdir('/saat/Charades_for_SAAT/temp_char/feat_3d_temp_charades')


	
	charades_temp_fr_size = h5py.File('/saat/charades_temp_fr_size.h5', 'w')


	for cnt, vid in tqdm(enumerate(sorted(anno))):
		# print('{}/({}, {})'.format(cnt, opt.start_idx, opt.end_idx))
		curr_fr_path = os.path.join(path_, vid[:-4])
	
		for i in range(14, 15):
			tmppath = os.path.join(curr_fr_path, str(i)+'.jpg')
			print(tmppath)
			img = cv2.imread(tmppath)
			shape = img.shape
			print(shape)
			# break
			charades_temp_fr_size.create_dataset(str(cnt), data=np.array(shape), dtype='i8')
		
	print('done')
	charades_temp_fr_size.close()