import cv2
import imageio
import numpy as np
import os
import sys # to get the 3D model

from model import generate_model
import torchvision.transforms as trn
import torch
import argparse
from mean import get_mean, get_std
from spatial_transforms import (
	Compose, Normalize, Scale, CenterCrop, CornerCrop, ToTensor)
import process_anno
from tqdm import tqdm
import json

class Identity(torch.nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

def extract_feats(opt, file_path, net, filenames, frame_num, batch_size, save_path, dataset_name, anno):
	"""Extract 3D features (saved in .npy) for a video. """
	
	net.fc = Identity()
	# print(net)
	# return
	net.eval()
	net.cuda()
	mean = get_mean(255, dataset_name='kinetics')
	std = get_std(255, dataset_name)
	# transform = Compose([trn.ToPILImage(),
	# 	Scale(112),
	# 	CornerCrop(112, 'c'),
	# 	ToTensor(),
	# 	Normalize(mean, std)])
	transform = Compose([trn.ToPILImage(),
		Scale(112),
		CornerCrop(112, 'c'),
		ToTensor(),
		Normalize(mean, std)])
	
		
	print("Network loaded")
	#Read videos and extract features in batches
	for file in tqdm(filenames):
		# start / end time list
		# print(fname)
		# if file[:-4] in list(anno.keys()):
		# 	bd_info = anno[file[:-4]]
		# else:
		# 	continue
		bd_info = anno[file]
		for cnt, bd_ in enumerate(bd_info):
			# get each set of start / end time form list
			start_time = bd_['start']
			end_time = bd_['end']
			feat_file = os.path.join(save_path, file + '_' + str(cnt) + '.npy')
			if os.path.exists(feat_file):
				print('already existing :', feat_file)
				continue
			print(os.path.join(file_path, file))
			vid = imageio.get_reader(os.path.join(file_path, file + '.mp4'), 'ffmpeg')
			
			curr_frames = []
			for frame in vid:
				if len(frame.shape)<3:
					frame = np.repeat(frame,3)
				curr_frames.append(transform(frame).unsqueeze(0))
			# get the amount of video frames
			fr_cnt = len(curr_frames) - 1
			# concatenation of frames
			curr_frames = torch.cat(curr_frames, dim=0)
			print("Shape of frames: {0}".format(curr_frames.shape))
			
			# print('curr_frame count', len(curr_frames))
			# start frame / end frame
			st_fr = fr_cnt * start_time
			end_fr = fr_cnt * end_time

			idx = np.linspace(st_fr, end_fr, frame_num) # .astype(int)
			idx = np.round(idx).astype(int)

			# for idx_, i in enumerate(idx):
			# 	if i >= len(curr_frames):
			# 		idx[idx_] = len(curr_frames) - 1

			print("Captured {} clips: {}".format(len(idx), curr_frames.shape))
			# print(len(curr_frames))
			curr_frames = curr_frames[idx,:,:,:] # sample the frames with the given indices
			# remapping the idx for getting the frames (sequence of 16 frames in 28 linearly sampled ones)
			to_get_idx = np.arange(8, 21) # len : 13
			curr_feats = []
			for i in range(0, len(to_get_idx), batch_size):
				print(i)
				curr_batch = [curr_frames[x-8:x+8,...].unsqueeze(0) for x in to_get_idx[i:i+batch_size]]
				
				curr_batch = torch.cat(curr_batch, dim=0).cuda()
				print('curr_batch shape', curr_batch.shape)
				# if i == 0:
				# 	continue
				# if i > 2:
				# 	break
				print(curr_batch.transpose(1,2).shape)
				out = net(curr_batch.transpose(1,2).cuda())
				curr_feats.append(out.detach().cpu())
				print("Appended {} features {}".format(i+1,out.shape))
			curr_feats = torch.cat(curr_feats, 0)
			del out
			#set_trace()	
			print(curr_feats.shape)
			np.save(feat_file,curr_feats.numpy())
			# print("Saved file {}\nExiting".format(file[:-4] + '.npy'))


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--model', type=str, default='resnext')
	parser.add_argument('--model_depth', type=int, default=101)
	parser.add_argument('--pretrain_path', type=str, default='/saat/misc/checkpoints/resnext-101-kinetics.pth')
	parser.add_argument('--n_classes', type=int, default=400) #
	parser.add_argument('--n_finetune_classes', type=int, default=400)
	parser.add_argument('--ft_begin_index', type=int, default=0)
	parser.add_argument('--resnet_shortcut', type=str, default='B')
	parser.add_argument('--resnext_cardinality', type=int, default=32)
	parser.add_argument('--sample_size', type=int, default=112)
	parser.add_argument('--sample_duration', type=int, default=16)
	parser.add_argument('--no_cuda', type=bool, default=False)
	parser.add_argument('--no_train', type=bool, default=True)

	parser.add_argument('--file_path', type=str, default='/saat/Charades_for_SAAT') 
	parser.add_argument('--dataset_name', type=str, default='Charades') 
	parser.add_argument('--frame_per_video', type=int, default=28) # 28
	# parser.add_argument('--start_idx', type=int, default=0) # 0
	# parser.add_argument('--end_idx', type=int, default=1) # -1
	parser.add_argument('--batch_size', type=int, default=13)

	opt = parser.parse_args()
	opt.arch = '{}-{}'.format(opt.model, opt.model_depth)
	with open('/saat/train_charades_with_boundary_annotations.json', 'r') as f:
		anno_info = json.load(f)
	anno, train_keys = anno_info, list(anno_info.keys())

	model = generate_model(opt)
	model_weights = torch.load(opt.pretrain_path)['state_dict']
	for old_key in list(model_weights.keys()):
		new_key = old_key.split('module.')[1]
		model_weights[new_key] = model_weights.pop(old_key)
	model.load_state_dict(model_weights)

	dir_for_vid = os.path.join(opt.file_path, opt.dataset_name)
	
	save_path = os.path.join(opt.file_path, 'Feature_3D')
	extract_feats(opt, dir_for_vid, model, train_keys, opt.frame_per_video, opt.batch_size, save_path, opt.dataset_name, anno)
