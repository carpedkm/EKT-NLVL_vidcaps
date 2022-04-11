import cv2
# import imageio
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

from PIL import Image

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
	cnt = 0
		
	print("Network loaded")
	#Read videos and extract features in batches
	for file in tqdm(filenames):
		vid_dir = os.path.join(file_path, file + '.mp4')
		vid_all = os.listdir(vid_dir)
		fr_cnt = len(vid_all) - 1
		duration = anno[file]['duration']
		bd_info = anno[file]['segment']
		for bd_ in bd_info:
			# get each set of start / end time form list
			start_time = bd_[0]
			end_time = bd_[1]
			

			start_time = start_time / duration
			end_time = end_time / duration


			feat_file = os.path.join(save_path, str(cnt) + '.npy')
			if os.path.exists(feat_file):
				print('already existing :', feat_file)
				cnt += 1
				continue
			# print(os.path.join(file_path, file))

			start_frame = fr_cnt * start_time
			end_frame = fr_cnt * end_time

			idx = np.linspace(start_frame + 1, end_frame + 1, frame_num)
			idx = np.round(idx).astype(int)
			
			opened_imgs = []
			for i in idx:
				img = transform(np.array(Image.open(os.path.join(vid_dir, str(i).zfill(5) + '.jpg'))))
				opened_imgs.append(img.unsqueeze(0))

			curr_frames = torch.cat(opened_imgs, dim=0)
			# remapping the idx for getting the frames (sequence of 16 frames in 28 linearly sampled ones)
			to_get_idx = np.arange(8, 21) # len : 13
			curr_feats = []
			for i in range(0, len(to_get_idx), batch_size):
				print(i)
				curr_batch = [curr_frames[x-8:x+8,...].unsqueeze(0) for x in to_get_idx[i:i+batch_size]]
				
				curr_batch = torch.cat(curr_batch, dim=0).cuda()
				# print('curr_batch shape', curr_batch.shape)
				# if i == 0:
				# 	continue
				# if i > 2:
				# 	break
				# print(curr_batch.transpose(1,2).shape)
				out = net(curr_batch.transpose(1,2).cuda())
				curr_feats.append(out.detach().cpu())
				print("Appended {} features {}".format(i+1,out.shape))
			curr_feats = torch.cat(curr_feats, 0)
			del out
			#set_trace()	
			cnt += 1
			# print(curr_feats.shape)
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
	parser.add_argument('--set', type=int)
	opt = parser.parse_args()
	opt.arch = '{}-{}'.format(opt.model, opt.model_depth)
	with open('/saat/charades_vid_cap_mapping_unsup_mnli_bartsumm/charades_train_mapping_for_vid_cap_unsup_set_{}.json'.format(opt.set), 'r') as f:
		anno_info = json.load(f)
	anno, train_keys = anno_info, list(anno_info.keys())

	model = generate_model(opt)
	model_weights = torch.load(opt.pretrain_path)['state_dict']
	for old_key in list(model_weights.keys()):
		new_key = old_key.split('module.')[1]
		model_weights[new_key] = model_weights.pop(old_key)
	model.load_state_dict(model_weights)
	dir_for_vid = '/saat/charades_frames_jpg_fps_2'
	# namelist = os.listdir(dir_for_vid)
	# namelist_to_pass = []
	
	# for id_ in namelist:
	# 	if id_[:-4] in train_keys:
	# 		namelist_to_pass.append(id_)
	# namelist_to_pass = sorted(namelist_to_pass)
	save_path = os.path.join(opt.file_path, 'Feature_3D_{}'.format(str(opt.set)))
	if not os.path.exists(save_path):
		os.makedirs(save_path)
	extract_feats(opt, dir_for_vid, model, train_keys, opt.frame_per_video, opt.batch_size, save_path, opt.dataset_name, anno)
