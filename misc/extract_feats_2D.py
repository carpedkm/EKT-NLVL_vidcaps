import cv2
import imageio
# imageio.plugins.ffmpeg.download()
import numpy as np
import os
#from inceptionresnetv2 import inceptionresnetv2
from torchvision.models import resnet101
import torchvision.transforms as trn
import torch
import argparse
import process_anno
from tqdm import tqdm

def extract_feats(file_path, filenames, frame_num, batch_size, save_path, anno):
	"""Extract 2D features (saved in .npy) for frames in a video."""
	#net = inceptionresnetv2(num_classes=1001, pretrained='imagenet+background', load_path='./pretrained_models/inceptionresnetv2-520b38e4.pth')
	net = resnet101(pretrained=True)
	net.eval()
	net.cuda()
	transform = trn.Compose([trn.ToPILImage(),
		trn.Resize((224, 224)), # 299 for IRV2
		trn.ToTensor(),
		trn.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])])#trn.Normalize(net.mean, net.std)])
		
	print("res101 Network loaded")
	#Read videos and extract features in batches
	for fname in tqdm(filenames):
		# start / end time list
		# print(fname)
		if fname[:-4] in list(anno.keys()):
			bd_info = anno[fname[:-4]]
		else:
			continue
		for cnt, bd_ in enumerate(bd_info):
			# get each set of start / end time form list
			start_time = bd_['start']
			end_time = bd_['end']

			feat_file = os.path.join(save_path, fname[:-4] + '_' + str(cnt)+'.npy')
			# print('fname', fname[:-4])

			if os.path.exists(feat_file):
				continue
			vid = imageio.get_reader(os.path.join(file_path, fname), 'ffmpeg')
			curr_frames = []
			for frame in vid:
				if len(frame.shape)<3:
					frame = np.repeat(frame,3)
				curr_frames.append(transform(frame).unsqueeze(0))
			fr_cnt = len(curr_frames)
			# print('curr_frame count', len(curr_frames))
			# start frame / end frame
			st_fr = fr_cnt * start_time
			end_fr = fr_cnt * end_time

			curr_frames = torch.cat(curr_frames, dim=0)
			# print("Shape of frames: {0}".format(curr_frames.shape))
			# get it by linspace, and rounding if the count of frames is smaller than sampling amount
			idx = np.linspace(st_fr, end_fr, frame_num) # .astype(int)
			idx = np.round(idx).astype(int)

			for idx_, i in enumerate(idx):
				if i >= len(curr_frames):
					idx[idx_] = len(curr_frames) - 1

			curr_frames = curr_frames[idx,:,:,:].cuda()
			# print("Captured {} frames: {}".format(frame_num, curr_frames.shape))
			# print('frame count', len(curr_frames))
			curr_feats = []
			for i in range(0, frame_num, batch_size):
				curr_batch = curr_frames[i:i+batch_size,:,:,:]
				out = net(curr_batch)
				curr_feats.append(out.detach().cpu())
				# print("Appended {} features {}".format(i+1,out.shape))
			curr_feats = torch.cat(curr_feats, 0)
			del out
			np.save(feat_file,curr_feats.numpy())
			# print("Saved file {}\nExiting".format(fname[:-4] + '.npy'))
			# if fname[:-4] == '138LG':
			# 	break

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--file_path', type=str, default='./Data') # /saat/Charades_for_SAAT
	parser.add_argument('--dataset_name', type=str, default='YouTubeClips') # Charades
	parser.add_argument('--frame_per_video', type=int, default=28)
	# parser.add_argument('--start_idx', type=int, default=0)
	# parser.add_argument('--end_idx', type=int, default=1)
	parser.add_argument('--batch_size', type=int, default=1)
	opt = parser.parse_args()

	anno, train_keys = process_anno.process_annotation()

	save_path = os.path.join(opt.file_path, 'Feature_2D')
	namelist = os.listdir(os.path.join(opt.file_path, opt.dataset_name))
	namelist_to_pass = []
	for id_ in namelist:
		if id_[:-4] in train_keys:
			namelist_to_pass.append(id_)
	
	# print(namelist_to_pass)
	
	read_in_path = os.path.join(opt.file_path, opt.dataset_name)
	extract_feats(read_in_path, namelist_to_pass, opt.frame_per_video, opt.batch_size, save_path, anno)
