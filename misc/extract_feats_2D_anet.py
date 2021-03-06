import cv2
import imageio
# imageio.plugins.ffmpeg.download()
import numpy as np
import os
#from inceptionresnetv2 import inceptionresnetv2
# from torchvision.models import resnet101
import pretrainedmodels
model_name = 'inceptionresnetv2'
import torchvision.transforms as trn
import torch
import argparse
import process_anno
from tqdm import tqdm
import json
import argparse
from PIL import Image


class Identity(torch.nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

def extract_feats(file_path, filenames, frame_num, batch_size, save_path, anno):
	"""Extract 2D features (saved in .npy) for frames in a video."""
	net = pretrainedmodels.__dict__[model_name](num_classes=1001, pretrained='imagenet+background')
	# net = resnet101(pretrained=True)
	net.last_linear = Identity()
	# net = inceptionresnetv2(num_classes=1001, pretrained='imagenet+background')
	
	net.eval()
	net.cuda()
	transform = trn.Compose([trn.ToPILImage(),
		trn.Resize((299, 299)), # 299 for IRV2 # 224 for ResNet
		trn.ToTensor(),
		trn.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])])#trn.Normalize(net.mean, net.std)])
		
	# print("res101 Network loaded")
	print("inceptionresnetv2 Network loaded")	

	#Read videos and extract features in batches
	cnt = 0
	for fname in tqdm(filenames):
		# start / end time list
		# print(fname)
		# if fname[:-4] in list(anno.keys()):
		# 	bd_info = anno[fname[:-4]]
		# else:
		# 	continue
		imgs_dir = os.path.join(file_path, fname + '.mp4')
		all_imgs_lst = os.listdir(imgs_dir)
		fr_cnt = len(all_imgs_lst) - 1
		bd_info = anno[fname]['timestamps']
		duration = anno[fname]['duration']

		for bd_ in bd_info:
			
			# get each set of start / end time form list
			# cnt = bd_['count']
			start_time = bd_[0]
			end_time = bd_[1]
			
			start_time = float(start_time) / float(duration)
			end_time = float(end_time) / float(duration)
			# print('file name', fname, 'bd_info', bd_info)
			feat_file = os.path.join(save_path, str(cnt)+'.npy')
			# print('fname', fname)

			if os.path.exists(feat_file):
				cnt += 1
				continue
			
			start_frame = fr_cnt * start_time
			end_frame = fr_cnt * end_time
			# print('start frame, end frame', start_frame, end_frame)
			idx = np.linspace(start_frame + 1, end_frame + 1, frame_num)
			idx = np.round(idx).astype(int)
			# print(idx)
			opened_imgs = []
			for i in idx:
				img = transform(np.array(Image.open(os.path.join(imgs_dir, str(i).zfill(5) + '.png'))))
				opened_imgs.append(img.unsqueeze(0))
				# print(img.shape)
			curr_frames = torch.cat(opened_imgs, dim=0)
			# print(curr_frames.shape)

			curr_feats = []
			for i in range(0, frame_num, batch_size):
				curr_batch = curr_frames[i:i+batch_size,:,:,:].cuda()
				out = net(curr_batch)
				curr_feats.append(out.detach().cpu())
				# print("Appended {} features {}".format(i+1,out.shape))
			curr_feats = torch.cat(curr_feats, 0)
			# print(curr_feats.shape)
			del out
			np.save(feat_file,curr_feats.numpy())

			cnt += 1

if __name__ == "__main__":
	parser = argparse.ArgumentParser()

	anet_frame_path ='/saat/anet_frames_nas_mount'
	parser.add_argument('--frame_per_video', type=int, default=28)
	parser.add_argument('--batch_size', type=int, default=28)
	opt = parser.parse_args()

	# read in annotation file
	with open('/saat/ActivityNet_for_SAAT/anet_train_anno.json', 'r') as f:
		anno_info = json.load(f)

	anno, train_keys = anno_info, list(anno_info.keys())

	read_in_path = anet_frame_path
	save_path = os.path.join('ActivityNet_for_SAAT', 'Feature_2D')
	if not os.path.exists(save_path):
		os.makedirs(save_path)
	# extract_feats(read_in_path, train_keys, opt.frame_per_video, opt.batch_size, save_path, anno, opt.start, opt.end)
	extract_feats(read_in_path, train_keys, opt.frame_per_video, opt.batch_size, save_path, anno)
