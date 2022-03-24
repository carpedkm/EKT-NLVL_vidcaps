import imageio
# imageio.plugins.ffmpeg.download()
import numpy as np
import os
import argparse
import process_anno
from tqdm import tqdm
import torch
import torchvision.transforms as trn
from spatial_transforms import (
	Compose,ToTensor)
import json
def extract_frames(output, dirname, filenames, frame_num, anno):
	transform = Compose([trn.ToPILImage(),
	ToTensor()])
	"""Extract frames in a video. """
	#Read videos and extract features in batches
	for file_cnt, fname in tqdm(enumerate(filenames)):
		if fname[:-4] in list(anno.keys()):
			bd_info = anno[fname[:-4]]
		else:
			continue
		for bd_ in bd_info:
			bd_cnt = bd_['count']
			start_time = bd_['start_time']
			end_time = bd_['end_time']
			if float(start_time) >= float(end_time):
				continue
			vid = imageio.get_reader(os.path.join(output, dirname, fname), 'ffmpeg')
			
			frames_dir = os.path.join(output, 'Frames', fname[:-4] + '_' + str(bd_cnt))
			print(frames_dir)
			if not os.path.exists(frames_dir):
				os.makedirs(frames_dir)
			if len(os.listdir(frames_dir)) == frame_num:
				print('already existing vid')
				continue
			curr_frames=[]
			for frame in vid:
				if len(frame.shape)<3:
					frame = np.repeat(frame,3)
				# curr_frames.append(frame)
				curr_frames.append(transform(frame).unsqueeze(0))
			vid_len = len(curr_frames) - 1
			curr_frames = torch.cat(curr_frames, dim=0)
			print("Shape of frames: {0}".format(curr_frames.shape))
			print('vid_len', vid_len)

			st_fr = vid_len * start_time
			end_fr = vid_len * end_time

			idx = np.linspace(st_fr, end_fr, frame_num)
			idx = np.round(idx).astype(int).tolist()
			frames_to_save = curr_frames[idx,:,:,:]
			# print(frames_to_save)


			for frame_n in range(frame_num):
				curr_frame = frames_to_save[frame_n,...]
				# print('curr_frame', curr_frame.shape)
				imageio.imwrite(os.path.join(frames_dir, str(frame_n)+'.jpg'), curr_frame.permute(1,2,0))
				
			print('{}/{} done'.format(file_cnt, len(filenames)))
			assert len(os.listdir(frames_dir)) == frame_num, 'Wrong frame number...'

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--file_path', type=str, default='/saat/Charades_for_SAAT/')
	parser.add_argument('--dataset_name', type=str, default='Charades')
	parser.add_argument('--frame_per_video', type=int, default=28)
	# parser.add_argument('--start_idx', type=int, default=0)
	# parser.add_argument('--end_idx', type=int, default=1)
	opt = parser.parse_args()

	with open('/saat/renewed_charades_label.json', 'r') as f:
		anno_info = json.load(f)
	anno, train_keys = anno_info, list(anno_info.keys())
	namelist = os.listdir(os.path.join(opt.file_path, opt.dataset_name))
	namelist_to_pass = []
	for id_ in namelist:
		if id_[:-4] in train_keys:
			namelist_to_pass.append(id_)
	extract_frames(opt.file_path, opt.dataset_name, namelist_to_pass, opt.frame_per_video, anno)
