import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import sys
sys.path.append('/saat/code')
from maskrcnn_benchmark.config import cfg
sys.path.append('/saat/code/demo')
from predictor import COCODemo
import cv2
import os
import h5py
import process_anno
from tqdm import tqdm
import json
import argparse
config_file = "/saat/code/configs/caffe2/e2e_faster_rcnn_R_101_FPN_1x_caffe2.yaml"

# update the config options with the config file
cfg.merge_from_file(config_file)
# manual override some options
cfg.merge_from_list(["MODEL.DEVICE", "cuda:0"])

coco_demo = COCODemo(
    cfg,
    min_image_size=800,
    confidence_threshold=0.3,
)

if __name__ == '__main__':
	# parser = argparse.ArgumentParser()z
	# parser.add_argument('--cnt', type=int, default=0)
	# opt = parser.parse_args()
	frame_path = '/saat/charades_frames_jpg_fps_2' #  /saat/Charades_for_SAAT/Charades_frames # make dir
	feat_path = '/saat/Charades_for_SAAT_unsup_set_11'# /saat/Charades_for_SAAT/roi_feat_fc # make dir
	# anno_mapping = '/saat/charades_train_mapping_for_vid_cap_for_3d_{}.json'.format(str(opt.cnt))
	anno_mapping = '/saat/charades_vid_cap_mapping_unsup_mnli_bartsumm/charades_train_mapping_for_vid_cap_unsup_set_11.json'
	# anno = process_anno.process_annotation()
	tmp_ = []
	# vid_names = os.listdir(frame_path)
	# vid_names = sorted(vid_names)

	if not os.path.exists(feat_path):
		os.makedirs(feat_path)

	# roi_feats_h5 = h5py.File(os.path.join(feat_path, 'roi_feats_{}.h5'.format(str(opt.cnt))), 'w')
	# roi_box_h5 = h5py.File(os.path.join(feat_path, 'roi_box_{}.h5'.format(str(opt.cnt))), 'w')
	roi_feats_h5 = h5py.File(os.path.join(feat_path, 'roi_feats_set_11.h5'), 'w')
	roi_box_h5 = h5py.File(os.path.join(feat_path, 'roi_box_set_11.h5'), 'w')

	with open(anno_mapping, 'r') as f:
		tmp = json.load(f)
	
	vid_names = list(tmp.keys())


	print('start')
	cnt = 0
	for vid in tqdm(vid_names):
		# print('{}/({}, {})'.format(cnt, opt.start_idx, opt.end_idx))
		curr_fr_path = os.path.join(frame_path, vid + '.mp4')
		fr_lst = os.listdir(curr_fr_path)

		fr_len = len(fr_lst)

		tmp_bd = tmp[vid]
		segments = tmp_bd['segment']
		duration = tmp_bd['duration']
		print(cnt)
		for seg in segments:
			start = seg[0]
			end = seg[1]

			start_n = float(seg[0]) / float(duration)
			end_n = float(seg[0]) / float(duration)

			fr_start = start_n * fr_len
			fr_end = end_n * fr_len

			half_fr = (fr_start + fr_end) / 2
			half_fr = int(half_fr)

			if half_fr == 0:
				half_fr = 1
			elif half_fr >= fr_len:
				half_fr = fr_len - 1

			tmppath = os.path.join(curr_fr_path, str(half_fr).zfill(5) + '.jpg')
			img = cv2.imread(tmppath)

			# print(tmppath)
			# print(img.shape)

			result, top_preds, top_roi_feats = coco_demo.run_on_opencv_image(img)

			if top_roi_feats.shape[0] > 0:

				roi_feats_h5.create_dataset(str(cnt), data=top_roi_feats.numpy(), dtype='f4')
				roi_box_h5.create_dataset(str(cnt), data=top_preds.bbox.numpy(), dtype='f4')
				print(top_roi_feats.numpy().shape, top_preds.bbox.numpy().shape)
			cnt += 1
			if top_roi_feats.shape[0] == 0:
				tmp_.append({vid: [seg, cnt]})
	# print(tmp_)



		


		
		# for i in range(14, 15):
		# 	tmppath = os.path.join(curr_fr_path, str(i)+'.jpg')
		# 	img = cv2.imread(tmppath)
		# 	print(tmppath)
		# 	print(img.shape)
		# 	result, top_preds, top_roi_feats = coco_demo.run_on_opencv_image(img)
		# 	if top_roi_feats.shape[0] > 0:
		# 		roi_feats_h5.create_dataset(str(cnt), data=top_roi_feats.numpy(), dtype='f4')
		# 		roi_box_h5.create_dataset(str(cnt), data=top_preds.bbox.numpy(), dtype='f4')


	print('done')
	roi_feats_h5.close()
	roi_box_h5.close()
