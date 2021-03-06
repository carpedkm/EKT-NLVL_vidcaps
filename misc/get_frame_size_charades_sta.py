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
import json

if __name__ == '__main__':
	path_ = '/saat/charades_frames_jpg_fps_2' #  /saat/Charades_for_SAAT/Charades_frames # make dir
	anno_path = '/saat/Charades_for_SAAT_unsup_set_10/charades_train_mapping_for_vid_cap_unsup_set_10.json'
	# not existing in RoI
	# todel= [{'0BX9N': [[0.0, 9.8], 78]}, {'0F0WE': [[1.3, 10.4], 109]}, {'0WLCJ': [[1.8, 7.2], 242]}, {'1DNAX': [[0.4, 6.9], 370]}, {'1GQAJ': [[1.8, 17.7], 394]}, {'1LZ53': [[1.7, 13.0], 428]}, {'1TAMK': [[3.2, 9.1], 478]}, {'1W2NR': [[22.0, 29.1], 506]}, {'2FL0X': [[1.0, 6.8], 639]}, {'2FL0X': [[2.8, 9.6], 640]}, {'2H9YB': [[16.7, 21.6], 657]}, {'2MJ72': [[11.0, 17.0], 699]}, {'2Q5Y2': [[0.0, 14.6], 719]}, {'35W7G': [[14.1, 23.6], 803]}, {'3Q3YY': [[1.3, 13.8], 925]}, {'3QXPC': [[0.6, 7.0], 929]}, {'3YY88': [[0.0, 13.7], 1000]}, {'406LH': [[5.7, 17.9], 1023]}, {'406LH': [[5.9, 17.4], 1024]}, {'40DSU': [[0.0, 4.9], 1025]}, {'40DSU': [[1.5, 7.8], 1026]}, {'42MC3': [[10.5, 16.8], 1050]}, {'42MC3': [[9.2, 16.0], 1052]}, {'469ZJ': [[0.0, 5.9], 1079]}, {'4GWNV': [[3.6, 12.8], 1151]}, {'4LDRK': [[22.4, 32.71], 1180]}, {'4S3UZ': [[14.3, 23.2], 1210]}, {'56ASU': [[0.2, 6.9], 1310]}, {'5AHQV': [[0.1, 4.8], 1357]}, {'5DYQR': [[4.3, 9.8], 1390]}, {'6BFKO': [[23.8, 28.42], 1636]}, {'6BUU6': [[29.2, 30.88], 1639]}, {'6ZSB2': [[0.0, 4.8], 1790]}, {'6ZSB2': [[0.6, 7.2], 1791]}, {'706BT': [[0.0, 5.5], 1795]}, {'70S6Y': [[5.2, 13.3], 1799]}, {'70S6Y': [[5.2, 15.4], 1800]}, {'74S1R': [[0.0, 6.5], 1821]}, {'74S1R': [[2.1, 8.4], 1822]}, {'759UE': [[2.6, 10.4], 1828]}, {'7P5R2': [[1.8, 16.1], 1975]}, {'8EU89': [[1.9, 7.6], 2156]}, {'8XXZZ': [[0.0, 9.5], 2296]}, {'9C4JX': [[0.0, 9.4], 2404]}, {'AGJH7': [[0.5, 8.4], 2658]}, {'AMLI4': [[18.5, 26.4], 2687]}, {'AOMNM': [[0.0, 5.1], 2699]}, {'APQSV': [[2.6, 9.7], 2708]}, {'APQSV': [[3.6, 13.8], 2709]}, {'AQ5M6': [[4.8, 9.5], 2720]}, {'AQXBN': [[2.2, 15.4], 2722]}, {'BQ6UI': [[7.6, 21.7], 2930]}, {'BUL4V': [[18.2, 26.1], 2957]}, {'C8BKE': [[3.1, 9.9], 3046]}, {'C8WLX': [[8.7, 16.8], 3052]}, {'CCCUJ': [[11.6, 18.9], 3085]}, {'CXO6P': [[5.7, 16.0], 3265]}, {'D19IR': [[11.5, 18.1], 3294]}, {'D7KU2': [[7.7, 14.1], 3337]}, {'DDWK5': [[0.0, 4.0], 3374]}, {'DEZJ5': [[0.0, 2.8], 3384]}, {'DH9JU': [[0.0, 14.8], 3408]}, {'DJ6ZW': [[13.2, 22.4], 3428]}, {'DUAOJ': [[6.9, 16.6], 3521]}, {'E6P07': [[1.5, 13.3], 3610]}, {'E6PSM': [[0.9, 9.6], 3612]}, {'EBHC9': [[0.0, 12.8], 3658]}, {'EHHRS': [[1.6, 6.8], 3708]}, {'EHYXP': [[2.2, 7.7], 3710]}, {'EI5M3': [[0.0, 8.2], 3714]}, {'EI5M3': [[1.1, 7.0], 3715]}, {'EIK9W': [[0.5, 6.8], 3719]}, {'FQAAB': [[0.0, 6.3], 4008]}, {'FR4K2': [[23.4, 30.1], 4019]}, {'FRXS5': [[0.0, 6.9], 4022]}, {'FV9AL': [[0.0, 6.0], 4044]}, {'FYDYO': [[10.4, 18.9], 4067]}, {'GIC6A': [[0.0, 2.7], 4238]}, {'GIC6A': [[0.0, 7.7], 4239]}, {'HM7J7': [[0.0, 4.4], 4523]}, {'HPAYB': [[5.2, 12.7], 4537]}, {'HPEE5': [[20.3, 24.21], 4539]}, {'I31V9': [[0.0, 4.5], 4607]}, {'I31V9': [[2.5, 7.4], 4608]}, {'I5L3Y': [[5.4, 11.7], 4620]}, {'I5L3Y': [[5.7, 13.5], 4621]}, {'IA5TC': [[2.3, 12.3], 4672]}, {'ILKXV': [[0.0, 4.8], 4757]}, {'JK4Q2': [[5.6, 15.0], 5007]}, {'JSLW5': [[0.0, 6.4], 5058]}, {'JTTAP': [[0.6, 11.0], 5063]}, {'JTZZW': [[0.0, 7.3], 5073]}, {'JYBGS': [[2.5, 9.1], 5102]}, {'KFGXC': [[0.0, 14.8], 5216]}, {'KPAS0': [[14.0, 20.0], 5276]}, {'KZ7Y8': [[1.8, 11.1], 5361]}, {'LAJ9K': [[0.9, 9.0], 5435]}, {'LE5F4': [[0.0, 6.2], 5460]}, {'LK3BW': [[0.0, 3.2], 5498]}, {'LK3BW': [[0.0, 3.5], 5499]}, {'LK3BW': [[0.0, 4.6], 5500]}, {'M5YLS': [[4.9, 9.9], 5638]}, {'MC6J7': [[19.3, 32.6], 5676]}, {'NFAA5': [[22.9, 31.25], 5972]}, {'NHTSB': [[23.6, 32.38], 5990]}, {'NIRNP': [[3.0, 8.5], 5997]}, {'NJU3G': [[2.0, 9.1], 6003]}, {'NKNVR': [[24.7, 34.1], 6008]}, {'NWFOF': [[0.0, 9.0], 6097]}, {'O0C2Z': [[3.4, 7.8], 6123]}, {'O45BC': [[0.6, 11.0], 6135]}, {'OEYA3': [[0.0, 11.2], 6211]}, {'OL2QI': [[3.7, 15.9], 6270]}, {'OQM2I': [[6.0, 11.9], 6305]}, {'OZPA9': [[0.4, 8.5], 6361]}, {'PDK24': [[0.0, 4.6], 6449]}, {'PZ2W1': [[2.1, 9.2], 6583]}, {'QCBIG': [[0.0, 13.4], 6690]}, {'QDZ38': [[2.4, 9.2], 6700]}, {'R0OI6': [[14.0, 25.4], 6868]}, {'RDHNQ': [[0.2, 5.0], 6942]}, {'RZLAZ': [[0.0, 8.3], 7105]}, {'SIUU5': [[0.6, 7.2], 7238]}, {'TCQXI': [[0.0, 4.0], 7452]}, {'TRFB0': [[0.0, 13.1], 7546]}, {'TW6NZ': [[1.6, 12.4], 7594]}, {'UG0TA': [[0.0, 5.2], 7747]}, {'UWL8I': [[14.6, 24.2], 7857]}, {'V3RAX': [[9.6, 21.2], 7917]}, {'VMOBC': [[6.5, 20.5], 8039]}, {'WCIBT': [[0.0, 2.6], 8209]}, {'WCIBT': [[0.0, 4.1], 8210]}, {'WV8SY': [[12.8, 20.0], 8367]}, {'WZO6V': [[0.0, 7.0], 8408]}, {'XJS1X': [[5.6, 12.3], 8589]}, {'XKPLB': [[0.3, 5.2], 8598]}, {'XWKKZ': [[0.2, 6.3], 8667]}, {'XXU6H': [[0.0, 6.2], 8683]}, {'XXU6H': [[0.1, 6.7], 8684]}, {'Y1BWP': [[3.6, 10.1], 8712]}, {'Y5NDR': [[1.0, 6.5], 8733]}, {'YBW3D': [[1.1, 9.4], 8768]}, {'YH85Z': [[15.4, 25.1], 8794]}, {'Z3DBQ': [[15.1, 22.8], 8942]}, {'Z5FNI': [[15.1, 22.7], 8946]}, {'Z5FNI': [[18.3, 26.1], 8947]}, {'ZSSNJ': [[0.0, 2.9], 9145]}]
	# delval = []
	# for elem in todel:
	# 	id_ = list(elem.keys())[0]
	# 	val_ = elem[id_][1]
	# 	delval.append(val_)

	with open(anno_path, 'r') as f:
		anno = json.load(f)

	to_save_path = '/saat/Charades_for_SAAT_unsup_set_10'
	if not os.path.exists(to_save_path):
		os.makedirs(to_save_path)
	charades_temp_fr_size = h5py.File(os.path.join(to_save_path, 'charades_unsup_set_10_fr_size.h5'), 'w')

	cnt = 0
	for vid in tqdm(list(anno.keys())):
		# print('{}/({}, {})'.format(cnt, opt.start_idx, opt.end_idx))
		curr_fr_path = os.path.join(path_, vid + '.mp4')
		segcnt = len(anno[vid]['segment'])
	
		for i in range(segcnt):
			tmppath = os.path.join(curr_fr_path, str(1).zfill(5) +'.jpg')
			print(tmppath)
			img = cv2.imread(tmppath)
			shape = img.shape
			print(shape)
			# break
			cnt += 1
			# if cnt not in delval:
			charades_temp_fr_size.create_dataset(str(cnt), data=np.array(shape), dtype='i8')
		
	print('done')
	charades_temp_fr_size.close()