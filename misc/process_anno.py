# import enum


def process_annotation():
    # to get start / end boundary - annotation path
    # anno_info_train_path = '/saat/Charades_for_SAAT/charades_sta_train.txt'
    # anno_info_test_path = '/saat/Charades_for_SAAT/charades_sta_test.txt'

    # for roi
    anno_info_train_path = '/saat/Charades_for_SAAT/charades_sta_train.txt'
    anno_info_test_path = '/saat/Charades_for_SAAT/charades_sta_test.txt'
    # to get the total length - annotation path
    # info_train_path = '/saat/Charades_for_SAAT/Charades_v1_train.csv'
    # info_test_path = '/saat/Charades_for_SAAT/Charades_v1_test.csv'

    # for roi
    info_train_path = '/saat/Charades_for_SAAT/Charades_v1_train.csv'
    info_test_path = '/saat/Charades_for_SAAT/Charades_v1_test.csv'
    with open(anno_info_train_path, 'r') as f:
        anno_info_train = f.readlines()
    with open(anno_info_test_path, 'r') as f:
        anno_info_test = f.readlines()

    with open(info_train_path, 'r') as f:
        len_info_train = f.readlines()
    with open(info_test_path, 'r') as f:
        len_info_test = f.readlines()
    # Get the length of each video
    dic_len_to_save = {}

    for i, line in enumerate(len_info_train):
        if i == 0:
            continue
        id_ = line.split(',')[0]
        len_ = line.split(',')[-1].split('\n')[0]
        dic_len_to_save[id_] = len_
    train_keys = list(dic_len_to_save.keys())
    for i, line in enumerate(len_info_test):
        if i == 0:
            continue
        id_ = line.split(',')[0]
        len_ = line.split(',')[-1].split('\n')[0]
        dic_len_to_save[id_] = len_

    anno_info_temp = {}

    id_old = ''
    # Process to get start / end event boundary
    for line in anno_info_train:
        id_ = line.split(' ')[0]
        start_t = line.split(' ')[1]
        end_t = line.split(' ')[2].split('##')[0]
        # val_ = {'start':start_t, 'end':end_t}
        len_ = dic_len_to_save[id_]
        if float(start_t) >= float(end_t):
            continue
        if float(end_t) > float(len_):
            val_ = {'start': float(start_t) / float(len_), 'end': 1.0}
        else:
            val_ = {'start': float(start_t) / float(len_), 'end': float(end_t) / float(len_)}
        if id_ == id_old and start_t != start_old and end_t != end_old:
            anno_info_temp[id_].append(val_)
        else:
            anno_info_temp[id_] = [val_]

        id_old = id_
        start_old = start_t
        end_old = end_t
    id_old = ''
    for line in anno_info_test:
        id_ = line.split(' ')[0]
        start_t = line.split(' ')[1]
        end_t = line.split(' ')[2].split('##')[0]
        # val_ = {'start':start_t, 'end':end_t}
        len_ = dic_len_to_save[id_]
        if float(start_t) >= float(end_t):
            continue
        val_ = {'start': float(start_t) / float(len_), 'end': float(end_t) / float(len_)}
        if id_ == id_old and start_t != start_old and end_t != end_old:
            anno_info_temp[id_].append(val_)
        else:
            anno_info_temp[id_] = [val_]

        id_old = id_
        start_old = start_t
        end_old = end_t

   
    
    return anno_info_temp, train_keys
    