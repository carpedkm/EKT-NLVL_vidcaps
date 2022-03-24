def get_mean(norm_value=255, dataset_name='activitynet'):
    # assert dataset_name in ['activitynet', 'kinetics']
    # print('dataset_name')
    if dataset_name == 'activitynet':
        return [
            114.7748 / norm_value, 107.7354 / norm_value, 99.4750 / norm_value
        ]
    elif dataset_name == 'kinetics':
        # Kinetics (10 videos for each class)
        return [
            110.63666788 / norm_value, 103.16065604 / norm_value,
            96.29023126 / norm_value
        ]
    elif dataset_name == 'Charades':
        # from charades-rgb python github
        return [0.485, 0.456, 0.406]


def get_std(norm_value=255, dataset_name='Charades'):
    # Kinetics (10 videos for each class)
    if dataset_name =='kinetics' or 'activitynet':
        return [
            38.7568578 / norm_value, 37.88248729 / norm_value,
            40.02898126 / norm_value
        ]
    elif dataset_name == 'Charades':
        return [0.229, 0.224, 0.225]