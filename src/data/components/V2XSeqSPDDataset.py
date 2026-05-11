import os
import torch
import numpy as np
import torch.utils.data as data
import pickle
import logging
import h5py
import open3d


def random_sample_with_distance_np(points, num_points):
    if num_points < len(points):
        pts_depth = np.linalg.norm(points[:, 0:3], axis=1)
        pts_near_flag = pts_depth < 80.0
        far_idxs_choice = np.where(pts_near_flag == 0)[0]
        near_idxs = np.where(pts_near_flag == 1)[0]
        choice = []
        if num_points > len(far_idxs_choice):
            near_idxs_choice = np.random.choice(near_idxs, num_points - len(far_idxs_choice), replace=False)
            choice = np.concatenate((near_idxs_choice, far_idxs_choice), axis=0) \
                if len(far_idxs_choice) > 0 else near_idxs_choice
        else:
            choice = np.arange(0, len(points), dtype=np.int32)
            choice = np.random.choice(choice, num_points, replace=False)
        np.random.shuffle(choice)
    else:
        choice = np.arange(0, len(points), dtype=np.int32)
        if num_points > len(points):
            extra_choice = np.random.choice(choice, num_points - len(points))
            choice = np.concatenate((choice, extra_choice), axis=0)
        np.random.shuffle(choice)
    return choice


class IO:
    @classmethod
    def get(cls, file_path):
        _, file_extension = os.path.splitext(file_path)

        if file_extension in ['.npy']:
            return cls._read_npy(file_path)
        elif file_extension in ['.pcd', '.ply']:
            return cls._read_pcd(file_path)
        elif file_extension in ['.h5']:
            return cls._read_h5(file_path)
        elif file_extension in ['.txt']:
            return cls._read_txt(file_path)
        elif file_extension in ['.bin']:
            return cls.read_bin_float32(file_path)
        else:
            raise Exception('Unsupported file extension: %s' % file_extension)

    # References: https://github.com/numpy/numpy/blob/master/numpy/lib/format.py
    @classmethod
    def _read_npy(cls, file_path):
        return np.load(file_path)

    # References: https://github.com/dimatura/pypcd/blob/master/pypcd/pypcd.py#L275
    # Support PCD files without compression ONLY!
    @classmethod
    def _read_pcd(cls, file_path):
        pc = open3d.io.read_point_cloud(file_path)
        ptcloud = np.array(pc.points)
        return ptcloud

    @classmethod
    def _read_txt(cls, file_path):
        return np.loadtxt(file_path)

    @classmethod
    def _read_h5(cls, file_path):
        f = h5py.File(file_path, 'r')
        return f['data'][()]

    @classmethod
    def read_bin_float32(cls, path_file):
        return np.fromfile(path_file, dtype=np.float32).reshape(-1, 4)[:, :3]

def read_pkl(pkl_path):
    with open(pkl_path, 'rb') as f:
        file = pickle.load(f)
        return file

def read_bin_int8(path_file):
    return np.fromfile(path_file, dtype=np.uint8).reshape(-1)


class V2XSeqSPD(data.Dataset):
    def __init__(self, data_path, train: bool, transform=None):
        folder_name = os.path.basename(data_path) + '/'
        data_path = os.path.dirname(data_path) + '/'
        self.data_root = data_path
        self.pc_path = data_path + 'infrastructure-side/velodyne_filtered/'
        self.gt_path = data_path + 'completion_dataset/__point_cloud/'
        self.color_path = data_path + 'completion_dataset/__semantic_label/'
        self.subset = "train" if train else "test"
        self.pkl_file_path = os.path.join(self.data_root + folder_name, f'{self.subset}.pkl')
        self.input_num = 26624

        file_dict = read_pkl(self.pkl_file_path)
        data_list = file_dict['data_list']

        self.file_list = []
        for i, item in enumerate(data_list):
            # if self.subset == "train" and i % 1000 != 0:
            #     continue
            file_path = item['inf_info']['lidar_path']
            taxonomy_id = item['intersection_loc']
            model_id = item['frame_id']
            self.file_list.append({
                'taxonomy_id': taxonomy_id,
                'model_id': model_id,
                'file_path': file_path
            })

    def pc_norm(self, pc):
        """ pc: NxC, return NxC """
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
        pc = pc / m
        return pc

    def label_remap(self, labels):
        # First map the original labels used during preprocessing.
        labels[labels==0] = 32
        labels[labels==5] = 1
        labels[labels==9] = 17
        labels[labels==13] = 151
        labels[labels==14] = 151
        labels[labels==15] = 151
        labels[labels==16] = 151
        labels[labels==19] = 151
        labels[labels==21] = 151
        labels[labels==27] = 151
        labels[labels==34] = 151
        labels[labels==36] = 136
        labels[labels==38] = 151
        labels[labels==40] = 151
        labels[labels==41] = 151
        labels[labels==42] = 151
        labels[labels==46] = 151
        labels[labels==52] = 151
        labels[labels==53] = 151
        labels[labels==59] = 151
        labels[labels==61] = 151
        labels[labels==66] = 17
        labels[labels==69] = 151
        labels[labels==76] = 151
        labels[labels==82] = 136
        labels[labels==86] = 151
        labels[labels==88] = 151
        labels[labels==90] = 151
        labels[labels==98] = 151
        labels[labels==100] = 151
        labels[labels==104] = 151
        labels[labels==105] = 151
        labels[labels==108] = 151
        labels[labels==115] = 151
        labels[labels==116] = 127
        labels[labels==119] = 151
        labels[labels==125] = 93
        labels[labels==132] = 151
        labels[labels==138] = 93
        labels[labels==139] = 151
        labels[labels==147] = 151
        
        # Remap categories to 19 classes.
        labels[labels==1] = 0
        labels[labels==2] = 1
        labels[labels==4] = 2
        labels[labels==6] = 3
        labels[labels==11] = 4
        labels[labels==12] = 5
        labels[labels==17] = 6
        labels[labels==20] = 7
        labels[labels==32] = 8
        labels[labels==43] = 9
        labels[labels==80] = 10
        labels[labels==83] = 11
        labels[labels==87] = 12
        labels[labels==93] = 13
        labels[labels==102] = 14
        labels[labels==127] = 15
        labels[labels==136] = 16
        labels[labels==150] = 17
        labels[labels==151] = 18
        
        assert np.isin(labels, np.arange(19)).all()
        
        return labels

    def __getitem__(self, idx):
        sample = self.file_list[idx]
        # Load input points.
        partial_data = IO.get(self.pc_path + sample['file_path']).astype(np.float32)
        sample_idx = random_sample_with_distance_np(partial_data, self.input_num)
        partial_data = partial_data[sample_idx]
        # partial_data = self.pc_norm(partial_data)

        # Load ground-truth data.
        gt_file_path = os.path.join(self.gt_path, os.path.basename(sample['file_path']))
        gt_label_path = os.path.join(self.color_path, os.path.basename(sample['file_path']))
        gt_data = IO.get(gt_file_path).astype(np.float32)
        gt_label = read_bin_int8(gt_label_path)
        gt_label = self.label_remap(gt_label)

        # Return input points and ground-truth labels.
        data = {'partial': torch.from_numpy(partial_data).float(), 'gt': torch.from_numpy(gt_data).float(), 'label': torch.from_numpy(gt_label).float()}

        gt = torch.cat([data['gt'], data['label'].unsqueeze(1)], dim=1)
        return data['partial'], gt

    def __len__(self):
        return len(self.file_list)
