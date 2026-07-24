from collections import defaultdict
from time import time
from data import BaseDataset
from data.utils import create_modal_mask
from data.nii_data_loader import nii_slides_loader, load_set, normalize_nii
import os
import os.path
import numpy as np
import cv2
import torch
import pickle


class BratsDataset(BaseDataset):
    def __init__(self, opt):
        # 1. 加载数据根目录
        if opt.isTrain:
            data_root = opt.dataroot
        else:
            data_root = opt.test_dataroot
        self.mode = opt.dataset_mode
        transform = normalize_nii  # 归一化处理
        loader = nii_slides_loader  # 加载NIfTI切片
        self.choose_slice_num = 78  # 中心切片位置
        self.resize = 256  # 图像尺寸
        # 定义9个切片的索引范围：从choose_slice_num-8到choose_slice_num+8，步长2（共9个切片）
        self.slice_indices = range(self.choose_slice_num - 8, self.choose_slice_num + 9, 2)

        # 定义模态路径
        t1_path = os.path.join(data_root, 't1')
        t2_path = os.path.join(data_root, 't2')
        flair_path = os.path.join(data_root, 'flair')
        t1ce_path = os.path.join(data_root, 't1ce')

        # 加载模态数据集
        self.t1_set = load_set(t1_path)
        self.t2_set = load_set(t2_path)
        self.flair_set = load_set(flair_path)
        self.t1ce_set = load_set(t1ce_path)

        # 数据集样本数量（以flair模态为准）
        self.n_data = len(self.flair_set)
        # 总数据量 = 样本数 × 每个样本的切片数（9个）
        self.total_samples = self.n_data * len(self.slice_indices)

        # 2. 固定模态顺序：输入为t1、t2、flair，目标为t1ce
        self.input_modal_names = ['t1', 't2', 'flair']  # 输入模态
        self.target_modal_name = 't1ce'  # 目标模态
        self.modal_order = self.input_modal_names + [self.target_modal_name]  # 完整顺序

        # 创建目标模态的掩码
        n_modal = len(self.modal_order)
        self.n_modal = n_modal
        self.modal_mask_dict = create_modal_mask(self.modal_order)

        # 3. 加载所有模态数据（每个样本加载9个切片）
        print('Loading BraTS Dataset for task: t1+t2+flair -> t1ce (9 slices per sample)...')
        start = time()
        cache_path = os.path.join(data_root, 'AE-GAN-text-t1ce-9slices-cache.pkl')  # 缓存文件路径
        if os.path.exists(cache_path):
            print('Loading data from cache:', cache_path)
            with open(cache_path, 'rb') as fin:
                self.data_dict = pickle.load(fin)
        else:
            print('Loading data from raw files...')
            self.data_dict = defaultdict(list)  # 键：模态名，值：所有切片数据列表
            for sample_idx in range(self.n_data):  # 遍历每个样本
                # 遍历9个切片索引
                for slice_i in self.slice_indices:
                    # 加载输入模态（t1、t2、flair）的当前切片
                    for modal in self.input_modal_names:
                        modal_path, _ = getattr(self, f'{modal}_set')[sample_idx]
                        # 加载第slice_i个切片
                        modal_img = loader(modal_path, num=slice_i, transform=transform)
                        modal_img = cv2.resize(modal_img, (self.resize, self.resize))
                        self.data_dict[modal].append((modal_img, slice_i))  # 存入列表
                    # 加载目标模态（t1ce）的当前切片
                    t1ce_path, _ = self.t1ce_set[sample_idx]
                    t1ce_img = loader(t1ce_path, num=slice_i, transform=transform)
                    t1ce_img = cv2.resize(t1ce_img, (self.resize, self.resize))
                    self.data_dict[self.target_modal_name].append((t1ce_img, slice_i))  # 存入列表
            # 保存缓存
            with open(cache_path, 'wb') as fin:
                pickle.dump(self.data_dict, fin)
        end = time()
        print(f'Finish Loading, total samples: {self.total_samples}, cost {end - start:.1f}s')

    def __getitem__(self, index):
        # 输入模态：t1、t2、flair
        A = []
        target_mask = self.modal_mask_dict[self.target_modal_name]  # 目标模态掩码

        for modal_name in self.input_modal_names:
            # 直接通过index获取当前切片的模态数据（已按样本+切片顺序存储）
            modal_numpy, slice_i = self.data_dict[modal_name][index]
            # 拼接模态数据与目标掩码（保持原逻辑）
            modal_with_mask = np.concatenate([modal_numpy[None], target_mask])
            A.append(torch.tensor(modal_with_mask, dtype=torch.float))

        # 目标模态：t1ce
        target_modal_numpy, slice_i = self.data_dict[self.target_modal_name][index]
        # 获取当前样本的目标模态路径（需计算样本索引）
        sample_idx = index // len(self.slice_indices)  # 总索引 → 样本索引, 范围是 0 到 total_samples-1
        target_path, _ = self.t1ce_set[sample_idx]
        slice_idx = self.slice_indices[index % len(self.slice_indices)]  # 当前切片索引, 范围是 0 到 8

        # 构建返回字典
        input = {
            'A': torch.cat(A),  # 拼接输入模态
            'B': torch.tensor(target_modal_numpy[None], dtype=torch.float),  # 目标模态
            'modal_names': self.modal_order,
            'B_path': target_path,
            'Slice_Index': slice_i + 1,  # index of slice
        }
        return input

    def __len__(self):
        # 总样本数 = 原始样本数 × 每个样本的切片数（9）
        return self.total_samples

    def get_modal_names(self):
        return self.modal_order