from collections import defaultdict
from time import time
from data import BaseDataset
from data.utils import create_modal_mask, permute_modal_names
from data.nii_data_loader import nii_slides_loader, load_set, normalize_nii
import os
import os.path
import numpy as np
import cv2
import torch
import pickle


class BratsDataset(BaseDataset):
    def __init__(self, opt):
        # 1. load form nii file
        if opt.isTrain: # 判断是否处于训练模式
            data_root = opt.dataroot
        else:
            data_root = opt.test_dataroot
        self.mode = opt.dataset_mode
        transform = normalize_nii   # 对加载的 NIfTI 数据进行归一化处理
        loader = nii_slides_loader  # 加载三维的NIfTI数据特定的切片
        choose_slice_num = 78
        resize = 256

        flair_path = os.path.join(data_root, 'flair')
        t1_path = os.path.join(data_root, 't1')
        t1ce_path = os.path.join(data_root, 't1ce')
        t2_path = os.path.join(data_root, 't2')

        # 通过调用 load_set 函数分别加载四种模态的图像数据集
        self.flair_set = load_set(flair_path)
        self.t1_set = load_set(t1_path)
        self.t1ce_set = load_set(t1ce_path)
        self.t2_set = load_set(t2_path)

        self.n_data = len(self.flair_set)   # 数据集的样本数量, 等于 FLAIR 模态数据集的大小

        # 2. create modal mask
        modal_names = ['t1', 't1ce', 't2', 'flair']
        n_modal = len(modal_names)
        self.n_modal = n_modal
        self.modal_mask_dict = create_modal_mask(modal_names)   # 通过掩码标记哪些模态(模态包括T1, T2, T1CE, FLAIR)启用或禁用, 缺失模态[1, 256, 256]对应像素值为 1

        if self.mode == 'all':
            self.modal_permutations = permute_modal_names(modal_names)
            # 返回的排列组合(四种情况)为：
            # [['t1', 't1ce', 't2', 'flair'],   # self.modal_permutations[0]
            #  ['t1ce', 't2', 'flair', 't1'],   # self.modal_permutations[1]
            #  ['t2', 'flair', 't1', 't1ce'],   # self.modal_permutations[2]
            #  ['flair', 't1', 't1ce', 't2']]   # self.modal_permutations[3]
        elif self.mode == 'same':
            self.modal_permutations = modal_names # ['t1', 't1ce', 't2', 'flair']
        else:
            self.source = opt.source
            self.dst = opt.dst
            self.modal_permutations = [opt.source]

        # 3. load_all modal into memory
        # self.data_dict 字典中存储的 modal_img 图像是未带掩码标记的原始图像数据
        print('Loading BraTS Dataset with "{}" mode...'.format(self.mode))
        start = time()
        cache_path = os.path.join(data_root, 'AE-GAN-cache.pkl')   # 构建缓存文件的路径, 指定缓存文件的路径为 data_root/cache.pkl, 该文件将用于存储已加载的数据, 以便加速后续的数据读取过程
        if os.path.exists(cache_path):
            print('load data cache from: ', cache_path)
            with open(cache_path, 'rb') as fin:
                self.data_dict = pickle.load(fin)
        else:
            print('load data from raw')
            self.data_dict = defaultdict(list) # 自动初始化 不存在的键对应的 值 为列表
            for index in range(self.n_data):    # n_data 表示该数据集的样本数量
                for modal in ['t1', 't1ce', 't2', 'flair']:
                    modal_path, modal_target = getattr(self, modal+'_set')[index]   # getattr 函数用于访问对象的属性
                    modal_img = loader(modal_path, num=choose_slice_num, transform=transform) # 归一化并提取加载指定切片
                    modal_img = cv2.resize(modal_img, (resize, resize)) # [240, 240] --> [256, 256], modal_img.dtype: float64
                    self.data_dict[modal].append(modal_img)     # 最终字典 self.data_dict 里的键是模态名称, 值是一个列表, 包含该模态下的所有图像数据
            with open(cache_path, 'wb') as fin:
                pickle.dump(self.data_dict, fin)
        end = time()
        print('Finish Loading, cost {:.1f}s'.format(end - start))


    def __getitem__(self, index):

        if self.mode == 'all':
            modal_order = self.modal_permutations[index % len(self.modal_permutations)]     # index % len(self.modal_permutations)最终结果是一个按 0, 1, 2, 3 顺序重复的循环模式, 用来确定当前使用的是哪种模态组合

            # 三种模态(input_modal_names)生成一种模态(target_modal_name)
            input_modal_names = modal_order[:-1]
            target_modal_name = modal_order[-1]

            # get mask of tartget modal
            target_mask = self.modal_mask_dict[target_modal_name] # [4, 256, 256], 缺失模态(target_modal_name)所在通道数像素值为 1
            # append target modal mask to every input modal image array
            A = []

            for modal_name in input_modal_names:
                modal_numpy = self.data_dict[modal_name][index // len(self.modal_permutations)] # data_dict字典里的键是模态名称, 值是一个列表, 包含该模态下的所有图像数据, index // len(self.modal_permutations)用来确定当前使用的是哪个样本
                modal_with_mask = np.concatenate([modal_numpy[None], target_mask]) # modal_numpy[None]:[1, 256, 256], target_mask:[4, 256, 256] --> [5, 256, 256]
                A.append(torch.tensor(modal_with_mask, dtype=torch.float))  # 将每个输入模态的图像数据与目标模态的掩码拼接, 生成包含模态数据和掩码的 modal_with_mask, 并添加到 A 列表中
            # get ith target modal image array
            target_modal_numpy = self.data_dict[target_modal_name][index // len(self.modal_permutations)] # [256, 256]
            # 获取目标图像的路径
            target_modal_numpy_path = None
            for modal in ['t1', 't1ce', 't2', 'flair']:
                if modal == target_modal_name:
                    modal_path, _ = getattr(self, modal + '_set')[index // len(self.modal_permutations)]
                    target_modal_numpy_path = modal_path
                    break
            input = {
                'A': torch.cat(A), # A：输入模态的图像, [15, 256, 256]
                'B': torch.tensor(target_modal_numpy[None], dtype=torch.float), # B：目标模态的图像, [1, 256, 256], NumPy 数组 --> Pytorch 张量
                'modal_names': modal_order,
                'B_path': target_modal_numpy_path # 示例B_path: /root/data1/AE-GAN-main/data/MICCAI_BraTS2020_TrainingData295/flair/BraTS20_Training_200_flair.nii
            }
            return input

        elif self.mode == 'same': # 输入图像和目标模态图像相同(图像重建)
            modal_name = self.modal_permutations[index % len(self.modal_permutations)]
            modal_input = self.data_dict[modal_name][index // len(self.modal_permutations)]
            modal_input = torch.tensor(modal_input[None], dtype=torch.float)
            input = {
                'A': modal_input,
                'B': modal_input
            }
            return input

        elif self.mode == 'single':
            modal_input = self.data_dict[self.source][index]
            modal_target = self.data_dict[self.dst][index]
            modal_input = torch.tensor(modal_input[None], dtype=torch.float)
            modal_target = torch.tensor(modal_target[None], dtype=torch.float)
            input = {
                'A': modal_input,
                'B': modal_target
            }
            return input

    def __len__(self):
        return self.n_data * len(self.modal_permutations)

    def get_modal_names(self):
        return ['t1', 't1ce', 't2', 'flair']
