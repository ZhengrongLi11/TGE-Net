import os
import os.path

import SimpleITK as sitk
import numpy as np


IMG_EXTENSIONS = ['.nii.gz']


def is_nii_file(filename):
    """Checks if a file is an image.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()  # change upper case to lower case
    return any(filename_lower.endswith(ext) for ext in IMG_EXTENSIONS)


def find_classes(dir):
    photoClasses = [d for d in os.listdir(dir) if os.path.isfile(os.path.join(dir, d))]
    photoClasses.sort()
    photo_class_to_idx = {photoClasses[i]: i for i in range(len(photoClasses))}
    return photoClasses, photo_class_to_idx     # photoClasses 是列表, photo_class_to_idx 是字典
# eg:假设dir = "/path/to/dataset/,并且该目录结构有三个文件img1.jpg, img2.png, img3.tif
# 运行find_classes("/path/to/dataset/")
# 运行结果是：
# photoClasses: ['img1.jpg', 'img2.png', 'img3.tif']
# photo_class_to_idx: {'img1.jpg': 0, 'img2.png': 1, 'img3.tif': 2}

def make_dataset(dir, photo_class_to_idx):
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isfile(d):
            continue
        path = d
        item = (path, photo_class_to_idx[target])
        images.append(item)

    return images   # images是一个列表
# eg:假设dir = "/path/to/dataset/,并且该目录结构有三个文件img1.jpg, img2.png, img3.tif
# photo_class_to_idx: {'img1.jpg': 0, 'img2.png': 1, 'img3.tif': 2}
# 运行make_dataset("/path/to/dataset/", photo_class_to_idx)
# 运行结果是：
# [
#     ('/path/to/dataset/img1.jpg', 0),
#     ('/path/to/dataset/img2.png', 1),
#     ('/path/to/dataset/img3.tif', 2)
# ]

def collect_nii_path(path):
    # walk all the .nii files in path
    all_file_list = []
    gci(path, all_file_list)
    all_file_list.append(path)

    return all_file_list


def gci(filepath, all_file_list):
    files = os.listdir(filepath)
    for fi in files:
        fi_d = os.path.join(filepath, fi)
        if os.path.isdir(fi_d):
            all_file_list.append(fi_d)
            gci(fi_d, all_file_list)

def nii_slides_loader(nii_file_path, num, transform=None):
    item = sitk.ReadImage(nii_file_path)        # 使用 SimpleITK 包中的 ReadImage 函数读取指定的 NIfTI 文件, SimpleITK 是一个用于处理医学图像的库, 特别适用于处理 NIfTI, DICOM 等常见的医学图像格式
    nii_slides = sitk.GetArrayFromImage(item)   # 将 SimpleITK 格式的图像转换为 NumPy 数组, 形状为 (depth, height, width)

    if transform is not None:
        nii_slides = transform(nii_slides)      # 对三维数据归一化, 再提取切片

    return nii_slides[num, :, :]                # 返回索引为 num 的切片

def matrix_resize(filein, sacle_size, crop_size, random_crop_para):
    # TODO:
    temp = np.reshape(filein, [sacle_size, sacle_size])


def normalize_nii(mrnp):
    matLPET = mrnp / mrnp.max() * 2.0 - 1 # mrnp / mrnp.max(): [0, 1],  mrnp / mrnp.max() * 2.0 - 1: [-1, 1]
    return matLPET


def load_set(path):
    classes, class_to_idx = find_classes(path)    # classes: 存放文件夹中所有文件名称的列表, class_to_idx: 是一个字典, 键是文件名称, 值是索引(0, 1, ...)
    loaded_set = make_dataset(path, class_to_idx) # loaded_set: 是一个列表, 列表的每一个元素的类型都是元组, 每个元组中有两个元素, 分别是文件名称和索引
    if len(loaded_set) == 0:
        raise (RuntimeError("Found 0 images in subfolders of: " + path + "\n" "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))
    return loaded_set


def seg_transform(seg):
    seg = np.where(seg > 0, 1, np.finfo(float).eps)
    return seg