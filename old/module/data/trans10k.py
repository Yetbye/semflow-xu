import os
import os.path as osp
import torch
import numpy as np
import torch.utils.data as data
from PIL import Image
from .builder import build_palette


class Normalize(object):
    def __init__(self, mean=0.5, std=0.5):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        for k in sample.keys():
            if k in ['image', 'image_panseg', 'image_semseg']:
                sample[k] = (sample[k] - self.mean) / self.std
        return sample


'''
image为原图，gt_semseg为类别标签（原png），image_semseg为RBG掩码
'''

class Trans10k(data.Dataset):
    # Trans10k数据集类别名称（12类）
    # 注意：这里需要根据Trans10k数据集的实际情况进行调整
    CATEGORY_NAMES = [
        'Background', 'Shelf', 'Jar or Tank', 'Freezer', 'Window','Glass Door',
        'Eyeglass', 'Cup', 'Floor Glass', 'Glass Bow', 'Water Bottle', 'Storage Box'
    ]

    def __init__(
            self,
            data_root: str,
            split: str = 'val',
            transform=None,
            args_palette=None,
    ):
        print(f'init Trans10k dataset, split: {split}')

        self.data_root = data_root
        self.meta_data = {'category_names': self.CATEGORY_NAMES}  # 添加空字典作为默认值         ############
        self.split = split
        self.transform = transform
        self.post_norm = Normalize()
        self.palette = build_palette(args_palette[0], args_palette[1]) 
        self.num_classes = 12
        self.ignore_index = 255        #############
        
        # 设置图像和标注路径
        _img_dir = osp.join(data_root, split, 'images')
        _seg_dir = osp.join(data_root, split, 'masks_12')

        # 获取文件列表
        self.images = []
        self.semsegs = []

        # 遍历图像目录
        for img_name in sorted(os.listdir(_img_dir)):
            if img_name.endswith(('.jpg', '.png', '.jpeg')):
                base_name = osp.splitext(img_name)[0]
                # 根据命名规则生成掩码文件名
                seg_name = f"{base_name}_mask.png"
                seg_path = osp.join(_seg_dir, seg_name)

                # 确保掩码文件存在
                if osp.exists(seg_path):
                    self.images.append(osp.join(_img_dir, img_name))
                    self.semsegs.append(seg_path)
                else:
                    print(f"Warning: Mask file not found for {img_name}: {seg_path}")

        print(f'Processing {len(self.images)} images in Trans10k {split} set')

    def __len__(self):
        return len(self.images)

    def prepare_pm(self, x):
        """将分割标签转换为彩色图像"""
        assert len(x.shape) == 2
        h, w = x.shape
        pm = np.ones((h, w, 3)) * 255  # 默认白色背景

        clslist = np.unique(x).tolist()
        if 255 in clslist:  # 移除无效类别
            clslist.remove(255)

        for _c in clslist:
            _x, _y = np.where(x == _c)
            pm[_x, _y, :] = self.palette[int(_c) * 3:(int(_c) + 1) * 3]

        return pm

    def __getitem__(self, index):
        sample = {}

        # 加载图像
        _img = Image.open(self.images[index]).convert('RGB')
        sample['image'] = _img

        # 加载分割标注
        gt_semseg = Image.open(self.semsegs[index])
        gt_semseg_array = np.array(gt_semseg)
        # sample['gt_semseg'] = Image.fromarray(gt_semseg_array)
        sample['gt_semseg'] = gt_semseg

        # 创建彩色分割图
        # if self.palette is not None:
        #     sample['image_semseg'] = Image.fromarray(
        #         self.prepare_pm(gt_semseg_array).astype(np.uint8)
        #     )
        # else:
        #     # 如果没有调色板，使用原始标注
        #     sample['image_semseg'] = gt_semseg
        sample['image_semseg'] = Image.fromarray(self.prepare_pm(gt_semseg_array).astype(np.uint8))

        sample['text'] = None

        # 创建掩码（有效像素区域）
        # Trans10k中可能需要特殊处理无效区域
        sample['mask'] = np.ones_like(gt_semseg_array)
        if 255 in gt_semseg_array:  # 如果有无效区域
            sample['mask'][gt_semseg_array == 255] = 0
        sample['mask'] = Image.fromarray(sample['mask'].astype(np.uint8))

        # 元数据
        sample['meta'] = {
            'im_size': (_img.size[1], _img.size[0]),  # (height, width)
            'image_file': self.images[index],
            "image_id": osp.splitext(osp.basename(self.images[index]))[0]
        }

        # 应用变换
        if self.transform is not None:
            sample = self.transform(sample)

        # 标准化
        sample = self.post_norm(sample)

        return sample


# if __name__ == '__main__':
#     # 测试代码
#     dataset = Trans10k(
#         data_root='/path/to/Trans10K_v2',
#         split='validation',
#         args_palette=('random', 256)  # 示例参数
#     )
#
#     sample = dataset[0]
#     print(f"Image shape: {np.array(sample['image']).shape}")
#     print(f"Segmentation shape: {np.array(sample['gt_semseg']).shape}")
#     print(f"Meta data: {sample['meta']}")