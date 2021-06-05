import os
import sys
import re
import six
import time
import math
import random

from natsort import natsorted
import PIL
import lmdb
import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset, Subset
from torch._utils import _accumulate
import torchvision.transforms as transforms


class Batch_Balanced_Dataset(object):

    def __init__(self, opt, dataset_root, select_data, batch_ratio, log, learn_type=None):
        """
        Modulate the data ratio in the batch.
        For example, when select_data is "MJ-ST" and batch_ratio is "0.5-0.5",
        the 50% of the batch is filled with MJ and the other 50% of the batch is filled with ST.
        """
        self.opt = opt
        dashed_line = '-' * 80
        print(dashed_line)
        log.write(dashed_line + '\n')
        print(f'dataset_root: {dataset_root}\nselect_data: {select_data}\nbatch_ratio: {batch_ratio}')
        log.write(f'dataset_root: {dataset_root}\nselect_data: {select_data}\nbatch_ratio: {batch_ratio}\n')
        assert len(select_data) == len(batch_ratio)

        if learn_type == 'semi':
            _AlignCollate = AlignCollate_SemiSL(self.opt)
            data_type = 'unlabel'
        elif learn_type == 'self':
            _AlignCollate = AlignCollate_SelfSL(self.opt)
            data_type = 'unlabel'
        else:
            _AlignCollate = AlignCollate(self.opt)
            data_type = 'label'

        self.data_loader_list = []
        self.dataloader_iter_list = []
        batch_size_list = []
        Total_batch_size = 0
        for selected_d, batch_ratio_d in zip(select_data, batch_ratio):
            _batch_size = max(round(self.opt.batch_size * float(batch_ratio_d)), 1)
            print(dashed_line)
            log.write(dashed_line + '\n')
            _dataset, _dataset_log = hierarchical_dataset(root=dataset_root, opt=self.opt,
                                                          select_data=[selected_d], data_type=data_type)
            total_number_dataset = len(_dataset)
            log.write(_dataset_log)

            if data_type == 'label':
                """
                The total number of data can be modified with opt.total_data_usage_ratio.
                ex) opt.total_data_usage_ratio = 1 indicates 100% usage, and 0.2 indicates 20% usage.
                """
                number_dataset = int(total_number_dataset * float(self.opt.total_data_usage_ratio))
                dataset_split = [number_dataset, total_number_dataset - number_dataset]
                indices = range(total_number_dataset)
                _dataset, _ = [Subset(_dataset, indices[offset - length:offset])
                               for offset, length in zip(_accumulate(dataset_split), dataset_split)]
                selected_d_log = f'num total samples of {selected_d}: {total_number_dataset} x {self.opt.total_data_usage_ratio} (total_data_usage_ratio) = {len(_dataset)}\n'
            else:
                # as a default, we use always 100% of unlabeled dataset.
                selected_d_log = f'num total samples of {selected_d}: {len(_dataset)}\n'
            selected_d_log += f'num samples of {selected_d} per batch: {self.opt.batch_size} x {float(batch_ratio_d)} (batch_ratio) = {_batch_size}'
            print(selected_d_log)
            log.write(selected_d_log + '\n')
            batch_size_list.append(str(_batch_size))
            Total_batch_size += _batch_size

            # for faster training, we multiply small datasets itself.
            if len(_dataset) < 50000:
                multiple_times = int(50000 / len(_dataset))
                dataset_self_multiple = [_dataset] * multiple_times
                _dataset = ConcatDataset(dataset_self_multiple)

            _data_loader = torch.utils.data.DataLoader(
                _dataset, batch_size=_batch_size,
                shuffle=True,
                num_workers=int(self.opt.workers),
                collate_fn=_AlignCollate, pin_memory=False, drop_last=False)

            self.data_loader_list.append(_data_loader)
            self.dataloader_iter_list.append(iter(_data_loader))

        Total_batch_size_log = f'{dashed_line}\n'
        batch_size_sum = '+'.join(batch_size_list)
        Total_batch_size_log += f'Total_batch_size: {batch_size_sum} = {Total_batch_size}\n'
        Total_batch_size_log += f'{dashed_line}'
        if data_type == 'label':
            self.opt.Total_batch_size = Total_batch_size
        elif data_type == 'unlabel':
            self.opt.Total_unlabel_batch_size = Total_batch_size

        print(Total_batch_size_log)
        log.write(Total_batch_size_log + '\n')

    def get_batch(self):
        balanced_batch_images = []
        balanced_batch_labels = []

        for i, data_loader_iter in enumerate(self.dataloader_iter_list):
            try:
                image, label = data_loader_iter.next()
                balanced_batch_images.append(image)
                balanced_batch_labels += label
            except StopIteration:
                self.dataloader_iter_list[i] = iter(self.data_loader_list[i])
                image, label = self.dataloader_iter_list[i].next()
                balanced_batch_images.append(image)
                balanced_batch_labels += label
            except ValueError:
                pass

        balanced_batch_images = torch.cat(balanced_batch_images, 0)

        return balanced_batch_images, balanced_batch_labels

    def get_batch_ema(self):
        balanced_batch_images = []
        balanced_batch_images_ema = []
        balanced_batch_texts = []

        for i, data_loader_iter in enumerate(self.dataloader_iter_list):
            try:
                image, image_ema, text = data_loader_iter.next()
                balanced_batch_images.append(image)
                balanced_batch_images_ema.append(image_ema)
                balanced_batch_texts += text
            except StopIteration:
                self.dataloader_iter_list[i] = iter(self.data_loader_list[i])
                image, image_ema, text = self.dataloader_iter_list[i].next()
                balanced_batch_images.append(image)
                balanced_batch_images_ema.append(image_ema)
                balanced_batch_texts += text
            except ValueError:
                pass

        balanced_batch_images = torch.cat(balanced_batch_images, 0)
        balanced_batch_images_ema = torch.cat(balanced_batch_images_ema, 0)

        return balanced_batch_images, balanced_batch_images_ema, balanced_batch_texts

    def get_batch_two_images(self):
        """ two images 
        ex) For MoCo, q and k
        ex) For Mean Teacher, image with aug1 and image with aug2
        """
        balanced_batch_img1 = []
        balanced_batch_img2 = []

        for i, data_loader_iter in enumerate(self.dataloader_iter_list):
            try:
                image_aug1, image_aug2 = data_loader_iter.next()
                balanced_batch_img1.append(image_aug1)
                balanced_batch_img2.append(image_aug2)
            except StopIteration:
                self.dataloader_iter_list[i] = iter(self.data_loader_list[i])
                image_aug1, image_aug2 = self.dataloader_iter_list[i].next()
                balanced_batch_img1.append(image_aug1)
                balanced_batch_img2.append(image_aug2)
            except ValueError:
                pass

        balanced_batch_img1 = torch.cat(balanced_batch_img1, 0)
        balanced_batch_img2 = torch.cat(balanced_batch_img2, 0)

        return balanced_batch_img1, balanced_batch_img2


def hierarchical_dataset(root, opt, select_data='/', data_type='label', mode='train'):
    """ select_data='/' contains all sub-directory of root directory """
    dataset_list = []
    dataset_log = f'dataset_root:    {root}\t dataset: {select_data[0]}'
    print(dataset_log)
    dataset_log += '\n'
    for dirpath, dirnames, filenames in os.walk(root + '/'):
        if not dirnames:
            select_flag = False
            for selected_d in select_data:
                if selected_d in dirpath:
                    select_flag = True
                    break

            if select_flag:
                if data_type == 'label':
                    dataset = LmdbDataset(dirpath, opt, mode=mode)
                else:
                    dataset = LmdbDataset_unlabel(dirpath, opt)
                sub_dataset_log = f'sub-directory:\t/{os.path.relpath(dirpath, root)}\t num samples: {len(dataset)}'
                print(sub_dataset_log)
                dataset_log += f'{sub_dataset_log}\n'
                dataset_list.append(dataset)

    concatenated_dataset = ConcatDataset(dataset_list)

    return concatenated_dataset, dataset_log


class LmdbDataset(Dataset):

    def __init__(self, root, opt, mode='train'):

        self.root = root
        self.opt = opt
        self.mode = mode
        self.env = lmdb.open(root, max_readers=32, readonly=True, lock=False, readahead=False, meminit=False)
        if not self.env:
            print('cannot open lmdb from %s' % (root))
            sys.exit(0)

        with self.env.begin(write=False) as txn:
            self.nSamples = int(txn.get('num-samples'.encode()))
            self.filtered_index_list = []
            for index in range(self.nSamples):
                index += 1  # lmdb starts with 1
                label_key = 'label-%09d'.encode() % index
                label = txn.get(label_key).decode('utf-8')

                # length filtering
                length_of_label = len(label)
                if length_of_label > opt.batch_max_length:
                    continue

                self.filtered_index_list.append(index)

            self.nSamples = len(self.filtered_index_list)

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        index = self.filtered_index_list[index]

        with self.env.begin(write=False) as txn:
            label_key = 'label-%09d'.encode() % index
            label = txn.get(label_key).decode('utf-8')
            img_key = 'image-%09d'.encode() % index
            imgbuf = txn.get(img_key)
            buf = six.BytesIO()
            buf.write(imgbuf)
            buf.seek(0)

            try:
                img = PIL.Image.open(buf).convert('RGB')

            except IOError:
                print(f'Corrupted image for {index}')
                # make dummy image and dummy label for corrupted image.
                img = PIL.Image.new('RGB', (self.opt.imgW, self.opt.imgH))
                label = '[dummy_label]'

        return (img, label)


class LmdbDataset_unlabel(Dataset):

    def __init__(self, root, opt):

        self.root = root
        self.opt = opt
        self.env = lmdb.open(root, max_readers=32, readonly=True, lock=False, readahead=False, meminit=False)
        if not self.env:
            print('cannot open lmdb from %s' % (root))
            sys.exit(0)

        with self.env.begin(write=False) as txn:
            self.nSamples = int(txn.get('num-samples'.encode()))
            self.index_list = [index + 1 for index in range(self.nSamples)]

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        index = self.index_list[index]

        with self.env.begin(write=False) as txn:
            img_key = 'image-%09d'.encode() % index
            imgbuf = txn.get(img_key)
            buf = six.BytesIO()
            buf.write(imgbuf)
            buf.seek(0)

            try:
                img = PIL.Image.open(buf).convert('RGB')

            except IOError:
                print(f'Corrupted image for {img_key}')
                # make dummy image for corrupted image.
                img = PIL.Image.new('RGB', (opt.imgW, opt.imgH))

        return img


class RawDataset(Dataset):

    def __init__(self, root, opt):
        self.opt = opt
        self.image_path_list = []
        for dirpath, dirnames, filenames in os.walk(root):
            for name in filenames:
                _, ext = os.path.splitext(name)
                ext = ext.lower()
                if ext == '.jpg' or ext == '.jpeg' or ext == '.png':
                    self.image_path_list.append(os.path.join(dirpath, name))

        self.image_path_list = natsorted(self.image_path_list)
        self.nSamples = len(self.image_path_list)

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):

        try:
            img = PIL.Image.open(self.image_path_list[index]).convert('RGB')

        except IOError:
            print(f'Corrupted image for {index}')
            # make dummy image and dummy label for corrupted image.
            img = PIL.Image.new('RGB', (self.opt.imgW, self.opt.imgH))

        return (img, self.image_path_list[index])


class AlignCollate(object):

    def __init__(self, opt, mode='train'):
        self.opt = opt
        self.mode = mode

        if opt.Aug == 'None' or mode != 'train':
            self.transform = ResizeNormalize((opt.imgW, opt.imgH))
        else:
            self.transform = Text_augment(opt)

    def __call__(self, batch):
        images, labels = zip(*batch)

        if 'MeanT' in self.opt.semi and self.mode == 'train':
            image_tensors = [self.transform(image) for image in images]
            image_tensors = torch.cat([t.unsqueeze(0) for t in image_tensors], 0)

            image_tensors_ema = [self.transform(image) for image in images]
            image_tensors_ema = torch.cat([t.unsqueeze(0) for t in image_tensors_ema], 0)

            return image_tensors, image_tensors_ema, labels

        else:
            image_tensors = [self.transform(image) for image in images]
            image_tensors = torch.cat([t.unsqueeze(0) for t in image_tensors], 0)

            return image_tensors, labels


class AlignCollate_SemiSL(object):

    def __init__(self, opt):
        self.opt = opt
        self.transform = Text_augment(opt)

    def __call__(self, batch):

        if 'MeanT' in self.opt.semi:
            student_list = []
            teacher_list = []
            for image in batch:
                student_data = self.transform(image)
                teacher_data = self.transform(image)
                student_list.append(student_data)
                teacher_list.append(teacher_data)

            student_tensors = torch.cat([t.unsqueeze(0) for t in student_list], 0)
            teacher_tensors = torch.cat([t.unsqueeze(0) for t in teacher_list], 0)

            return student_tensors, teacher_tensors

        else:
            image_tensors = [self.transform(image) for image in batch]
            image_tensors = torch.cat([t.unsqueeze(0) for t in image_tensors], 0)

            return image_tensors, image_tensors


class AlignCollate_SelfSL(object):

    def __init__(self, opt):
        self.opt = opt
        self.transform = ResizeNormalize((opt.imgW, opt.imgH))
        if 'MoCo' in opt.self:
            self.MoCo_augment = MoCo_augment(opt)

    def __call__(self, batch):

        if 'RotNet' in self.opt.self:
            rotate_images = []
            rotate_labels = []
            for image in batch:
                image_rotated_0 = image
                image_rotated_90 = image.transpose(PIL.Image.ROTATE_90)
                image_rotated_180 = image.transpose(PIL.Image.ROTATE_180)
                image_rotated_270 = image.transpose(PIL.Image.ROTATE_270)
                rotate_images.extend([image_rotated_0, image_rotated_90, image_rotated_180, image_rotated_270])
                rotate_labels.extend([0, 1, 2, 3])  # corresponds to 0, 90, 180, 270 degrees, respectively.

            image_tensors = [self.transform(image) for image in rotate_images]
            image_tensors = torch.cat([t.unsqueeze(0) for t in image_tensors], 0)

            return image_tensors, rotate_labels

        elif 'MoCo' in self.opt.self:
            q_list = []
            k_list = []
            for image in batch:
                q, k = self.MoCo_augment(image)
                q_list.append(q)
                k_list.append(k)

            q_tensors = torch.cat([t.unsqueeze(0) for t in q_list], 0)
            k_tensors = torch.cat([t.unsqueeze(0) for t in k_list], 0)

            return q_tensors, k_tensors


# from https://github.com/facebookresearch/moco
class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, image):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        image = image.filter(PIL.ImageFilter.GaussianBlur(radius=sigma))
        return image


class RandomCrop(object):
    """ RandomCrop, 
    RandomResizedCrop of PyTorch 1.6 and torchvision 0.7.0 work weird with scale 0.90-1.0.
    i.e. you can not always make 90%~100% cropped image scale 0.90-1.0, you will get central cropped image instead.
    so we made RandomCrop (keeping aspect ratio version) then use Resize.
    """

    def __init__(self, scale=[1, 1]):
        self.scale = scale

    def __call__(self, image):
        width, height = image.size
        crop_ratio = random.uniform(self.scale[0], self.scale[1])
        crop_width = int(width * crop_ratio)
        crop_height = int(height * crop_ratio)

        x_start = random.randint(0, width - crop_width)
        y_start = random.randint(0, height - crop_height)
        image_crop = image.crop((x_start, y_start, x_start + crop_width, y_start + crop_height))
        return image_crop


class ResizeNormalize(object):

    def __init__(self, size, interpolation=PIL.Image.BICUBIC):
        # CAUTION: it should be (width, height). different from size of transforms.Resize (height, width)
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, image):
        image = image.resize(self.size, self.interpolation)
        image = self.toTensor(image)
        image.sub_(0.5).div_(0.5)
        return image


class Text_augment(object):
    """Augmentation for Text recognition"""

    def __init__(self, opt):
        self.opt = opt
        augmentation = []
        aug_list = self.opt.Aug.split('-')
        for aug in aug_list:
            if aug.startswith('Blur'):
                maximum = float(aug.strip('Blur'))
                augmentation.append(transforms.RandomApply([GaussianBlur([.1, maximum])], p=0.5))

            if aug.startswith('Crop'):
                crop_scale = float(aug.strip('Crop')) / 100
                augmentation.append(RandomCrop(scale=(crop_scale, 1.0)))

            if aug.startswith('Rot'):
                degree = int(aug.strip('Rot'))
                augmentation.append(transforms.RandomRotation(degree, resample=PIL.Image.BICUBIC, expand=True))

        augmentation.append(transforms.Resize((self.opt.imgH, self.opt.imgW), interpolation=PIL.Image.BICUBIC))
        augmentation.append(transforms.ToTensor())
        self.Augment = transforms.Compose(augmentation)
        print('Use Text_augment', augmentation)

    def __call__(self, image):
        image = self.Augment(image)
        image.sub_(0.5).div_(0.5)

        return image


class MoCo_augment(object):
    """Take two random crops of one image as the query and key."""

    def __init__(self, opt):
        self.opt = opt

        # MoCo v1's aug: the same as InstDisc https://arxiv.org/abs/1805.01978
        augmentation = [
            transforms.RandomResizedCrop((opt.imgH, opt.imgW),
                                         scale=(0.2, 1.), interpolation=PIL.Image.BICUBIC),
            transforms.RandomGrayscale(p=0.2),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ]

        self.Augment = transforms.Compose(augmentation)
        print('Use MoCo_augment', augmentation)

    def __call__(self, x):
        q = self.Augment(x)
        k = self.Augment(x)
        q.sub_(0.5).div_(0.5)
        k.sub_(0.5).div_(0.5)

        return [q, k]
