from glob import glob
from os.path import dirname, join, basename, isfile
import sys
sys.path.append('./')
import csv
import torch
import cv2
from medpy.io import load
import numpy as np
from PIL import Image
from torch import nn
import torch.nn.functional as F
import random
import torchio as tio
from torchio import AFFINE, DATA
import torchio
from torchio import ScalarImage, LabelMap, Subject, SubjectsDataset, Queue
from torchio.data import UniformSampler
from torchio.transforms import (
    RandomFlip,
    RandomAffine,
    RandomElasticDeformation,
    RandomNoise,
    RandomMotion,
    RandomBiasField,
    RescaleIntensity,
    Resample,
    ToCanonical,
    ZNormalization,
    CropOrPad,
    HistogramStandardization,
    OneOf,
    Compose,
)
from pathlib import Path

from for_final_hparams import hparams as hp
 

class MedData_train(torch.utils.data.Dataset):
    def __init__(self, images_dir, labels_dir):

        if hp.mode == '3d':
            patch_size = hp.patch_size
        elif hp.mode == '2d':
            patch_size = hp.patch_size
        else:
            raise Exception('no such kind of mode!')

        queue_length = 5
        samples_per_volume = 5


        self.subjects = []

        if (hp.in_class == 1) and (hp.out_class == 1) :

            images_dir = Path(images_dir)
            self.image_paths = sorted(images_dir.glob(hp.fold_arch))
            labels_dir = Path(labels_dir)
            self.label_paths = sorted(labels_dir.glob(hp.label_arch))

            for (image_path, label_path) in zip(self.image_paths, self.label_paths):
                subject = tio.Subject(
                    source=tio.ScalarImage(image_path),
                    label=tio.LabelMap(label_path),
                )
                self.subjects.append(subject)
        else:
            images_dir = Path(images_dir)
            self.image_paths = sorted(images_dir.glob(hp.fold_arch))

            artery_labels_dir = Path(labels_dir+'/o_crop') # Frontal lobe right WM
            self.artery_label_paths = sorted(artery_labels_dir.glob(hp.label_arch))

            lung_labels_dir = Path(labels_dir+'/bean') # Frontal lobe right WM
            self.lung_label_paths = sorted(lung_labels_dir.glob(hp.label_arch))

            trachea_labels_dir = Path(labels_dir+'/maize') # Corpus Callosum
            self.trachea_label_paths = sorted(trachea_labels_dir.glob(hp.label_arch))

            vein_labels_dir = Path(labels_dir+'/weed') # CSF
            self.vein_label_paths = sorted(vein_labels_dir.glob(hp.label_arch))


            for (image_path,artery_label_path,lung_label_path,trachea_label_path,vein_label_path) in zip(self.image_paths, self.artery_label_paths, self.lung_label_paths,self.trachea_label_paths,self.vein_label_paths):
                #img_input = tio.ScalarImage(image_path)
                #print(image_path)
                
                #img_input = cv2.imread(str(image_path))
                #print(type(img_input))
                #channels = get_14_channels(img_input.data)
                subject = tio.Subject(
                    source=tio.ScalarImage(image_path),# 獲得圖片
                    atery=tio.LabelMap(artery_label_path),# 獲得標籤
                    lung=tio.LabelMap(lung_label_path),
                    trachea=tio.LabelMap(trachea_label_path),
                    vein=tio.LabelMap(vein_label_path),
                )

                #　常见的例子是在对象中封装图像本身，对应的mask，以及图像的信息如患者姓名，获取的医院等
                self.subjects.append(subject)


        self.transforms = self.transform()

        self.training_set = tio.SubjectsDataset(self.subjects, transform=self.transforms)


        self.queue_dataset = Queue( #　数据进入GPU之前先处理patch
            self.training_set,
            queue_length,
            samples_per_volume,
            UniformSampler(patch_size),
        )




    def transform(self):

        if hp.mode == '3d':
            if hp.aug:
                training_transform = Compose([
                # ToCanonical(),
                CropOrPad((hp.crop_or_pad_size), padding_mode='reflect'),
                # RandomMotion(),
                #RandomBiasField(),
                ZNormalization(),
                RandomNoise(),
                RandomFlip(axes=(0,)),
                OneOf({
                    RandomAffine(): 0.8,
                    RandomElasticDeformation(): 0.2,
                }),])
            else:
                # ~ print('hp.crop_or_pad_size = ',hp.crop_or_pad_size,'\n')
                training_transform = Compose([
                CropOrPad((hp.crop_or_pad_size), padding_mode='reflect'),
                ZNormalization(),
                ])
                # ~ print(zxc)##
        elif hp.mode == '2d':
            if hp.aug:
                training_transform = Compose([
                CropOrPad((hp.crop_or_pad_size), padding_mode='reflect'),
                # RandomMotion(),
                # RandomBiasField(),
                # ZNormalization(),# 將強度值重新縮放到特定範圍。
                RandomNoise(), # 加入雜訊
                RandomFlip(axes=(0,)),
                ])
            else:
                training_transform = Compose([
                CropOrPad((hp.crop_or_pad_size), padding_mode='reflect'),
                ZNormalization(),
                ])

        else:
            raise Exception('no such kind of mode!')


        return training_transform




class MedData_test(torch.utils.data.Dataset):
    def __init__(self, images_dir, labels_dir):

        self.subjects = []

        if (hp.in_class == 1) and (hp.out_class == 1) :

            images_dir = Path(images_dir)
            self.image_paths = sorted(images_dir.glob(hp.fold_arch))
            labels_dir = Path(labels_dir)
            self.label_paths = sorted(labels_dir.glob(hp.fold_arch))

            for (image_path, label_path) in zip(self.image_paths, self.label_paths):
                subject = tio.Subject(
                    source=tio.ScalarImage(image_path),
                    label=tio.LabelMap(label_path),
                )
                self.subjects.append(subject)
        else:
            images_dir = Path(images_dir)
            self.image_paths = sorted(images_dir.glob(hp.fold_arch))
            for (image_path) in zip(self.image_paths):
                name = str(image_path[0]).split('/')[-1].replace('.jpg','')
                subject = tio.Subject(
                    source=tio.ScalarImage(image_path),
                    img_name=name
                )
                self.subjects.append(subject)


        
        
        self.test_transform = self.test_transform()
        print('self.test_transform = ',self.test_transform)

        self.training_set = tio.SubjectsDataset(self.subjects, transform=None)

    def test_transform(self):
        training_transform = Compose([
        # ~ CropOrPad((hp.crop_or_pad_size), padding_mode='reflect'),
        CropOrPad((64,64,64), padding_mode='reflect'),
        ZNormalization(),
        ])
        print('hp.crop_or_pad_size = ',hp.crop_or_pad_size)##
        
        return training_transform



