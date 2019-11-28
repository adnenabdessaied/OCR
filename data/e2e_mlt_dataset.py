#!/usr/bin/env python
__author__ = "Mohamed Adnen Abdessaied"
__maintainer__ = "Mohamed Adnen Abdessaied"
__email__ = "adnenabdessayed@gmail.com"
__status__ = "Implementation"


import os
import cv2
import numpy as np

import torch
from torch.utils.data.dataset import Dataset


class E2E_MLT_Dataset(Dataset):

    def __init__(self, path_to_image_folder: str,
                 path_to_label_folder: str,
                 alphabet: str,
                 ):
        """
        E2E_MLT_Dataset constructor.

        :param path_to_image_folder: Path to the folder containing the images.
        :param path_to_label_folder: Path to the folder containing the labels.
        :param alphabet: The union of all characters of the different language alphabets.
        """
        super(E2E_MLT_Dataset, self).__init__()
        assert os.path.isdir(path_to_image_folder)
        assert os.path.isdir(path_to_label_folder)
        self.alphabet = alphabet
        self.path_to_image_folder = path_to_image_folder
        self.path_to_label_folder = path_to_label_folder

        self.classes, self.class_to_idx = self.find_classes()
        self.num_classes = len(self.classes)
        self.image_paths, self.label_paths, self.gt_indices = self.get_data()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        assert idx in range(self.__len__())
        path_img, _, gt = self.image_paths[idx], self.label_paths[idx], self.gt_indices[idx]
        img = torch.from_numpy(np.rollaxis(cv2.imread(path_img), -1, 0))
        img = img / 255.0
        return img, path_img, self.classes[gt]

    def find_classes(self):
        """
        Finds the class folders in a dataset.

        Args:
            dir (string): Root directory path.

        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.

        Ensures:
            No class is a subdirectory of another.
        """

        classes = [d.name for d in os.scandir(self.path_to_image_folder) if d.is_dir()]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def get_data(self):
        """
        This function gets the paths of the images and labels.
        :return: a list of (image_path, label_path) for all the data
        """
        image_paths = []
        classes = os.listdir(self.path_to_image_folder)
        gt_indices = []
        for _class in classes:
            class_folder = os.path.join(self.path_to_image_folder, _class)
            for image_name in os.listdir(class_folder):
                image_paths.append(os.path.join(class_folder, image_name))
                gt_indices.append(self.class_to_idx[_class])

        label_paths = list(map(lambda x: x.replace("images", "labels"), image_paths))
        label_paths = list(map(lambda x: x.replace("png", "json"), label_paths))

        return image_paths, label_paths, gt_indices
