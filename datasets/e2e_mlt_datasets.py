#!/usr/bin/env python
__author__ = "Mohamed Adnen Abdessaied"
__maintainer__ = "Mohamed Adnen Abdessaied"
__email__ = "adnenabdessayed@gmail.com"
__status__ = "Implementation"


import os
import cv2
import numpy as np
import logging
import copy
import json

from tqdm import tqdm

import torch
from torch.utils.data.dataset import Dataset
logging.basicConfig(level=logging.INFO)


class E2E_MLT_Dataset(Dataset):

    def __init__(self, path_to_image_folder: str, path_to_label_folder: str):
        """
        E2E_MLT_Dataset constructor.

        :param path_to_image_folder: Path to the folder containing the images.
        :param path_to_label_folder: Path to the folder containing the labels.
        :param alphabet: The union of all characters of the different language alphabets.
        """
        super(E2E_MLT_Dataset, self).__init__()
        assert os.path.isdir(path_to_image_folder)
        assert os.path.isdir(path_to_label_folder)
        self.path_to_image_folder = path_to_image_folder
        self.path_to_label_folder = path_to_label_folder
        self.classes, self.class_to_idx = self.find_classes()
        self.num_classes = len(self.classes)
        self.image_paths, self.label_paths, self.gt_indices = self.get_data()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        assert idx in range(self.__len__())
        path_img, path_label, gt = self.image_paths[idx], self.label_paths[idx], self.gt_indices[idx]
        img = cv2.imread(path_img)

        # From BGR to RGB as CV2 reads color images in BGR format
        B = copy.deepcopy(img[:, :, 0])
        R = copy.deepcopy(img[:, :, -1])
        img[:, :, 0], img[:, :, -1] = R, B
        img = torch.from_numpy(np.rollaxis(img, -1, 0)).float()

        with open(path_label, "r") as f:
            label_data = json.load(f)
        box_height = label_data["box_height"]
        return img, path_img, self.classes[gt], box_height

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


class E2E_MLT_Dataset_Synth(Dataset):

    def __init__(self, path_to_annotation: str, path_to_lexicon: str, output_filtered_data: str = None):
        """
        E2E_MLT_Dataset_Synth constructor. This dataset uses the synthetically-generated data by the
        "Visual Geometry Group" of the University of Oxford --> http://www.robots.ox.ac.uk/~vgg/data/text/#sec-synth.
        They provide annotation files for the training, validation and test data. We use this default split to pre-train
        our network.

        :param path_to_annotation: Path to the annotation .txt file.
        :param path_to_lexicon: Path to the lexicon .txt file.
        :param output_filtered_data: Path to where the filtered annotation file will be saved.
        """
        super(E2E_MLT_Dataset_Synth, self).__init__()
        assert os.path.isfile(path_to_annotation)
        assert os.path.isfile(path_to_lexicon)
        self.path_to_annotation = path_to_annotation
        self.path_to_lexicon = path_to_lexicon
        self.output_filtered_data = output_filtered_data

        self.lexicon = self.get_lexicon()
        self.image_paths, self.labels, _ = self.get_data()
        self.num_classes = len(self.image_paths)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        assert idx in range(self.__len__())
        path_img, gt = self.image_paths[idx], self.labels[idx]
        img = cv2.imread(path_img)
        img = cv2.resize(img, (100, 32))

        # From BGR to RGB as CV2 reads color images in BGR format
        B = copy.deepcopy(img[:, :, 0])
        R = copy.deepcopy(img[:, :, -1])
        img[:, :, 0], img[:, :, -1] = R, B

        img = torch.from_numpy(np.rollaxis(img, -1, 0)).float()
        return img, path_img, gt

    def get_lexicon(self):
        """
        Reads the words used to generate the synthetic data alongside their indices.
        :return: A dict of indices as keys and words as values
        """
        with open(self.path_to_lexicon, "rb") as f:
            lines = f.readlines()

        lines = list(map(lambda s: (s.decode("utf-8").rstrip()), lines))
        lexicon = dict(zip([i for i in range(len(lines))], lines))

        return lexicon

    def get_data(self):
        """
        Gets the paths of the images and their labels.
        :return: a list of image paths and a list of labels
        """
        with open(self.path_to_annotation, "rb") as f:
            lines = f.readlines()

        lines = list(map(lambda s: s.decode("utf-8").rstrip(), lines))
        image_paths = list(map(lambda s: s.split(" ")[0], lines))
        labels_idx = list(map(lambda s: int(s.split(" ")[-1]), lines))

        labels = [self.lexicon[idx] for idx in labels_idx]

        return image_paths, labels, labels_idx

    def create_filtered_data(self):
        """
        Filters bad data out to avoid unexpected interruptions and exceptions whilst training and writes the results
        into a filtered annotation .txt file.

        :return: None
        """
        image_paths, _, labels_idx = self.get_data()
        pbar = tqdm(zip(image_paths, labels_idx))
        lines = []
        for (path, label_idx) in pbar:
            try:
                img = cv2.imread(path)
                img = cv2.resize(img, (100, 32))
                line = path + " " + str(label_idx) + "\n"
                lines.append(line.encode("utf-8"))
            except cv2.error:
                continue

        with open(self.output_filtered_data, "wb") as f:
            f.writelines(lines)


if __name__ == "__main__":
    lexicon_path = os.path.join(os.getcwd(), os.pardir, "lexicon.txt")
    for split in ["train", "val", "test"]:
        output_filtered_data = os.path.join(os.getcwd(), os.pardir, "filtered_annotation_{}.txt".format(split))
        path_to_annotation_file = os.path.join(os.getcwd(), os.pardir, "my_annotation_{}.txt".format(split))
        dataset = E2E_MLT_Dataset_Synth(path_to_annotation_file, lexicon_path, output_filtered_data)
        logging.info(50 * "-" + "Filtering {} data".format(split) + 50 * "-")
        dataset.create_filtered_data()
