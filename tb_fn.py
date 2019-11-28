#!/usr/bin/env python
__author__ = "Mohamed Adnen Abdessaied"
__maintainer__ = "Mohamed Adnen Abdessaied"
__email__ = "adnenabdessayed@gmail.com"
__status__ = "Implementation"


import cv2
import torch


def decorate_tb_image(image_paths: list, gt_texts: list, pred_texts: list):
    """
    This function randomly takes an image and displays the gt label and the predicted label on top
    of it.
    :param image_paths: The paths to the images.
    :param gt_texts: The ground truth texts.
    :param pred_texts: The predicted texts.
    :return: (numpy.ndarray) Decorated images.
    """
    decorated_images = []
    for (image_path, gt_text, pred_text) in zip (image_paths, gt_texts, pred_texts):
        img = cv2.imread(image_path)
        img = cv2.resize(img, (300, 200))
        cv2.putText(img, "gt: " + gt_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(img, "pred: " + pred_text, (10, 170), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        img = img / 255.0
        decorated_images.append(torch.from_numpy(img))

    return torch.stack(decorated_images, dim=0)
