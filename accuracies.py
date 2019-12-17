#!/usr/bin/env python
__author__ = "Mohamed Adnen Abdessaied"
__maintainer__ = "Mohamed Adnen Abdessaied"
__email__ = "adnenabdessayed@gmail.com"
__status__ = "Implementation"


import torch
import functools
import textdistance
import numpy as np
import re

def from_probabilities_to_letters(ocr_output, alphabet):
    """
    Computes a "greedy" decoding of the ocr outputs. The ocr outputs can be thought of as probabilities over the
    different letters of the alphabet. This decoding takes the letter with the highest probability at each step.
    :param ocr_output: The output of the ocr net.
                       It has the general shape (sequence_length, batch_size, alphabet_size)
    :param alphabet: The alphabet used for text recognition.
    :param blank_idx: The index of the blank character in the alphabet.
    :return: A list of batch_size strings resulting from the decoding of the ocr_output.
    """
    # Detach the ocr output to avoid memory leaks
    ocr_output = ocr_output.detach()

    # Reshape the ocr output --> Shape: (batch_size, 100, 49)
    ocr_output = ocr_output.permute(1, 0, 2)

    greedy_decodings = ocr_output.argmax(dim=2).to(torch.device("cpu"))
    greedy_decodings = greedy_decodings.tolist()

    def remove_duplicates(sequence):
        idx_to_delete = []
        for i in range(len(sequence) - 1):
            if sequence[i + 1] == sequence[i]:
                idx_to_delete.append(i + 1)
        sequence = [sequence[i] for i in range(len(sequence)) if i not in idx_to_delete]
        return sequence

    # Remove duplicates and blanks
    greedy_decodings = list(map(functools.partial(remove_duplicates), greedy_decodings))

    # Get the decoding in letters. We discard idx = 0 since it is reserved for the blank character.
    greedy_decodings = list(map(lambda x: "".join([alphabet[idx - 1] for idx in x if idx > 0]),
                                greedy_decodings))
    greedy_decodings = list(map(lambda x: x.lower(), greedy_decodings))
    greedy_decodings = list(map(lambda x: x.replace(" ", ""), greedy_decodings))

    return greedy_decodings


def compute_accuracies(gts: list, greedy_decodings: list, all_gts: list, mode: str):
    """
    Computes the per batch accuracies.
    :param gts: List of the ground-truth labels.
    :param greedy_decodings: List of the predicted labels.
    :param all_gts: The list of all ground truth labels.
    :param mode: The mode of the experiment --> Training or validation.
    :return: A dict containing the accuracies and a list of closest gts.
    """
    assert len(gts) == len(greedy_decodings)

    gts = list(map(lambda x: x.lower().replace(" ", ""), gts))
    all_gts = list(map(lambda x: x.lower().replace(" ", ""), all_gts))

    corrects = 0
    closest_gts = []

    # We try here to recover the closest label to the predicted one from the set of all labels.
    corrects_after_mapping = 0

    # compute the accuracies related to the box heights

    for gt, predected_label in zip(gts, greedy_decodings):
        if gt == predected_label:
            corrects += 1
            corrects_after_mapping += 1
            closest_gts.append(predected_label)

        else:
            similarities = [textdistance.levenshtein.normalized_similarity(_gt, predected_label) for _gt in all_gts]
            similarities = np.array(similarities)
            closest_gt = all_gts[similarities.argmax()]
            closest_gts.append(closest_gt)
            if closest_gt == gt:
                corrects_after_mapping += 1

    accuracies = {"exact_acc_" + mode: corrects / len(gts),
                  "exact_acc_after_mapping_" + mode: corrects_after_mapping / len(gts)}

    return accuracies, closest_gts


def compute_accuracies_per_box_height(gts: list,
                                      greedy_decodings: list,
                                      all_gts: list,
                                      box_heights: list,
                                      box_heights_intervals: list,
                                      mode: str):
    """
    This function computes the accuracies based on the heights of the crops.
    :param gts: List of the ground-truth labels.
    :param greedy_decodings: List of the predicted labels.
    :param all_gts: The list of all ground truth labels.
    :param box_heights: List of the crops heights.
    :param box_heights_intervals: List of height intervals.
    :param mode: The mode of the experiment --> Training or validation.
    :return: A dict containing the accuracies per height interval.
    """
    interval_indicies = []
    sub_indices = []
    for j in range(len(box_heights_intervals) - 1):
        sub_indices.append("{}-{}".format(box_heights_intervals[j], box_heights_intervals[j + 1]))
        idx = []
        for i in range(len(box_heights)):
            if box_heights_intervals[j] <= box_heights[i] < box_heights_intervals[j + 1]:
                idx.append(i)

        interval_indicies.append(idx)

    gts_all_intervals = [[gts[i] for i in idx] for idx in interval_indicies if len(idx) > 0]
    greedy_decodings_all_intervals = [[greedy_decodings[i] for i in idx] for idx in interval_indicies if len(idx) > 0]
    accuracies_all_intervals = {}
    for gts_per_interval, greedy_decodings_per_interval, sub_idx in zip(
            gts_all_intervals, greedy_decodings_all_intervals, sub_indices):
        accuracies, _ = compute_accuracies(gts_per_interval,
                                           greedy_decodings_per_interval,
                                           all_gts,
                                           mode + "_" + sub_idx)

        accuracies_all_intervals[mode + "_accuracies_" + sub_idx] = accuracies

    return accuracies_all_intervals


def compute_number_detection_accuracies(gts: list,
                                        greedy_decodings: list,
                                        all_gts: list,
                                        mode: str):
    """
    This function computes the accuracy of detecting numerical values in the labels.
    :param gts: List of the ground-truth labels.
    :param greedy_decodings: List of the predicted labels.
    :param all_gts: The list of all ground truth labels.
    :param mode: The mode of the experiment --> Training or validation.
    :return: A dict containing the accuracies.
    """
    assert len(gts) == len(greedy_decodings)

    gts = list(map(lambda x: x.lower().replace(" ", ""), gts))
    all_gts = list(map(lambda x: x.lower().replace(" ", ""), all_gts))

    corrects = 0
    corrects_after_mapping = 0

    # <num_numbers> stores how many numbers we have in all labels per batch.
    num_numbers = 0
    for gt, predected_label in zip(gts, greedy_decodings):

        similarities = [textdistance.levenshtein.normalized_similarity(_gt, predected_label) for _gt in all_gts]
        similarities = np.array(similarities)
        closest_gt = all_gts[similarities.argmax()]

        # https://stackoverflow.com/questions/4289331/how-to-extract-numbers-from-a-string-in-python
        # -->
        numbers_in_gt = re.findall(r"\d+", gt)
        numbers_in_pred = re.findall(r"\d+", predected_label)
        numbers_in_closest = re.findall(r"\d+", closest_gt)
        # <--
        if len(numbers_in_gt) == 0:
            continue
        num_numbers += len(numbers_in_gt)

        if len(numbers_in_pred) > len(numbers_in_gt):
            numbers_in_pred = numbers_in_pred[:len(numbers_in_gt)]
        elif len(numbers_in_pred) < len(numbers_in_gt):
            numbers_in_pred.extend(["*" for _ in range(len(numbers_in_gt) - len(numbers_in_pred))])

        if len(numbers_in_closest) > len(numbers_in_gt):
            numbers_in_closest = numbers_in_closest[:len(numbers_in_gt)]
        elif len(numbers_in_closest) < len(numbers_in_gt):
            numbers_in_closest.extend(["*" for _ in range(len(numbers_in_gt) - len(numbers_in_closest))])

        for num_gt, num_pred, num_closest in zip(numbers_in_gt, numbers_in_pred, numbers_in_closest):
            if num_gt == num_pred:
                corrects += 1
            if num_gt == num_closest:
                corrects_after_mapping += 1

    if num_numbers > 0:
        accuracies = {"exact_num_det_acc_" + mode: corrects / num_numbers,
                      "exact_num_det_acc_after_mapping_" + mode: corrects_after_mapping / num_numbers}
    else:
        accuracies = None

    return accuracies
