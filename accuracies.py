#!/usr/bin/env python
__author__ = "Mohamed Adnen Abdessaied"
__maintainer__ = "Mohamed Adnen Abdessaied"
__email__ = "adnenabdessayed@gmail.com"
__status__ = "Implementation"


import torch
import functools
import textdistance


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


def compute_accuracies(gts: list, greedy_decodings: list, mode: str):
    """
    Computes the per batch accuracies.
    :param gts: List of the ground-truth labels
    :param greedy_decodings: List of the predicted labels
    :param mode: The mode of the experiment --> Training or validation
    :return: A dict containing  exact, hamming and the levenshtein accuracies.
    """
    assert len(gts) == len(greedy_decodings)

    gts = list(map(lambda x: x.lower(), gts))
    gts = list(map(lambda x: x.replace(" ", ""), gts))

    hamming_corrects = 0
    levenshtein_corrects = 0
    corrects = 0
    for gt, predected_label in zip(gts, greedy_decodings):
        # print("target: {} --- pred: \"{}\"".format(gt, predected_label))
        if gt == predected_label:
            corrects += 1
        hamming_score = textdistance.hamming.normalized_similarity(gt, predected_label)
        levenshtein_score = textdistance.levenshtein.normalized_similarity(gt, predected_label)

        if hamming_score >= 0.7:
            hamming_corrects += 1

        if levenshtein_score >= 0.7:
            levenshtein_corrects += 1

    accuracies = {"exact_acc_" + mode: corrects / len(gts),
                  "hamming_acc_" + mode: hamming_corrects / len(gts),
                  "levenshtein_acc_" + mode: levenshtein_corrects / len(gts)}

    return accuracies
