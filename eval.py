#!/usr/bin/env python
__author__ = "Mohamed Adnen Abdessaied"
__maintainer__ = "Mohamed Adnen Abdessaied"
__email__ = "adnenabdessayed@gmail.com"
__status__ = "Implementation"

import logging
import argparse
import random
from tqdm import tqdm
from time import time
from tensorboardX import SummaryWriter
from accuracies import (
    from_probabilities_to_letters,
    compute_accuracies,
    compute_accuracies_per_box_height,
    compute_number_detection_accuracies
)
from tb_fn import decorate_tb_image
import torch
from torch.nn.modules.loss import CTCLoss
from torch.utils.data.dataloader import DataLoader
from E2E_MLT.datasets.e2e_mlt_datasets import E2E_MLT_Dataset
from train import _get_device, _get_ctc_tensors, _load_net

logging.basicConfig(level=logging.INFO)


def _get_args():
    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument("-evali",
                            "--evaluation_images_path",
                            default="/lhome/mabdess/VirEnv/OCR/data/usa_data/images/val",
                            help="Path to the folder containing the validation images.")

    arg_parser.add_argument("-evall",
                            "--evaluation_labels_path",
                            default="/lhome/mabdess/VirEnv/OCR/data/usa_data/labels/val",
                            help="Path to the folder containing the validation labels.")

    arg_parser.add_argument("-b",
                            "--batch_size",
                            default=8,
                            help="Batch size.")

    arg_parser.add_argument("-bhi",
                            "--box_heights_intervals",
                            default=[8, 10, 12, 14, 16],
                            help="The box height intervals based on which we want to investigate the accuracies.")

    arg_parser.add_argument("-tb",
                            "--tensorboard",
                            default="/lhome/mabdess/VirEnv/OCR/src/E2E_MLT/summaries_ger_p_0.2/eval",
                            help="Tensorboard summaries directory.")

    arg_parser.add_argument("-chkpt",
                            "--checkpoints",
                            required=False,
                            default="/lhome/mabdess/VirEnv/OCR/src/E2E_MLT/checkpoints_ger_p_0.2/best",
                            help="Directory for check-pointing the network.")

    args = vars(arg_parser.parse_args())

    # Make sure the interval boundaries are in ascending order
    args["box_heights_intervals"].sort()

    return args


def eval(args):
    """
    This function evaluates the best network after training is completed. By best network we mean the one that had the
    minimal loss on the validation data set.
    :param args:
    """
    device = _get_device()
    net = _load_net(args["checkpoints"], device)[0]
    batch_iter_eval = 0
    val_dataset = E2E_MLT_Dataset(args["evaluation_images_path"], args["evaluation_labels_path"])
    logging.info("Data successfully loaded ...")
    summary_writer = SummaryWriter(log_dir=args["tensorboard"])

    # Construct train and validation data loaders
    logging.info("Constructing the data loaders ...")
    val_data_loader = DataLoader(val_dataset, batch_size=int(args["batch_size"] / 2), shuffle=True, num_workers=6)
    logging.info("Data loaders successfully constructed ...")
    pbar_val = tqdm(val_data_loader)
    pbar_val.set_description("Eval")
    total_t = 0
    for images, image_paths, labels, box_heights in pbar_val:
        images = images.to(device)
        ctc_targets, target_lengths = _get_ctc_tensors(labels, net.alphabet, device)

        # Strip all whitespaces because standard 'best path decoding' can't deal with whitespaces between
        # words
        labels = list(map(lambda x: x.replace(" ", ""), labels))
        labels = list(map(lambda x: x.lower(), labels))
        with torch.no_grad():
            start_t = time()
            # Shape: (100, batch_size, 49)
            ocr_outputs = net(images)
            total_t += time() - start_t

        # The ctc loss requires the input lengths as it allows for variable length inputs. In our case,
        # all the inputs have the same length.
        ocr_input_lengths = torch.full(size=(ocr_outputs.size(1),),
                                       fill_value=ocr_outputs.size(0),
                                       dtype=torch.long).to(device)

        criterion = CTCLoss(zero_infinity=True)
        # ocr_outputs are the inputs of the ctc_loss.
        ctc_loss_eval = criterion(ocr_outputs, ctc_targets, ocr_input_lengths, target_lengths)

        # compute the validation accuracies
        start_t = time()
        greedy_decodings = from_probabilities_to_letters(ocr_outputs, net.alphabet)
        total_t += time() - start_t
        accuracies_eval, closest_gts = compute_accuracies(labels,
                                                          greedy_decodings,
                                                          val_dataset.classes,
                                                          "eval")

        # compute the training accuracies related to box heights
        accuracies_eval_hb = compute_accuracies_per_box_height(labels,
                                                               greedy_decodings,
                                                               val_dataset.classes,
                                                               box_heights,
                                                               args["box_heights_intervals"],
                                                               "eval")

        # compute the validation accuracies of number detection
        accuracies_num_det_eval = compute_number_detection_accuracies(labels,
                                                                      greedy_decodings,
                                                                      val_dataset.classes,
                                                                      "eval")
        # choose 4 random pictures for tb visualization
        try:
            random_idx = random.sample(range(len(image_paths)), 4)
        except ValueError:  # This exception can occur if the very batch contains less then 3 elements
            random_idx = range(len(image_paths))

        decorated_images = decorate_tb_image([image_paths[idx] for idx in random_idx],
                                             [labels[idx] for idx in random_idx],
                                             [greedy_decodings[idx] for idx in random_idx],
                                             [closest_gts[idx] for idx in random_idx])

        # Write the summaries to tensorboard
        summary_writer.add_scalar("CTC_loss_eval", ctc_loss_eval.item(), batch_iter_eval)
        summary_writer.add_scalars("Eval_accuracies", accuracies_eval, batch_iter_eval)
        summary_writer.add_image("Eval", decorated_images, batch_iter_eval, dataformats="NHWC")
        for k, v in accuracies_eval_hb.items():
            summary_writer.add_scalars(k, v, batch_iter_eval)

        if accuracies_num_det_eval is not None:
            summary_writer.add_scalars("Eval_num_det_accuracies",
                                       accuracies_num_det_eval,
                                       batch_iter_eval)
        logging.info("\n Val. iter. {}: loss = {} | exact_acc = {} | exact_acc_after_mapping_ = {}".format(
            batch_iter_eval,
            ctc_loss_eval.item(),
            accuracies_eval["exact_acc_eval"],
            accuracies_eval["exact_acc_after_mapping_" + "eval"])
        )

        batch_iter_eval += 1

        # Release GPU memory cache
        torch.cuda.empty_cache()

    print("Inference time pro ts = {} seconds".format(total_t / len(val_dataset)))


if __name__ == "__main__":
    args = _get_args()
    eval(args)
