#!/usr/bin/env python
__author__ = "Mohamed Adnen Abdessaied"
__maintainer__ = "Mohamed Adnen Abdessaied"
__email__ = "adnenabdessayed@gmail.com"
__status__ = "Implementation"

import os
import logging
import numpy as np
import functools
import argparse
from datetime import datetime
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
import torch.optim as optim
from torch.nn.modules.loss import CTCLoss
from torch.utils.data.dataloader import DataLoader
from E2E_MLT.datasets.e2e_mlt_datasets import E2E_MLT_Dataset
from E2E_MLT.ocr.ocr_net import OCR_NET


logging.basicConfig(level=logging.INFO)


def _get_current_timestamp() -> str:
    """
    A function that returns the current timestamp to be used to name the checkpoint files of the model.
    :return: The current timestamp.
    """
    current_time = datetime.now()
    current_time = current_time.strftime("%d-%b-%Y (%H:%M:%S.%f)")
    return current_time


def _get_ctc_tensors(label_batch, alphabet, device):
    """
    This functions computes the ctc targets and their lengths. The target are padded to have the same lengths as the
    longest label in the batch. The original lengths will be calculated, i.e. before padding.
    :param label_batch: The label batch.
    :param alphabet: The alphabet considered.
    :param device: The device that will perform the computations.
    :return: The ctc target and the target lengths as torch.tensors of shape (N, S)  and (N) respectively where
                N: Batch size
                S: The length of the longest label in the batch
    """

    label_batch = list(map(lambda x: x.lower(), label_batch))
    label_batch = list(map(lambda x: x.replace(" ", ""), label_batch))
    max_label_length = len(max(label_batch, key=len))
    target_lengths = torch.tensor([len(label) for label in label_batch]).long()
    pad_index = alphabet.find("-")

    def get_indices_for_label(label: str):
        # +1 accounts for the blank character. Note that the ctc loss function we use expects blank_idx = 0
        return [alphabet.find(c) + 1 for c in label]

    def pad_labels(label_indices_, max_label_length_, pad_index_):
        assert len(label_indices_) <= max_label_length_
        return label_indices_ + [pad_index_] * (max_label_length_ - len(label_indices_))

    ctc_targets = list(map(get_indices_for_label, label_batch))
    ctc_targets = list(map(functools.partial(pad_labels,
                                             max_label_length_=max_label_length,
                                             pad_index_=pad_index), ctc_targets))
    ctc_targets = torch.tensor(ctc_targets).long()

    return ctc_targets.to(device), target_lengths.to(device)


def _get_args():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-tri",
                            "--training_images_path",
                            help="Path to the folder containing the training images.")

    arg_parser.add_argument("-trl",
                            "--training_labels_path",
                            help="Path to the folder containing the training labels.")

    arg_parser.add_argument("-vali",
                            "--validation_images_path",
                            help="Path to the folder containing the validation images.")

    arg_parser.add_argument("-vall",
                            "--validation_labels_path",
                            help="Path to the folder containing the validation labels.")

    arg_parser.add_argument("-b",
                            "--batch_size",
                            default=8,
                            help="Batch size.")

    arg_parser.add_argument("-e",
                            "--epochs",
                            default=50,
                            help="Number of epochs.")

    arg_parser.add_argument("-pre",
                            "--pre_trained",
                            default=False,
                            help="Use a pre-trained net on synthetic data.")

    arg_parser.add_argument("-bhi",
                            "--box_heights_intervals",
                            default=[8, 10, 12, 14, 16],
                            help="The box height intervals based on which we want to investigate the accuracies.")

    arg_parser.add_argument("-tb",
                            "--tensorboard",
                            help="Tensorboard summaries directory.")

    arg_parser.add_argument("-chkpt",
                            "--checkpoints",
                            required=False,
                            help="Directory for check-pointing the network.")

    arg_parser.add_argument("-chkpt_synth",
                            "--checkpoints_synthetic",
                            required=False,
                            help="Directory containing check-points of pre-trained network.")
    
    arg_parser.add_argument("-bchkpt",
                            "--best_checkpoint",
                            required=False,
                            help="Directory for check-pointing the network.")

    args = vars(arg_parser.parse_args())

    # Make sure the interval boundaries are in ascending order
    args["box_heights_intervals"].sort()

    return args


def _save_checkpoint(net, optimizer, epoch, batch_iter_tr, batch_iter_val, args):
    """
    A function that saves a checkpoint of a network.
    :param net: The network we want to checkpoint.
    :param optimizer: The optimizer used in the experiment.
    :param epoch: The epoch when check-pointing the net.
    :param batch_iter_tr: The training batch iteration at the moment of check-pointing.
    :param batch_iter_val: The validation batch iteration at the moment of check-pointing.
    :param args: Arguments of the experiment.
    :return:
    """
    timestamp = _get_current_timestamp()
    torch.save({
        "net": net,
        "net_state_dict": net.state_dict(),
        "optimizer": optimizer,
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "batch_iter_tr": batch_iter_tr,
        "batch_iter_val": batch_iter_val,
    }, os.path.join(args["checkpoints"], "checkpoint_{}.pth".format(timestamp)))


def _load_net(checkpoints_path, device):
    """
    Loads a saved model for fine tuning or evaluation
    :param checkpoints_path: Path to the folder containing the checkpoints.
    :param device: The device used for training/evaluation
    :return: net, optimizer, start_epoch, batch_iter_tr, batch_iter_val
    """
    # Load the most recent checkpoint. Otherwise start training from scratch.
    checkpoints = [ckpt for ckpt in os.listdir(checkpoints_path) if ckpt.endswith("pth")]
    checkpoints = [os.path.join(checkpoints_path, checkpoint) for checkpoint in checkpoints]
    most_recent_chkpt_path = max(checkpoints, key=os.path.getctime)
    most_recent_chkpt = torch.load(most_recent_chkpt_path)
    net = most_recent_chkpt["net"]
    net.load_state_dict(most_recent_chkpt["net_state_dict"])

    # We want to train further
    net.train()

    # Send the net first to the device to avoid potential runtime errors caused by the optimizer if we resume
    # training on a different device
    net.to(device)

    optimizer = most_recent_chkpt["optimizer"]
    optimizer.load_state_dict(most_recent_chkpt["optimizer_state_dict"])

    start_epoch = most_recent_chkpt["epoch"]
    batch_iter_tr = most_recent_chkpt["batch_iter_tr"]
    batch_iter_val = most_recent_chkpt["batch_iter_val"]
    chkpt_timestamp = os.path.getmtime(most_recent_chkpt_path)
    logging.info("Network loaded from the latest checkpoint saved on {}".format(datetime.fromtimestamp(
        chkpt_timestamp)))
    return net, optimizer, start_epoch, batch_iter_tr, batch_iter_val


def _get_device():
    """
    Returns the available device: cuda:0 when using a GPU or CPU if cuda was not detected.
    :return: device
    """
    # Use GPU if available
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        logging.info("Using {}".format(torch.cuda.get_device_name()))
    else:
        device = torch.device("cpu")
        logging.info("Using the CPU")
    return device


def train(args):
    # Reproducibility --> https://pytorch.org/docs/stable/notes/randomness
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(0)

    device = _get_device()

    if args["pre_trained"]:
        checkpoints_synth = [ckpt for ckpt in os.listdir(args["checkpoints_synthetic"]) if ckpt.endswith("pth")]
        checkpoints_synth = [os.path.join(args["checkpoints_synthetic"],
                                          checkpoint) for checkpoint in checkpoints_synth]
        most_recent_chkpt_path = max(checkpoints_synth, key=os.path.getctime)
        most_recent_chkpt = torch.load(most_recent_chkpt_path)
        net = most_recent_chkpt["net"]
        net.load_state_dict(most_recent_chkpt["net_state_dict"])

        # We use learning rate of 1e-4 as suggested in the paper --> https://arxiv.org/pdf/1801.09919.pdf
        optimizer = optim.Adam(net.parameters(), lr=1e-4, weight_decay=1e-5)
        start_epoch = 0
        batch_iter_tr = 0
        batch_iter_val = 0
        logging.info("Pre-trained net successfully loaded!")
    else:
        try:
            net, optimizer, start_epoch, batch_iter_tr, batch_iter_val = _load_net(args["checkpoints"], device)
        except:
            # Construct the OCR network from scratch
            net = OCR_NET()
            net.to(device)
            logging.info("OCR network successfully constructed...")

            # We use learning rate of 1e-4 as suggested in the paper --> https://arxiv.org/pdf/1801.09919.pdf
            optimizer = optim.Adam(net.parameters(), lr=1e-4, weight_decay=1e-5)
            start_epoch = 0
            batch_iter_tr = 0
            batch_iter_val = 0

    # Load the datasets
    tr_dataset = E2E_MLT_Dataset(args["training_images_path"], args["training_labels_path"])
    val_dataset = E2E_MLT_Dataset(args["validation_images_path"], args["validation_labels_path"])
    logging.info("Data successfully loaded ...")

    # Construct train and validation data loaders
    logging.info("Constructing the data loaders ...")
    tr_data_loader = DataLoader(tr_dataset, batch_size=args["batch_size"], shuffle=True, num_workers=6)
    val_data_loader = DataLoader(val_dataset, batch_size=int(args["batch_size"] / 2), shuffle=True, num_workers=6)
    logging.info("Data loaders successfully constructed ...")

    # We use the ctc loss function.
    criterion = CTCLoss(zero_infinity=True)

    # Define the summary writer to be used for tensorboard visualizations.
    summary_writer = SummaryWriter(log_dir=args["tensorboard"])

    modes = ["Training", "Validation"]
    best_loss_val = 0
    for epoch in range(start_epoch + 1, args["epochs"] + 1):
        for mode in modes:
            if mode == "Training":
                pbar_train = tqdm(tr_data_loader)
                pbar_train.set_description("{} | Epoch {} / {}".format(mode, epoch, args["epochs"]))
                for i, (images, image_paths, labels, box_heights) in enumerate(pbar_train):
                    optimizer.zero_grad()

                    # Strip all whitespaces because standard 'best path decoding' can't deal with whitespaces between
                    # words
                    labels = list(map(lambda x: x.replace(" ", ""), labels))
                    labels = list(map(lambda x: x.lower(), labels))

                    images = images.to(device)
                    ctc_targets, target_lengths = _get_ctc_tensors(labels, net.alphabet, device)

                    # Shape: (50, batch_size, 49)
                    ocr_outputs = net(images)

                    # The ctc loss requires the input lengths as it allows for variable length inputs. In our case,
                    # all the inputs have the same length.
                    ocr_input_lengths = torch.full(size=(ocr_outputs.size(1),),
                                                   fill_value=ocr_outputs.size(0),
                                                   dtype=torch.long).to(device)

                    # ocr_outputs are the inputs of the ctc_loss.
                    ctc_loss_tr = criterion(ocr_outputs, ctc_targets, ocr_input_lengths, target_lengths)

                    # Back propagation with anomaly detection -> Makes it easier to locate the faulty parts of the net
                    # if some undesirable phenomena happen, e.g. if some layers produce NaN of Inf values.
                    with torch.autograd.detect_anomaly():
                        ctc_loss_tr.backward()

                    torch.nn.utils.clip_grad_norm_(net.parameters(), 0.1)
                    optimizer.step()

                    box_heights = box_heights.tolist()

                    # compute the training accuracies
                    greedy_decodings = from_probabilities_to_letters(ocr_outputs, net.alphabet)
                    accuracies_tr, closest_gts = compute_accuracies(labels,
                                                                    greedy_decodings,
                                                                    tr_dataset.classes,
                                                                    mode)

                    # compute the training accuracies related to box heights
                    accuracies_tr_hb = compute_accuracies_per_box_height(labels,
                                                                         greedy_decodings,
                                                                         tr_dataset.classes,
                                                                         box_heights,
                                                                         args["box_heights_intervals"],
                                                                         mode)

                    # compute the training accuracies of number detection
                    accuracies_num_det_tr = compute_number_detection_accuracies(labels,
                                                                                greedy_decodings,
                                                                                tr_dataset.classes,
                                                                                mode)

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
                    summary_writer.add_scalar("CTC_loss_tr", ctc_loss_tr.item(), batch_iter_tr)
                    summary_writer.add_scalars("Training_accuracies", accuracies_tr, batch_iter_tr)
                    for k, v in accuracies_tr_hb.items():
                        summary_writer.add_scalars(k, v, batch_iter_tr)
                    if accuracies_num_det_tr is not None:
                        summary_writer.add_scalars("Training_num_det_accuracies", accuracies_num_det_tr, batch_iter_tr)

                    summary_writer.add_images(mode, decorated_images, batch_iter_tr, dataformats="NHWC")

                    logging.info("\n Tr. iter. {}: loss = {} | exact_acc = {} | exact_acc_after_mapping_ = {}".format(
                        batch_iter_tr,
                        ctc_loss_tr.item(),
                        accuracies_tr["exact_acc_" + mode],
                        accuracies_tr["exact_acc_after_mapping_" + mode])
                    )

                    batch_iter_tr += 1
                    torch.cuda.empty_cache()

                _save_checkpoint(net, optimizer, epoch, batch_iter_tr, batch_iter_val, args)

                # Delete the oldest checkpoint if the number of checkpoints exceeds 10 to save disk space.
                checkpoints = [ckpt for ckpt in os.listdir(args["checkpoints"]) if ckpt.endswith("pth")]
                checkpoints = [os.path.join(args["checkpoints"], checkpoint) for checkpoint in checkpoints]
                if len(checkpoints) > 10:
                    oldest_checkpoint_pth = min(checkpoints, key=os.path.getctime)
                    os.remove(oldest_checkpoint_pth)
            else:
                # Set the net to eval mode
                net.eval()
                pbar_val = tqdm(val_data_loader)
                pbar_val.set_description("{} | Epoch {} / {}".format(mode, epoch, args["epochs"]))
                total_t = 0
                val_loss_epoch = 0
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

                    # ocr_outputs are the inputs of the ctc_loss.
                    ctc_loss_val = criterion(ocr_outputs, ctc_targets, ocr_input_lengths, target_lengths)

                    # Update the val_loss_epoch
                    val_loss_epoch += ctc_loss_val

                    # compute the validation accuracies
                    start_t = time()
                    greedy_decodings = from_probabilities_to_letters(ocr_outputs, net.alphabet)
                    total_t += time() - start_t
                    accuracies_val, closest_gts = compute_accuracies(labels,
                                                                     greedy_decodings,
                                                                     val_dataset.classes,
                                                                     mode)
                    # compute the training accuracies related to box heights
                    accuracies_val_hb = compute_accuracies_per_box_height(labels,
                                                                          greedy_decodings,
                                                                          val_dataset.classes,
                                                                          box_heights,
                                                                          args["box_heights_intervals"],
                                                                          mode)

                    # compute the validation accuracies of number detection
                    accuracies_num_det_val = compute_number_detection_accuracies(labels,
                                                                                 greedy_decodings,
                                                                                 val_dataset.classes,
                                                                                 mode)
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
                    summary_writer.add_scalar("CTC_loss_val", ctc_loss_val.item(), batch_iter_val)
                    summary_writer.add_scalars("Validation_accuracies", accuracies_val, batch_iter_val)
                    summary_writer.add_image(mode, decorated_images, batch_iter_val, dataformats="NHWC")
                    for k, v in accuracies_val_hb.items():
                        summary_writer.add_scalars(k, v, batch_iter_val)

                    if accuracies_num_det_val is not None:
                        summary_writer.add_scalars("val_num_det_accuracies",
                                                   accuracies_num_det_val,
                                                   batch_iter_val)
                    logging.info("\n Val. iter. {}: loss = {} | exact_acc = {} | exact_acc_after_mapping_ = {}".format(
                        batch_iter_tr,
                        ctc_loss_val.item(),
                        accuracies_val["exact_acc_" + mode],
                        accuracies_val["exact_acc_after_mapping_" + mode])
                    )

                    batch_iter_val += 1

                    # Release GPU memory cache
                    torch.cuda.empty_cache()

                # Save the best net, i.e. the net that scores the best loss function on the validation set.
                if epoch == 1:
                    _save_checkpoint(net, optimizer, epoch, batch_iter_tr, batch_iter_val, args)

                    best_loss_val = val_loss_epoch
                elif val_loss_epoch <= best_loss_val:
                    # Delete the older best checkpoints
                    checkpoints = os.listdir(args["best_checkpoint"])
                    checkpoints = [os.path.join(args["best_checkpoint"], checkpoint) for checkpoint in checkpoints]
                    [os.remove(checkpoint) for checkpoint in checkpoints]

                    _save_checkpoint(net, optimizer, epoch, batch_iter_tr, batch_iter_val, args)

                    best_loss_val = val_loss_epoch

                print("Inference time pro ts = {} seconds".format(total_t/len(val_dataset)))
                # Switch back to train mode
                net.train()


if __name__ == "__main__":
    args = _get_args()
    train(args)
    
