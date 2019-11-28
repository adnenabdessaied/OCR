#!/usr/bin/env python
__author__ = "Mohamed Adnen Abdessaied"
__maintainer__ = "Mohamed Adnen Abdessaied"
__email__ = "adnenabdessayed@gmail.com"
__status__ = "Implementation"


import torch.nn as nn


class ConvINReLUBlock(nn.Sequential):
    """
    A Class that implements the conv2D -> Instance normalization -> ReLU block.
    Please refer to "E2E-MLT - an Unconstrained End-to-End Method for Multi-Language Scene Text" for further information
    -->  (arXiv:1801.09919)
    """
    def __init__(self, conv_channels_in, conv_channels_out, kernel_size, padding, affine=True):
        """
        Class constructor.
        :param conv_channels_in: The number of input channels to the 2D convolution.
        :param conv_channels_out: The number of output channels of the 2D convolution.
        :param kernel_size: The size of the convolution kernel.
        :param padding: Padding of the convolution layer.
        :param affine: If true, the $\alpha$ and $\beta$ parameters of the normalization will be learnt during training.
        """
        super(ConvINReLUBlock, self).__init__(nn.Conv2d(conv_channels_in,
                                                        conv_channels_out,
                                                        kernel_size,
                                                        padding=padding),
                                              nn.InstanceNorm2d(conv_channels_out, affine=affine),
                                              nn.ReLU())


class OCR_NET(nn.Module):
    """
    This class implements the OCR branch introduced in "E2E-MLT - an Unconstrained End-to-End Method for Multi-Language
    Scene Text" (arXiv:1801.09919).
    """
    def __init__(self, alphabet="1234567890abcdefghijklmnopqrstuvwxyzßäöü().,_+-#"):
        """
        Class constructor.
        :param alphabet_size: The size of the alphabet that we want to use in OCR. For our purposes, only the german
                              letters will be considered, i.e.
                              alphabet = "1234567890abcdefghijklmnopqrstuvwxyzßäöü().,_+-#"
        """
        super(OCR_NET, self).__init__()
        self.alphabet = alphabet
        self.blank_idx = self.alphabet.index("-")
        self.block1 = ConvINReLUBlock(3, 128, 3, (0, 1))
        self.block2 = ConvINReLUBlock(128, 128, 3, (0, 1))
        self.maxpool1 = nn.MaxPool2d((2, 1), stride=(2, 1))
        self.block3 = ConvINReLUBlock(128, 256, 3, (0, 1))
        self.block4 = ConvINReLUBlock(256, 256, 3, (0, 1))
        self.maxpool2 = nn.MaxPool2d((2, 1), stride=(2, 1))
        self.block5 = ConvINReLUBlock(256, 512, 3, (0, 1))
        self.block6 = ConvINReLUBlock(512, 512, 3, (0, 1))
        self.dropout = nn.Dropout2d(p=0.2)

        # the +1 in "len(self.alphabet) + 1" accounts for the blank character.
        self.conv_last = nn.Conv2d(512, len(self.alphabet) + 1, 1)

    def forward(self, x):
        """
        Forward pass of the ocr branch.
        :param x: torch.Tensor of size (batch_size, C, H, W) where C = 3, H = 32, W = 100
        :return: The output of the ocr branch after the forward pass.
        """
        # As described in the paper "E2E-MLT - an Unconstrained End-to-End Method for Multi-Language Scene Text", we
        # keep the width of the input unchanged and we reduce its height to 1.

        # Shape: (batch_size, 128, 30, 100)
        x = self.block1(x)

        # Shape: (batch_size, 128, 28, 100)
        x = self.block2(x)

        # Shape: (batch_size, 128, 14, 50)
        x = self.maxpool1(x)

        # Shape: (batch_size, 256, 12, 50)
        x = self.block3(x)

        # Shape: (batch_size, 256, 10, 50)
        x = self.block4(x)

        # Shape: (batch_size, 256, 5, 25)
        x = self.maxpool2(x)

        # Shape: (batch_size, 256, 3, 25)
        x = self.block5(x)

        # Shape: (batch_size, 256, 1, 25)
        x = self.block6(x)

        # Shape: (batch_size, 256, 1, 25)
        x = self.dropout(x)

        # Shape: (batch_size, 48, 1, 25) # len(alphabet) = 48
        x = self.conv_last(x)

        # Shape:(batch_size, 48, 25)
        x = x.squeeze(2)

        # Shape: (batch_size, 50, 48) --> keep the last dimension for the alphabet size
        x = x.permute(0, 2, 1)

        x_shape = x.size()

        # Shape: (batch_size * 25, 48)
        x = x.contiguous().view(-1, len(self.alphabet) + 1)

        # Shape: (batch_size * 25, 48) --> We apply a logsoftmax along dim=1, i.e. over the alphabet.
        x = nn.LogSoftmax(dim=1)(x)

        # Shape: (batch_size, 25, 48)
        x = x.view(*x_shape)

        # The input to the ctc loss function must have the shape (T, N, C):
        #         T: Input sequence length
        #         N: batch size
        #         C: Number of classes (including blank)
        # Shape: (25, batch_size, 48)
        x = x.permute(1, 0, 2)
        return x
