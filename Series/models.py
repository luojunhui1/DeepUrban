import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import pandas as pd
import numpy as np
from datetime import *
import sys, os
sys.path.append(os.path.dirname(__file__) + os.sep + '../')
from Utils.utils import *
from Utils.utils import *
import os


def mean_absolute_percentage_error(y_true, y_pred):
    ads = []
    for i in range(0, len(y_true)):
        ad = 1. * (y_true[i, 0] - y_pred[i, 0]) / (y_true[i, 0] + 1) + \
             1. * (y_true[i, 1] - y_pred[i, 1]) / (y_true[i, 1] + 1)
        ads.append(ad)
    return ads


class seq2seq(nn.Module):
    def __init__(self, input_dim1, input_dim2, hid_dim, output_dim):
        super(seq2seq, self).__init__()
        self.embedding_hour = nn.Embedding(24, 1)
        self.embedding_week = nn.Embedding(7, 1)

        self.encode = nn.LSTM(input_dim1, hid_dim, num_layers=1)
        self.decode = nn.LSTM(input_dim2, hid_dim, num_layers=1)
        self.out = nn.Linear(hid_dim, output_dim)

    def forward(self, x, trg, isTrain):
        x1 = self.embedding_hour(x[:, :, -1].long())
        x2 = self.embedding_week(x[:, :, -2].long())
        encode_len = x.shape[0]
        batch_size = x.shape[1]

        x_input = torch.cat([x1, x2, x[:, :, :-2]], 2)

        outputs, encode_state = self.encode(x_input)

        decode_len = trg.shape[0]
        batch_size = trg.shape[1]
        outputs = torch.zeros(decode_len, batch_size, 2)

        for t in range(decode_len):
            if t == 0:
                decode_input = torch.zeros_like(trg[0, :, :])
                h_next = encode_state
            else:
                decode_input = trg[t, :, :] if isTrain else prediction
            h_now = h_next
            decode_output, h_next = self.decode(decode_input.reshape(1, batch_size, 2), h_now)
            prediction = self.out(decode_output.squeeze(0))
            outputs[t] = prediction
        return outputs