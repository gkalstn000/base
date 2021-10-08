import torch
from torch import nn, optim


def fc_layer(size_in, size_out, keep_prob, time_series = None, last_layer = False):
    linear = nn.Linear(size_in, size_out)
    torch.nn.init.xavier_uniform_(linear.weight)
    modules = [linear, nn.BatchNorm1d(*(time_series, size_out) if time_series else size_out),
               nn.Dropout(p = 1 - keep_prob)]
    if not last_layer : modules.insert(-1, nn.ReLU())
    layer = nn.Sequential(*modules)
    return layer
