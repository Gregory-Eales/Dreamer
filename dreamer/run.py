"""
This file defines the core research contribution   
"""
import os
import torch
from torch.utils.data import DataLoader
from argparse import ArgumentParser
import pytorch_lightning as pl

from dreamer.models.dreamer import Dreamer

if __name__ == '__main__':
   
    parser = ArgumentParser(add_help=False)
    
    # dreamer hyperparameters
    parser.add_argument('--seed_episodes', default=10, type=int)
    parser.add_argument('--collect_interval', default=0.02, type=float)
    parser.add_argument('--batch_size', default=0.02, type=float)
    parser.add_argument('--sequence_length', default=0.02, type=float)
    parser.add_argument('--imagination_horizon', default=0.02, type=float)
    parser.add_argument('--learning_rate', default=0.02, type=float)

    # environment


    # parse params
    args = parser.parse_args()

    # init module
    dreamer = Dreamer(hparams=args)

    dreamer.dream()


