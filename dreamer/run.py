"""
This file defines the core research contribution   
"""
import os
import torch
from torch.utils.data import DataLoader
from argparse import ArgumentParser
import pytorch_lightning as pl

from models.dreamer import Dreamer

if __name__ == '__main__':
   
    parser = ArgumentParser(add_help=False)
    
    # dreamer hyperparameters
    parser.add_argument('--seed_episodes', default=10, type=int)
    parser.add_argument('--collect_interval', default=0.02, type=float)
    parser.add_argument('--batch_size', default=0.02, type=float)
    parser.add_argument('--sequence_length', default=0.02, type=float)
    parser.add_argument('--imagination_horizon', default=0.02, type=float)
    parser.add_argument('--learning_rate', default=0.02, type=float)
    parser.add_argument('--state_size', default=10, type=int,
     help="size of state generated internally by the representation model")

    # environment
    parser.add_argument('-env', default="procgen:procgen-coinrun-v0", type=str,
     help="environement used in run")

    parser.add_argument('--observation_size', default=(64, 64, 3), type=tuple)
    parser.add_argument('--action_size', default=15, type=int)

    # network architecture

    parser.add_argument('--kernal_size', default=3, type=int)
    parser.add_argument('--num_channels', default=64, type=int)
    parser.add_argument('--num_res', default=3, type=int)

    self.kernal_size = self.hparams.kernal_size
        self.num_channel = self.hparams.num_channels


    # parse params
    args = parser.parse_args()

    # init module
    dreamer = Dreamer(hparams=args)

    dreamer.dream()


