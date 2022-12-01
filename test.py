import torch
import torch.nn as nn
from math import inf
import numpy as np
from torch import optim
from neural_astar.data_loader import create_dataloader
from cnn_files.main import NeuralAstar, Astar
from time import time
import humanize
import datetime as dt

neural_star = NeuralAstar().to(device="cpu")
neural_star.load_state_dict(torch.load('models/l1_loss/Adam/best.pt'))

testing_dataset = create_dataloader('data/neural_astar/mazes_032_moore_c8.npz', 'test', 100, shuffle=False)
for batch in testing_dataset:
    neural_star.eval_and_calc_loss(batch, "cpu", nn.L1Loss())

