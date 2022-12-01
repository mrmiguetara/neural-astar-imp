import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from .modules import DifferentiableAstar 
from .encoders import CNN


class NeuralAstar(nn.Module):
    def __init__(self, g_ratio = 0.5, Tmax = 0.25) -> None:
        super().__init__()
        self.astar = DifferentiableAstar(g_ratio, Tmax)
        self.encoder_input = "1"
        self.encoder = CNN(1, 4, None)
    
    def forward(self, map_designs, start_maps, goal_maps):
        pred_cost_maps = self.encoder(map_designs)
        astar_outputs = self.astar(pred_cost_maps, start_maps, goal_maps, map_designs)
        return astar_outputs
    def train_and_calc_loss(self, batch, device, loss_fn):
        self.train()
        map_designs, start_maps, goal_maps, opt_trajs = batch
        map_designs = map_designs.to(device)
        start_maps = start_maps.to(device)
        goal_maps = goal_maps.to(device)
        opt_trajs = opt_trajs.to(device)
        astar_output = self(map_designs, start_maps, goal_maps)
        loss = loss_fn(astar_output.histories, opt_trajs)
        return loss, astar_output
    def eval_and_calc_loss(self, batch, device, loss_fn):
        self.eval()
        map_designs, start_maps, goal_maps, opt_trajs = batch
        map_designs = map_designs.to(device)
        start_maps = start_maps.to(device)
        goal_maps = goal_maps.to(device)
        opt_trajs = opt_trajs.to(device)
        astar_output = self(map_designs, start_maps, goal_maps)
        loss = loss_fn(astar_output.histories, opt_trajs)
        return loss, astar_output


class Astar(nn.Module):
    def __init__(self, g_ratio = 0.5):
        super().__init__()
        self.astar = DifferentiableAstar(g_ratio, Tmax=1)

    def forward(self, map_designs, start_maps, goal_maps):
        obstacles_maps = map_designs
        return self.astar(map_designs, start_maps, goal_maps, obstacles_maps)

    def train_and_calc_loss(self, batch, device, loss_fn):
        map_designs, start_maps, goal_maps, opt_trajs = batch
        map_designs = map_designs.to(device)
        start_maps = start_maps.to(device)
        goal_maps = goal_maps.to(device)
        opt_trajs = opt_trajs.to(device)
        astar_output = self(map_designs, start_maps, goal_maps)
        loss = loss_fn(astar_output.histories, opt_trajs)
        return loss, astar_output
    def eval_and_calc_loss(self, batch, device, loss_fn):
        map_designs, start_maps, goal_maps, opt_trajs = batch
        map_designs = map_designs.to(device)
        start_maps = start_maps.to(device)
        goal_maps = goal_maps.to(device)
        opt_trajs = opt_trajs.to(device)
        astar_output = self(map_designs, start_maps, goal_maps)
        loss = loss_fn(astar_output.histories, opt_trajs)
        return loss, astar_output