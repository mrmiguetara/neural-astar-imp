from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def _st_softmax_noexp(val: torch.tensor) -> torch.tensor:
    val_ = val.reshape(val.shape[0], -1)
    y = val_ / (val_.sum(dim=-1, keepdim=True))
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y)
    y_hard[range(len(y_hard)), ind] = 1
    y_hard = y_hard.reshape_as(val)
    y = y.reshape_as(val)
    return (y_hard - y).detach() + y

@dataclass
class AstarOutput:
    histories: torch.tensor
    paths: torch.tensor
def get_heuristic(goal_maps: torch.tensor,
                  tb_factor: float = 0.001) -> torch.tensor:
    num_samples, size = goal_maps.shape[0], goal_maps.shape[-1]
    grid = torch.meshgrid(torch.arange(0, size), torch.arange(0, size), indexing='ij')
    loc = torch.stack(grid, dim=0).type_as(goal_maps)
    loc_expand = loc.reshape(2, -1).unsqueeze(0).expand(num_samples, 2, -1)
    goal_loc = torch.einsum("kij, bij -> bk", loc, goal_maps)
    goal_loc_expand = goal_loc.unsqueeze(-1).expand(num_samples, 2, -1)

    dxdy = torch.abs(loc_expand - goal_loc_expand)
    h = dxdy.sum(dim=1) - dxdy.min(dim=1)[0]
    euc = torch.sqrt(((loc_expand - goal_loc_expand)**2).sum(1))
    h = (h + tb_factor * euc).reshape_as(goal_maps)

    return h

class DifferentiableAstar(nn.Module):
    def __init__(self, g_ratio, Tmax) -> None:
        super().__init__()

        neightbor_filter = torch.ones(1,1,3,3)
        neightbor_filter[0,0,1,1] = 0
        self.get_heuristic = get_heuristic
        self.neightbor_filter = nn.Parameter(neightbor_filter, requires_grad=False)

        self.g_ratio = g_ratio
        self.Tmax = Tmax
    @classmethod
    def expand(cls,x, neighbor_filter):
        x = x.unsqueeze(0)
        num_samples = x.shape[1]
        y = F.conv2d(x, neighbor_filter, padding=1, groups=num_samples).squeeze()
        y = y.squeeze(0)
        return y
    @classmethod
    def backtrack(cls, start_maps, goal_maps, parents, current_t):
        num_samples = start_maps.shape[0]
        parents = parents.type(torch.long)
        goal_maps = goal_maps.type(torch.long)
        start_maps = start_maps.type(torch.long)
        path_maps = goal_maps.type(torch.long)
        num_samples = len(parents)
        loc = (parents * goal_maps.view(num_samples, -1)).sum(-1)
        for _ in range(current_t):
            path_maps.view(num_samples, -1)[range(num_samples), loc] = 1
            loc = parents[range(num_samples), loc]
        return path_maps
    def forward(self, cost_maps: torch.tensor, start_maps, goal_maps, obstacles_maps):
        cost_maps = cost_maps[:, 0]
        start_maps = start_maps[:,0]
        goal_maps = goal_maps[:,0]
        obstacles_maps = obstacles_maps[:,0]

        num_samples = start_maps.shape[0]
        neightbor_filter = torch.repeat_interleave(self.neightbor_filter,num_samples,0)
        size = start_maps.shape[-1]

        histories = torch.zeros_like(start_maps)

        parents = (torch.ones_like(start_maps).reshape(num_samples, -1) * goal_maps.reshape(num_samples, -1).max(-1, keepdim=True)[-1])
        h = self.get_heuristic(goal_maps) + cost_maps
        g = torch.zeros_like(start_maps)
        size = cost_maps.shape[-1]
        Tmax = self.Tmax if self.training else 1.
        Tmax = int(Tmax * size * size)
        for t in range(Tmax):
            f = self.g_ratio * g + (1 - self.g_ratio) * h
            f_exp = torch.exp(-1 * f / math.sqrt(cost_maps.shape[-1])) * start_maps

            selected_node_maps = _st_softmax_noexp(f_exp)

            dist_to_goal = (selected_node_maps * goal_maps).sum((1,2), keepdim = True)
            is_unsolved = (dist_to_goal < 1e-8).float()

            histories = torch.clamp(histories + selected_node_maps, 0, 1)
            start_maps = torch.clamp(start_maps - is_unsolved * selected_node_maps, 0, 1)

            neighbor_nodes = self.expand(selected_node_maps, neightbor_filter) * obstacles_maps

            g2 = self.expand((g + cost_maps) * selected_node_maps, neightbor_filter)
            idx = (1 - start_maps) * (1 - histories) + start_maps * (g > g2)
            idx = (idx * neighbor_nodes).detach()
            g = (g2 * idx + g * (1 - idx)).detach()

            start_maps = torch.clamp(start_maps + idx, 0, 1).detach()

            idx = idx.reshape(num_samples, -1)
            snm = selected_node_maps.reshape(num_samples, -1)
            new_parents = snm.max(-1, keepdim=True)[1]
            parents = new_parents * idx + parents * (1 - idx)

            if torch.all(is_unsolved.flatten() == 0):
                break
        
        path_maps = self.backtrack(start_maps, goal_maps, parents, t)

        return AstarOutput(histories.unsqueeze(1), path_maps.unsqueeze(1))