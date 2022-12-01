
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

loss_fns = {'mean_square': nn.MSELoss, 'l1_loss': nn.L1Loss}
optimization_algorithms = {'RMSprop': optim.RMSprop, 'Adam': optim.Adam}

device = "cuda" if torch.cuda.is_available() else "cpu"

neural_astar = NeuralAstar()
neural_astar.to(device) 

astar = Astar()
astar.to(device)
epochs = 50

for loss_fn in loss_fns:
    for o in optimization_algorithms:
        optimization_algorithm = optimization_algorithms[o](neural_astar.parameters(), lr=0.001)
        train_dataset = create_dataloader('data/neural_astar/mazes_032_moore_c8.npz', 'train', 100, shuffle=True)
        validation_dataset = create_dataloader('data/neural_astar/mazes_032_moore_c8.npz', 'valid', 100, shuffle=False)
        testing_dataset = create_dataloader('data/neural_astar/mazes_032_moore_c8.npz', 'test', 100, shuffle=False)

        training_start = time()
        for i in range(epochs):
            print(f'=========starting epoch {i + 1}===========')
            start = time()
            train_loss = 0
            validation_loss = 0
            total_POR = 0
            total_RNE = 0
            total_HN = 0
            best_rne = -inf
            for batch in train_dataset:
                loss, _ = neural_astar.train_and_calc_loss(batch, device, loss_fns[loss_fn]())
                train_loss += loss.item()
                optimization_algorithm.zero_grad()
                loss.backward()
                optimization_algorithm.step()
            train_loss /= len(train_dataset)


            with torch.no_grad():
                for batch in validation_dataset:
                    neural_loss, neural_output = neural_astar.train_and_calc_loss(batch, device, loss_fns[loss_fn]())
                    astar_loss, astar_output = astar.train_and_calc_loss(batch, device, loss_fns[loss_fn]())
                    #reducing dimensionality to a 2D matrix and detaching them so pytorch doesn't calculate their gradients
                    reduced_neural_paths = neural_output.paths.sum((1,2,3)).detach().numpy()
                    reduced_astar_paths = astar_output.paths.sum((1,2,3)).detach().numpy()
                    reduced_neural_visited = neural_output.histories.sum((1,2,3)).detach().numpy()
                    reduced_astar_visited = astar_output.histories.sum((1,2,3)).detach().numpy()
                    # calculate path optimality ration
                    POR = (reduced_neural_paths == reduced_astar_paths).mean()
                    # calculates reduction Ratio of node explored
                    RNE = np.maximum((reduced_astar_visited - reduced_neural_visited) / reduced_astar_visited, 0).mean()
                    # 1e-5 is for ensuring the values are not zero, but a very small number
                    HM = 2 / ( 1  / max(1e-5, POR) + 1 / max(1e-5, RNE))

                    total_POR += POR
                    total_RNE += RNE
                    total_HN += HM
                    validation_loss += loss.item()
            validation_loss /= len(train_dataset)
            total_POR /= len(validation_dataset)
            total_RNE /= len(validation_dataset)
            total_HN  /= len(validation_dataset)
            end = time()

            print(f'epoch: {i + 1}|     training loss: {train_loss} |    validation loss: {validation_loss} {humanize.naturaldelta(dt.timedelta(seconds=(end-start)))}')
            if total_RNE > best_rne:
                best_rne =  total_RNE
                #save the best model for testing
                torch.save(neural_astar.state_dict(), f"models/{loss_fn}/{o}/best.pt")
        training_end = time()
        print(f'Training time: {humanize.naturaldelta(training_end - training_start)}')
        #testing
        neural_astar.load_state_dict(torch.load(f'models/{loss_fn}/{o}/best.pt'))
        total_POR = 0
        total_RNE = 0
        total_HN = 0
        testing_start = time()
        with torch.no_grad():
            for batch in testing_dataset:
                neural_loss, neural_output = neural_astar.eval_and_calc_loss(batch, device, loss_fns[loss_fn]())
                astar_loss, astar_output = astar.eval_and_calc_loss(batch, device, loss_fns[loss_fn]())
                reduced_neural_paths = neural_output.paths.sum((1,2,3)).detach().numpy()
                reduced_astar_paths = astar_output.paths.sum((1,2,3)).detach().numpy()
                reduced_neural_visited = neural_output.histories.sum((1,2,3)).detach().numpy()
                reduced_astar_visited = astar_output.histories.sum((1,2,3)).detach().numpy()
                # calculate path optimality ration
                POR = (reduced_neural_paths == reduced_astar_paths).mean()
                # calculates reduction Ratio of node explored
                RNE = np.maximum((reduced_astar_visited - reduced_neural_visited) / reduced_astar_visited, 0).mean()
                # 1e-5 is for ensuring the values are not zero, but a very small number
                HM = 2 / ( 1  / max(1e-5, POR) + 1 / max(1e-5, RNE))
                total_POR += POR
                total_RNE += RNE
                total_HN += HM
            total_POR /= len(testing_dataset)
            total_RNE /= len(testing_dataset)
            total_HN /= len(testing_dataset)
        testing_end = time()
        with open(f'models/{loss_fn}/{o}/results.txt', 'w+') as f:
            f.write(f'Training time: {training_end - training_start}\n')
            f.write(f'Testing time: {testing_end - testing_end}\n')
            f.write(f'Path optimality Ratio: {total_POR}\n')
            f.write(f'Reduction Ratio of Node Explored: {total_RNE}\n')
            f.write(f'Harmonic Mean: {total_HN}\n')

