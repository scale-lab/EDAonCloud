import csv
import dgl
import os
import sys
import torch
import argparse
import logging
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from torch.utils.data import DataLoader

from dataset import DesignDataset
from netlist import read_netlist


# setup device
device = torch.device("cuda:{}".format(0) if torch.cuda.is_available() else "cpu")

def collate(samples):
    IDs, Gs, Ms = [], [], []
    for s in samples:
        IDs.append(s['id'])
        Gs.append(s['design'])
        Ms.append(s['metrics'].view(-1, 1))
    return IDs, dgl.batch(Gs).to(device), torch.cat(Ms, dim=0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("output_dir", type=str, \
        help="Output directory of the model")
    parser.add_argument("-model", type=str, required=False, default=None, \
        help="Path of model. If passed, training is skipped.")
    parser.add_argument("-epochs", type=int, required=False, default=100, \
        help="Number of epochs")
    parser.add_argument("-lr", type=float, required=False, default=1e-4, \
        help="Learning rate")
    parser.add_argument("-batch_size", type=int, required=False, default=32, \
        help="Batch size")
    parser.add_argument("-gcn_hidden_dim", type=int, required=False, default=256, \
        help="Number of hidden units of GCN")
    parser.add_argument("-netlist_embedding_size", type=int, required=False, default=128, \
        help="Embedding size of the netlist")
    parser.add_argument("-predictor_hidden_dim", type=int, required=False, default=192, \
        help="Number of hidden units of the last-layer predictor")

    args = parser.parse_args()
    logging.getLogger('matplotlib.font_manager').disabled = True

    logger = get_logger()
    learning_rate = args.lr
    epochs = args.epochs
    batch_size = args.batch_size

    # setup playground
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    prefix = [args.epochs, args.lr, args.gcn_hidden_dim, args.netlist_embedding_size, \
        args.predictor_hidden_dim]
    prefix = '-'.join(list(map(str, prefix)))

    # load designs
    dgl_graphs = 'data/dgl'
    dgl_files = [f for f in os.listdir(dgl_graphs) if os.path.isfile(os.path.join(dgl_graphs, f)) and f.endswith(".dgl")]
    Gs = {}
    for dgl_file in dgl_files:
        design = '.'.join(dgl_file.split('.')[:-1])
        g = read_netlist('', os.path.join(dgl_graphs, design))
        design = '.'.join(dgl_file.split('.')[:-2])
        Gs[design] = g

    # load dataset
    train_dataset = DesignDataset(Gs, 'data/train.csv')
    test_dataset = DesignDataset(Gs, 'data/test.csv')
    
    # define model
    model = Predictor(2, args.gcn_hidden_dim, args.netlist_embedding_size, \
                      args.predictor_hidden_dim, 1).to(device)
    loss_func = nn.MSELoss()

    _min = 1.0
    _max = 12012.0
    if not args.model:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # train
        model.train()
        epoch_losses = []
        for epoch in range(epochs):
            epoch_loss = 0
            dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=batch_size, collate_fn=collate)
            for i_batch, (_, G, P, M) in enumerate(dataloader):
                _, prediction = model(G, P.to(device))
                loss = loss_func(prediction.to(device), M.to(device))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.detach().item()

                if i_batch % 100 == 0:
                    logger.info('Epoch: {}, iteration: {}, MSE: {:.4f}'.format(epoch, i_batch, epoch_loss / (i_batch+1)))
                
            i_batch += 1
            epoch_loss /= i_batch

            mse = epoch_loss
            rmse = mse ** 0.5
            scaled_rmse = (rmse * (_max - _min)) + _min
            logger.info('Epoch {}, MSE {:.4f}'.format(epoch, mse))
            logger.info('Epoch {}, RMSE {:.4f}'.format(epoch, rmse))
            logger.info('Epoch {}, Scaled RMSE {:.4f}'.format(epoch, scaled_rmse))
            epoch_losses.append(epoch_loss)

        
        with open(os.path.join(args.output_dir, prefix + '-training-losses.log'), 'w') as f:
            f.write('\n'.join(list(map(str, epoch_losses))))

        torch.save(model.state_dict(), os.path.join(args.output_dir, prefix + '-model.pth'))
    else:
        logger.info("Skipping training. Loading an existing model.")
        model.load_state_dict(torch.load(args.model))

    # test
    losses = []
    embeddings = []
    model.eval()
    dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=batch_size, collate_fn=collate)
    for i_batch, (ID, G, P, M) in enumerate(dataloader):
        print(i_batch)
        graph_embeeding, prediction = model(G, P.to(device))
        loss = loss_func(prediction.to(device), M.to(device))
        losses.append(loss.detach().item())

        embeddings.append((ID, graph_embeeding.tolist(), prediction.tolist(), M.tolist()))

        
    with open(os.path.join(args.output_dir, prefix + '.log'), 'w') as f:
        for arg in vars(args):
            f.write(str(arg) + ': ' + str(getattr(args, arg)) + '\n')
        f.write('\n\n')
        mse = torch.mean(torch.tensor(losses))
        rmse = mse ** 0.5
        scaled_rmse = (rmse * (_max - _min)) + _min
        f.write('MSE: {}\n'.format(mse))
        f.write('RMSE {:.4f}\n'.format(rmse))
        f.write('Scaled RMSE {:.4f}'.format(scaled_rmse))
        
    logger.info('MSE: {}\n'.format(mse))
    logger.info('RMSE {:.4f}\n'.format(rmse))
    logger.info('Scaled RMSE {:.4f}'.format(scaled_rmse))

    with open(os.path.join(args.output_dir, prefix + '-embeddings.csv'), 'w') as f:
        for e in embeddings:
            ID, graph_embeddings, prediction, actual = e
            for i in range(len(ID)):
                line = str(ID[i]) + ','
                line += ';'.join(list(map(str, graph_embeddings[i]))) + ','
                line += str(prediction[i][0]) + ','
                line += str(actual[i][0]) + '\n'
                f.write(line)
