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
from torch.utils.data import Dataset

from ictuner import read_netlist
from ictuner import get_logger


def gcn_message(edges):
    return {
        'msg': edges.src['h']
    }

def gcn_reduce(nodes):
    # Theoritically, the sum operation is better than the average
    # Or use attention (GAT)
    return {
        'h': torch.sum(nodes.mailbox['msg'], dim=1)
    }


class GraphConvolutionLayer(nn.Module):
    # The depth of this layer is irrelevant of the depth in the GCN module.
    # This is the depth of the NN that does the aggregation. 
    # Below in GCN module is the depth of the GCN, which goes deep in the graph itself.
    def __init__(self, in_features, out_features):
        super(GraphConvolutionLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
    
    def forward(self, g, inputs):
        g.ndata['h'] = inputs
        g.update_all(gcn_message, gcn_reduce)
        h = g.ndata.pop('h')

        return self.linear(h)


class GCN(nn.Module):
    def __init__(self, in_features, hidden_size, embedding_size):
        super(GCN, self).__init__()
        self.gcn1 = GraphConvolutionLayer(in_features, hidden_size)
        self.gcn2 = GraphConvolutionLayer(hidden_size, embedding_size)
    
    def forward(self, g, inputs):
        h = F.relu(self.gcn1(g, inputs))
        h = self.gcn2(g, h)
        
        # report graph state vector
        g.ndata['h'] = F.relu(h)
        graph_embedding = dgl.sum_nodes(g, 'h')
        g.ndata.pop('h')

        return graph_embedding


class Predictor(nn.Module):
    def __init__(self, netlist_in_dim, gcn_hidden_dim, netlist_embedding_size, predictor_hidden_dim, num_metrics):
        super(Predictor, self).__init__()
        self.gcn = GCN(netlist_in_dim, gcn_hidden_dim, netlist_embedding_size)
        self.predictor_fc1 = nn.Linear(netlist_embedding_size, predictor_hidden_dim, bias=True)
        self.predictor_fc2 = nn.Linear(predictor_hidden_dim, num_metrics, bias=True)
    
    def forward(self, g):
        h = torch.cat((g.in_degrees().view(-1 ,1).float(), g.out_degrees().view(-1, 1).float()), 1).to(device)
        graph_embedding = self.gcn(g, h)

        metrics = F.relu(self.predictor_fc1(graph_embedding)
        metrics = self.predictor_fc2(metrics)

        return graph_embedding, metrics

class NetlistDataset(Dataset):
    def __init__(self, Gs, dataset_file):
        self.Gs = Gs
        self.flows_df = pd.read_csv(dataset_file)

    def __len__(self):
        return len(self.flows_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        g_id = self.flows_df.iloc[idx, 0].split('/')[-1]
        g = self.Gs[g_id]

        metrics = self.flows_df.loc[idx, 1:5]        # runtime
        metrics = np.array([metrics]).astype('float')

        sample = {
            'id': g_id,
            'design': g,
            'metrics': torch.tensor(metrics).float()
        }

        return sample

# setup device
device = torch.device("cuda:{}".format(0) if torch.cuda.is_available() else "cpu")

def collate(samples):
    IDs, Gs, Ps, Ms = [], [], [], []
    for s in samples:
        IDs.append(s['id'])
        Gs.append(s['design'])
        Ms.append(s['metrics'].view(-1, 1))
    return IDs, dgl.batch(Gs).to(device), torch.cat(Ps, dim=0), torch.cat(Ms, dim=0)

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
    parser.add_argument("-params_hidden_dim", type=int, required=False, default=64, \
        help="Number of hidden units of parameters FC")
    parser.add_argument("-params_embedding_size", type=int, required=False, default=32, \
        help="Embedding size of the parameters")
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
    dgl_graphs = '/home/aibrahim/ICCAD20/data/dgl'
    dgl_files = [f for f in os.listdir(dgl_graphs) if os.path.isfile(os.path.join(dgl_graphs, f)) and f.endswith(".dgl")]
    Gs = {}
    for dgl_file in dgl_files:
        design = '.'.join(dgl_file.split('.')[:-1])
        g = read_netlist('/local-disk/tools/TSMC65LP/tsmc/merged.lef', os.path.join(dgl_graphs, design))
        design = '.'.join(dgl_file.split('.')[:-2])
        Gs[design] = g

    # load dataset
    train_dataset = NetlistDataset(Gs, '/home/aibrahim/ICCAD20/db-success-train.csv')
    test_dataset = NetlistDataset(Gs, '/home/aibrahim/ICCAD20/db-success-test.csv')
    
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
