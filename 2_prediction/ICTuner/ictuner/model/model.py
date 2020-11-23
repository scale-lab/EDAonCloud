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

device = torch.device("cuda:{}".format(0) if torch.cuda.is_available() else "cpu")

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
    def __init__(self, netlist_in_dim, gcn_hidden_dim, netlist_embedding_size, num_params, params_hidden_dim, params_embedding_size, predictor_hidden_dim, num_metrics):
        super(Predictor, self).__init__()
        self.gcn = GCN(netlist_in_dim, gcn_hidden_dim, netlist_embedding_size)
        self.params_fc1 = nn.Linear(num_params, params_hidden_dim, bias=True)
        self.params_fc2 = nn.Linear(params_hidden_dim, params_embedding_size, bias=True)
        self.predictor_fc1 = nn.Linear(netlist_embedding_size + params_embedding_size, predictor_hidden_dim, bias=True)
        self.predictor_fc2 = nn.Linear(predictor_hidden_dim, num_metrics, bias=True)
    
    def forward(self, g, params):
        h = torch.cat((g.in_degrees().view(-1 ,1).float(), g.out_degrees().view(-1, 1).float()), 1).to(device)
        graph_embedding = self.gcn(g, h)
        
        params_embedding = F.relu(self.params_fc1(params))
        params_embedding = self.params_fc2(params_embedding)

        metrics = F.relu(self.predictor_fc1(torch.cat((graph_embedding, params_embedding), 1)))
        metrics = self.predictor_fc2(metrics)

        return graph_embedding, params_embedding, metrics
