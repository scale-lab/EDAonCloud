import matplotlib.pyplot as plt
from dgl.data.utils import load_graphs
import numpy as np
from sklearn import linear_model

plt.rcParams.update({'font.size': 14})

def design_size(design_file):
    g, _ = load_graphs('data/dgl/' + design_file + '.def.dgl')
    return g[0].num_nodes(), g[0].num_edges()

def analyze():
    train_file = 'data/train.csv'
    test_file = 'data/test.csv'

    num_nodes = []
    num_edges = []
    runtimes = []

    with open(train_file, 'r') as f:
        f.readline() # to skip header

        for line in f:
            design_file, runtime = line.strip().split(',')
            nodes, edges = design_size(design_file)
            num_nodes.append(nodes)
            num_edges.append(edges)
            runtimes.append(float(runtime))

    with open(test_file, 'r') as f:
        f.readline() # to skip header

        for line in f:
            design_file, runtime = line.strip().split(',')
            nodes, edges = design_size(design_file)
            num_nodes.append(nodes)
            num_edges.append(edges)
            runtimes.append(float(runtime))
    
    s = [x for x in zip(num_nodes, runtimes) if x[0] >= 5000 and x[0] <= 20000]
    x, y = list(zip(*s))
    plt.scatter(x, y)
    plt.xlabel('Design Size (# cells)')
    plt.ylabel('Runtime (seconds)')
    plt.tight_layout()
    plt.show()
            
def train():
    train_file = 'data/train.csv'
    test_file = 'data/test.csv'

    num_nodes = []
    num_edges = []
    runtimes = []

    with open(train_file, 'r') as f:
        f.readline() # to skip header

        for line in f:
            design_file, runtime = line.strip().split(',')
            nodes, edges = design_size(design_file)
            num_nodes.append(nodes)
            num_edges.append(edges)
            runtimes.append(float(runtime))


    regr = linear_model.LinearRegression()
    regr.fit(np.array(num_nodes).reshape(-1, 1), np.array(runtimes).reshape(-1, 1))

    num_nodes = []
    num_edges = []
    runtimes = []

    with open(test_file, 'r') as f:
        f.readline() # to skip header

        for line in f:
            design_file, runtime = line.strip().split(',')
            nodes, edges = design_size(design_file)
            num_nodes.append(nodes)
            num_edges.append(edges)
            runtimes.append(float(runtime))
    
    
    pred = regr.predict(np.array(num_nodes).reshape(-1, 1))
    
    # calculate avg error
    errors = []
    for i in range(len(pred)):
        e = abs(pred[i].item() - runtimes[i]) / min(pred[i].item(), runtimes[i])
        errors.append(e)
    
    print(sum(errors) / len(errors))


if __name__ == '__main__':
    analyze()
    train()
