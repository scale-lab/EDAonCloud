import torch
from ictuner import get_logger
from ictuner import read_netlist
from model import Predictor
from dataset import NetlistDataset


def tune(lef_file, netlist_file, device=torch.device("cpu"), model_file='data/model.pth'):
    g = read_netlist(lef_file, netlist_file)

    model = Predictor(2, 256, 128, \
                      6, 64, 32, \
                      192, 1).to(device)
    model.load_state_dict(torch.load(model_file, map_location=device))
    model.eval()
    
    params = [[1.0, 0.5, 100, 0.7, 1, 7]]
    for i in range(1000):
        _ = model(g, torch.tensor(params).float().to(device))
        print(i)
        


device = torch.device("cuda:{}".format(0) if torch.cuda.is_available() else "cpu")
tune('/local-disk/tools/TSMC65LP/tsmc/merged.lef', '/home/aibrahim/ICCAD20/data/jpeg/netlists/jpeg_encoder.0.gl.v.def', model_file='results/drv/s4/100-0.0001-256-128-64-32-192-model.pth', device=device)