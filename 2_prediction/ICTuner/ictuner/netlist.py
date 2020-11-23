import os
import networkx as nx
import dgl as dgl

from dgl.data.utils import save_graphs, load_graphs
from ictuner.log import get_logger

def _build_graph(design, undirected=False):
    '''
    Input: design is an OpenDB representation of the chip
    Returns: DGL graph
    '''
    instances = design.getBlock().getInsts()
    pins = design.getBlock().getBTerms()

    # initialize graph with number of nodes
    g = dgl.DGLGraph()
    g.add_nodes(len(instances) + len(pins))

    # DGL represents nodes in numbers only. So, we need to assign mapping
    nodes_mapping = {}
    i = 0
    for inst in instances:
        nodes_mapping[inst.getName()] = i
        i += 1
    for pin in pins:
        nodes_mapping[pin.getName()] = i
        i += 1

    nets = design.getBlock().getNets()
    for net in nets:
        # exclude power nets
        if net.isSpecial():
            continue

        iterms = net.getITerms()
        bterms = net.getBTerms()

        # given a net, figure out the driving cell and the loads
        driving_cell = None
        loads = []

        # if iterm, then it needs to have direction output to be a driving cell
        for iterm in iterms:
            if iterm.getIoType() == 'OUTPUT':
                driving_cell = iterm.getInst().getName()
            else:
                loads.append(iterm.getInst().getName())
        
        # if bterm, then it needs to have direction input to be a driving cell
        for bterm in bterms:
            if bterm.getIoType() == 'INPUT':
                assert (driving_cell == None), "Something is wrong with the directions!"
                driving_cell = bterm.getName()
            else:
                loads.append(bterm.getName())
        
        assert (driving_cell != None), "Couldn't figure out the net directions"

        # add edges
        src = nodes_mapping[driving_cell]
        dst = list(map(lambda name: nodes_mapping[name], loads))
        g.add_edges(src, dst)
        
        # add self-loop
        g.add_edge(src, src)

        # add opposite-direction edges if undirected
        if undirected:
            g.add_edges(dst, src)

    return g


def _save_g(file_path, g, labels=None):
    save_graphs(file_path, g, labels=labels)


def read_netlist(lef_file, netlist_file, load_cache=True):
    logger = get_logger()

    if os.path.exists(netlist_file + '.dgl') and load_cache:
        graphs, _ = load_graphs(netlist_file + '.dgl')
        logger.info("Loaded a cached graph")
        return graphs[0]

    # intialize the database
    import opendbpy as odb
    db = odb.dbDatabase.create()

    # load the lef file
    try:
        odb.read_lef(db, lef_file)
    except Exception as e:
        logger.error("Problem loading the tech file!")
        return None

    # load the def file
    try:
        odb.read_def(db, netlist_file)
    except Exception as e:
        logger.error("Problem loading the design!")
        return None
    
    # parse the design into a DGL graph
    design = db.getChip()
    G = _build_graph(design)

    logger.info('Built a graph with %s nodes', str(G.number_of_nodes()))
    logger.info('.... Added %s edges', str(G.number_of_edges()))

    _save_g(netlist_file + '.dgl', G)

    return G

