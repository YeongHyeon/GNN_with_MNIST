def connect(nn):

    if(nn == 0): import neuralnet.net00_gcn as nn
    elif(nn == 1): import neuralnet.net01_gat as nn

    return nn
