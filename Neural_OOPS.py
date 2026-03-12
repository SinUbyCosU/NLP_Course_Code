#Inheritance

import torch
import torch.nn as nn

class MyNeuralNetworks(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1=nn.Linear(in_features=10,out_features=5)
        self.layer2=nn.Linear(in_features=5, out_features=1)

model=MyNeuralNetworks()