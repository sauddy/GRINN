import numpy as np

import torch
import torch.nn as nn
#from torch.autograd import Variable

class Sin(nn.Module):
    def forward(self, input):
        return torch.sin(input)

class PINN(nn.Module):
    def __init__(self, num_neurons=64):
        super(PINN, self).__init__()
        self.num_neurons = num_neurons
    
    # 1D branch (x, t)
        self.branch_1d = nn.Sequential(
            nn.Linear(2, num_neurons),
            Sin(),
            nn.Linear(num_neurons, num_neurons),
            Sin(),
            nn.Linear(num_neurons, num_neurons),
            Sin(),
            nn.Linear(num_neurons, num_neurons),
            Sin(),
            nn.Linear(num_neurons, 3))
        
    # 2D branch (x, y, t)
        self.branch_2d = nn.Sequential(
            nn.Linear(3, num_neurons),
            Sin(),
            nn.Linear(num_neurons, num_neurons),
            Sin(),
            nn.Linear(num_neurons, 4))
        
    # 3D branch (x, y, z, t)
        self.branch_3d = nn.Sequential(
            nn.Linear(4, num_neurons),
            Sin(),
            nn.Linear(num_neurons, num_neurons),
            Sin(),
            nn.Linear(num_neurons, 5))
        
        # Output layers per branch
        #self.output_layer_1d = nn.Linear(3, 1)
        #self.output_layer_2d = nn.Linear(4, 1)
        #self.output_layer_3d = nn.Linear(5, 1)


    def forward(self,X):
        x, t = X[0],X[-1]
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        t = t.unsqueeze(-1) if t.dim() == 1 else t

        if len(X) == 2:
            inputs = torch.cat([x, t], dim=1)
            return self.branch_1d(inputs)
        
        elif len(X) == 3:
            y = X[1]
            y = y.unsqueeze(-1) if y.dim() == 1 else y

            inputs = torch.cat([x, y, t], dim=1)
            return self.branch_2d(inputs)
        
        elif len(X) == 4:
            y = X[1]
            z = X[2]
            y = y.unsqueeze(-1) if y.dim() == 1 else y
            z = z.unsqueeze(-1) if z.dim() == 1 else z

            inputs = torch.cat([x, y, z, t], dim=1)
            return self.branch_3d(inputs)
        
        else:
            raise ValueError(f"Expected len(X) in [2, 3, 4] but got {len(X)}")
        
def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
