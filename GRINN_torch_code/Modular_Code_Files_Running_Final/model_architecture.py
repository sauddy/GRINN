import numpy as np

import torch
import torch.nn as nn
#from torch.autograd import Variable

class Sin(nn.Module):
    def forward(self, input):
        return torch.sin(input)

class ParameterEmbedding(nn.Module):
    def __init__(self):
        super(ParameterEmbedding, self).__init__()

        self.fc_in = nn.Linear(1, 32)
        self.fc_out = nn.Linear(32, 32)
        self.act = nn.Tanh()

    def forward(self, alpha):
        # Ensure alpha is [batch_size, 1]
        if alpha.dim() == 1:
            alpha = alpha.unsqueeze(-1)
        elif alpha.dim() > 2:
            alpha = alpha.view(-1, 1)
            
        # Process through network
        x = self.act(self.fc_in(alpha))
        x = self.fc_out(x)
        return x

class InputEmbedding(nn.Module):
    def __init__(self):
        super(InputEmbedding, self).__init__()

        self.fc_in_1d = nn.Linear(2, 64)
        self.fc_in_2d = nn.Linear(3, 64)
        self.fc_in_3d = nn.Linear(4, 64)
        self.fc_hidden1 = nn.Linear(64, 64)
        self.fc_hidden2 = nn.Linear(64, 64)
        self.fc_out = nn.Linear(64, 64)

        self.act = Sin()

    def forward(self, coord):
        # Check the number of features (columns) in the input tensor
        if coord.shape[1] == 2:
            x = self.fc_in_1d(coord)
        elif coord.shape[1] == 3:
            x = self.fc_in_2d(coord)
        elif coord.shape[1] == 4:
            x = self.fc_in_3d(coord)
        else:
            raise ValueError(f"Expected coord to have 2, 3, or 4 features, but got {coord.shape[1]} features with shape {coord.shape}")
            
        x = self.act(self.fc_hidden1(x))
        x = self.act(self.fc_hidden2(x))
        x = self.fc_out(x)

        return x

class PINN(nn.Module):
    def __init__(self, num_neurons=96):
        super(PINN, self).__init__()

        self.alpha_embedder = ParameterEmbedding()
        self.input_embedder = InputEmbedding()

        self.fc_in = nn.Linear(32 + 64, num_neurons)
        self.neurons1 = nn.Linear(num_neurons, num_neurons)
        self.neurons2 = nn.Linear(num_neurons, num_neurons)
        self.neurons3 = nn.Linear(num_neurons, num_neurons)
        self.neurons4 = nn.Linear(num_neurons, num_neurons)
        self.neurons5 = nn.Linear(num_neurons, num_neurons)

        self.fc_out_1d = nn.Linear(num_neurons, 3)
        self.fc_out_2d = nn.Linear(num_neurons, 4)
        self.fc_out_3d = nn.Linear(num_neurons, 5)

        self.sin_act = Sin()

    def forward(self, X):
        # Extract and ensure correct dimensions
        if len(X) == 3:
            x, t, alpha = X[0], X[1], X[2]
            y, z = None, None
        elif len(X) == 4:
            x, y, t, alpha = X[0], X[1], X[2], X[3]
            z = None
        elif len(X) == 5:
            x, y, z, t, alpha = X[0], X[1], X[2], X[3], X[4]
        else:
            raise ValueError(f"Expected len(X) to be 3, 4 or 5 but got {len(X)}")
        
        # Ensure all inputs are [batch_size, 1]
        x = x if x.dim() == 2 else x.unsqueeze(-1)
        t = t if t.dim() == 2 else t.unsqueeze(-1)
        if y is not None:
            y = y if y.dim() == 2 else y.unsqueeze(-1)
        if z is not None:
            z = z if z.dim() == 2 else z.unsqueeze(-1)
        alpha = alpha if alpha.dim() == 2 else alpha.unsqueeze(-1)
        
        # Get batch size from x
        batch_size = x.size(0)
        
        # Process input coordinates
        if len(X) == 3:
            input_coord = torch.cat([x, t], dim=1)
            input_emb = self.input_embedder(input_coord)
            alpha_emb = self.alpha_embedder(alpha)

        elif len(X) == 4:
            input_coord = torch.cat([x, y, t], dim=1)
            input_emb = self.input_embedder(input_coord)
            alpha_emb = self.alpha_embedder(alpha)

        elif len(X) == 5:
            input_coord = torch.cat([x, y, z, t], dim=1)
            input_emb = self.input_embedder(input_coord)
            alpha_emb = self.alpha_embedder(alpha)

        else:
            raise ValueError(f"Expected len(X) to be 3 but got {len(X)}")

        # Verify tensor sizes before concatenation
        assert input_emb.size(0) == batch_size, f"Input embedding batch size mismatch: {input_emb.size(0)} vs {batch_size}"
        assert alpha_emb.size(0) == batch_size, f"Alpha embedding batch size mismatch: {alpha_emb.size(0)} vs {batch_size}"
        
        # Concatenate embeddings
        inputs = torch.cat([input_emb, alpha_emb], dim=1)

        # Process through network
        h0 = self.sin_act(self.fc_in(inputs))
        h1 = self.sin_act(self.neurons1(h0))
        h2 = self.sin_act(self.neurons2(h1))
        h3 = self.sin_act(self.neurons3(h2))
        h4 = self.sin_act(self.neurons4(h3)) + h0
        h5 = self.sin_act(self.neurons5(h4))

        # Select appropriate output layer
        if len(X) == 3:
            output = self.fc_out_1d(h5)
        elif len(X) == 4:
            output = self.fc_out_2d(h5)
        elif len(X) == 5:
            output = self.fc_out_3d(h5)
        else:
            raise ValueError(f"Expected len(X) to be 3, 4 or 5 but got {len(X)}")

        return output
        
def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)