import numpy as np

import torch
import torch.nn as nn
#from torch.autograd import Variable
from config import rho_o, num_neurons, num_layers, PERTURBATION_TYPE, DEFAULT_ACTIVATION

class Sin(nn.Module):
    def forward(self, input):
        return torch.sin(input)


def get_activation(activation_type):
    """
    Factory function to create activation function instances.
    
    Args:
        activation_type: String identifier ('sin', 'tanh', 'relu', 'elu')
    
    Returns:
        nn.Module activation function
    """
    activation_type = activation_type.lower()
    if activation_type == 'sin':
        return Sin()
    elif activation_type == 'tanh':
        return nn.Tanh()
    elif activation_type == 'relu':
        return nn.ReLU()
    elif activation_type == 'elu':
        return nn.ELU()
    else:
        raise ValueError(f"Unknown activation type: {activation_type}. Choose from 'sin', 'tanh', 'relu', 'elu'.")

class PINN(nn.Module):
    def __init__(self, num_neurons=num_neurons, num_layers=num_layers, n_harmonics=1, activation_type=DEFAULT_ACTIVATION):
        super(PINN, self).__init__()
        self.num_neurons = num_neurons
        self.n_harmonics = n_harmonics
        self.num_layers = max(2, int(num_layers))  # total Linear layers including output
        self.activation_type = activation_type
        
        # Domain extents for periodic embeddings (set via set_domain)
        self.xmin = None
        self.xmax = None
        self.ymin = None
        self.ymax = None
        self.zmin = None
        self.zmax = None
    
    # Helper to build a branch with dynamic depth
        def _make_branch(in_dim, out_dim):
            layers = []
            # First layer
            layers.append(nn.Linear(in_dim, self.num_neurons))
            # Hidden layers: total linear layers = self.num_layers; we already added 1; 
            # add (self.num_layers - 2) hidden Linear blocks with activations after each
            for _ in range(self.num_layers - 2):
                layers.append(get_activation(self.activation_type))
                layers.append(nn.Linear(self.num_neurons, self.num_neurons))
            # Activation before output if there is at least one hidden block
            if self.num_layers > 2:
                layers.append(get_activation(self.activation_type))
            # Output layer
            layers.append(nn.Linear(self.num_neurons, out_dim))
            return nn.Sequential(*layers)

    # 1D branch (periodic x features + t)
        in_dim_1d = 2*self.n_harmonics + 1
        self.branch_1d = _make_branch(in_dim_1d, 3)
        
    # 2D branch (periodic x,y features + t)
        in_dim_2d = 4*self.n_harmonics + 1
        self.branch_2d = _make_branch(in_dim_2d, 4)
        
    # 3D branch (periodic x,y,z features + t)
        in_dim_3d = 6*self.n_harmonics + 1
        self.branch_3d = _make_branch(in_dim_3d, 5)
        
        # Output layers per branch
        #self.output_layer_1d = nn.Linear(3, 1)
        #self.output_layer_2d = nn.Linear(4, 1)
        #self.output_layer_3d = nn.Linear(5, 1)


    def set_domain(self, rmin, rmax, dimension):
        # rmin/rmax exclude time; follow ASTPN usage
        if dimension >= 1:
            self.xmin, self.xmax = float(rmin[0]), float(rmax[0])
        if dimension >= 2:
            self.ymin, self.ymax = float(rmin[1]), float(rmax[1])
        if dimension >= 3:
            self.zmin, self.zmax = float(rmin[2]), float(rmax[2])

    def _periodic_features(self, u, umin, umax):
        # u is [N,1]
        L = umax - umin
        theta = 2*np.pi*(u - umin)/L
        features = []

        for k in range(1, self.n_harmonics+1):
            
            scale = 1.0 / np.sqrt(k)

            features.append(scale * torch.sin(k*theta))
            features.append(scale * torch.cos(k*theta))

        return torch.cat(features, dim=1) if len(features) > 0 else u

    def _prepare_coordinate_features(self, X):
        """
        Prepare periodic features for all spatial coordinates.
        
        Args:
            X: List of coordinates [x, ...spatial..., t]
        
        Returns:
            Tuple (features, t_tensor, dimension)
        """
        x, t = X[0], X[-1]
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        t = t.unsqueeze(-1) if t.dim() == 1 else t
        dimension = len(X)
        
        if dimension == 2:
            if self.xmin is None or self.xmax is None:
                raise RuntimeError("Domain not set: call net.set_domain for dimension=1")
            x_feat = self._periodic_features(x, self.xmin, self.xmax)
            features = torch.cat([x_feat, t], dim=1)
        
        elif dimension == 3:
            if self.xmin is None or self.xmax is None or self.ymin is None or self.ymax is None:
                raise RuntimeError("Domain not set: call net.set_domain for dimension=2")
            y = X[1].unsqueeze(-1) if X[1].dim() == 1 else X[1]
            x_feat = self._periodic_features(x, self.xmin, self.xmax)
            y_feat = self._periodic_features(y, self.ymin, self.ymax)
            features = torch.cat([x_feat, y_feat, t], dim=1)
        
        elif dimension == 4:
            if (self.xmin is None or self.xmax is None or
                self.ymin is None or self.ymax is None or
                self.zmin is None or self.zmax is None):
                raise RuntimeError("Domain not set: call net.set_domain for dimension=3")
            y = X[1].unsqueeze(-1) if X[1].dim() == 1 else X[1]
            z = X[2].unsqueeze(-1) if X[2].dim() == 1 else X[2]
            x_feat = self._periodic_features(x, self.xmin, self.xmax)
            y_feat = self._periodic_features(y, self.ymin, self.ymax)
            z_feat = self._periodic_features(z, self.zmin, self.zmax)
            features = torch.cat([x_feat, y_feat, z_feat, t], dim=1)
        
        else:
            raise ValueError(f"Expected len(X) in [2, 3, 4] but got {dimension}")
        
        return features, t, dimension
    
    def _apply_density_constraint(self, outputs, t):
        """
        Apply hard density constraint for power spectrum perturbations.
        
        For non-sinusoidal cases, enforce ρ(t=0) = ρ₀ using linear trick:
        ρ = ρ₀ + t × ρ̂, where network predicts ρ̂.
        
        Args:
            outputs: Raw network outputs
            t: Time tensor
        
        Returns:
            Modified outputs with density constraint applied
        """
        if str(PERTURBATION_TYPE).lower() == "sinusoidal":
            return outputs
        
        # Linear trick: ρ = ρ₀ + t × ρ̂
        rho_hat = outputs[:, 0:1]
        other = outputs[:, 1:]
        rho = rho_o + t * rho_hat
        return torch.cat([rho, other], dim=1)
    
    def forward(self, X):
        """
        Forward pass of PINN.
        
        Args:
            X: List of coordinates [x, ...spatial..., t]
        
        Returns:
            Network predictions [rho, vx, vy?, vz?, phi]
        """
        features, t, dimension = self._prepare_coordinate_features(X)
        
        # Select appropriate branch based on dimension
        if dimension == 2:
            outputs = self.branch_1d(features)
        elif dimension == 3:
            outputs = self.branch_2d(features)
        elif dimension == 4:
            outputs = self.branch_3d(features)
        else:
            raise ValueError(f"Unexpected dimension: {dimension}")
        
        return self._apply_density_constraint(outputs, t)
        
def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
