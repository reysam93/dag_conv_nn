import torch.nn as nn
import torch.nn.functional as F
import torch

from src.baselines_archs import MLP

###############################   LAYERS   ###############################
class DAGConvLayer(nn.Module):
    """
    Implementation of a convolutional layer for DAGs.

    This layer applies convolutional operations on DAGs using Graph Shift Operators (GSOs)
    that capture the particular structure and properties of DAGs.

    Args:
        in_dim (int): Dimensionality of the input features.
        out_dim (int): Dimensionality of the output features.
        K (int): Number of Graph Shift Operators (GSOs) used.
        bias (bool): Indicates whether to include bias in the convolutional layer.

    Attributes:
        h (torch.nn.Parameter): Filter coefficient tensor of size (K,) for each GSO.
        W (torch.nn.Parameter): Weight matrix of size (in_dim, out_dim).
        b (torch.nn.Parameter or None): Bias vector of size (out_dim) if bias is True, otherwise None.

    Methods:
        forward(X, GSOs): Performs forward pass through the DAG convolutional layer.

    """
    def __init__(self, in_dim, out_dim, K, bias):
        super().__init__()

        # Store parameters
        self.K = K
        self.in_d = in_dim
        self.out_d = out_dim
        
        # Initialize learnable parameters
        self._init_parameters(bias)

    def _init_parameters(self, bias):
        """
        Initializes learnable parameters (filter coefficients, weights and bias) of the
        convolutional layer.

        Args:
            bias (bool): Indicates whether to consider bias in the convolution operation.
        """
        # Initialize filter parameter tensor
        self.h = nn.Parameter(torch.sqrt(torch.ones(self.K)/self.K))

        # Initialize weight parameter tensor
        self.W = nn.Parameter(torch.empty((self.in_d, self.out_d)))
        nn.init.xavier_uniform_(self.W)

        # Initialize bias parameter tensor if bias is True
        if bias:
            self.b = nn.Parameter(torch.empty(self.out_d))
            nn.init.constant_(self.b.data, 0.)
        else:
            self.b = None

    def forward(self, X, GSOs):
        """
        Performs forward pass through the DAG convolutional layer.

        Args:
            X (torch.Tensor): Input tensor of size (N, in_dim).
            GSOs (torch.Tensor): Graph Shift Operators tensor of size (K, N, N).

        Returns:
            torch.Tensor: Output tensor after convolution of size (N, out_dim).
        """
        XW = X @ self.W
        if len(X.shape) == 3:  # XW.shape: M x N x F_out
            H_exp = (self.h.view((self.K, 1, 1)) * GSOs).unsqueeze(1)  # Shape: (K, 1, N, N)
            X_out = H_exp @ XW                                         # Shape: (K, M, N, F_out)
            X_out = torch.sum(X_out, dim=0)                            # Shape: (M, N, F_out)

        else:  # XW.shape: N x F_out
            X_out = self.h.view((self.K, 1, 1)) * GSOs @ XW  # Shape: (K, N, F_out)
            X_out = torch.sum(X_out, dim=0)                  # Shape: (N, F_out)

        # Add bias if available
        if self.b is not None:
            return X_out + self.b[None,:]
        else:
            return X_out


class SF_DAGConvLayer(nn.Module):
    """
    Implementation of a convolutional layer for DAGs based on a single learnable filter.

    """
    def __init__(self, out_dim, K, bias):
        super().__init__()
        self.K = K
        self.out_d = out_dim
        self._init_parameters(bias)

    def _init_parameters(self, bias):
        """
        Initializes learnable parameters (filter coefficients, weights and bias) of the
        convolutional layer.
        """
        # Initialize filter parameter tensor
        self.h = nn.Parameter(torch.ones(self.K))

        # Initialize bias parameter tensor if bias is True
        if bias:
            self.b = nn.Parameter(torch.empty(self.out_d))
            nn.init.constant_(self.b.data, 0.)
        else:
            self.b = None

    def forward(self, X, GSOs):
        """
        Performs forward pass through the DAG convolutional layer.

        Args:
            X (torch.Tensor): Input tensor of size (N, in_dim).
            GSOs (torch.Tensor): Graph Shift Operators tensor of size (K, N, N).

        Returns:
            torch.Tensor: Output tensor after convolution of size (N, out_dim).
        """
        if len(X.shape) == 3:  # XW.shape: M x N x 1
            # Expand GSOs to match the shape of X:
            H_exp = (self.h.view((self.K, 1, 1)) * GSOs).unsqueeze(1)  # Shape: (K, 1, N, N)
            X_out = H_exp @ X                                          # Shape: (K, M, N, F_out)
            X_out = torch.sum(X_out, dim=0)                            # Shape: (M, N, F_out)

        else:  # XW.shape: N x F_out
            X_out = GSOs @ X                # Shape: (K, N, F_out)
            X_out = torch.sum(X_out, dim=0)  # Shape: (N, F_out)

        # Add bias if available
        if self.b is not None:
            return X_out + self.b[None,:]
        else:
            return X_out
        

class FB_DAGConvLayer(nn.Module):
    """
    Implementation of a convolutional layer for DAGs using a Filter Bank.

    This layer applies convolutional operations on DAGs using Graph Shift Operators (GSOs)
    that capture the particular structure and properties of DAGs.
    """
    def __init__(self, in_dim, out_dim, K, bias):
        super().__init__()

        # Store parameters
        self.K = K
        self.in_d = in_dim
        self.out_d = out_dim
        
        # Initialize learnable parameters
        self._init_parameters(bias)

    def _init_parameters(self, bias):
        """
        Initializes learnable parameters (filter coefficients, weights and bias) of the
        convolutional layer.

        Args:
            bias (bool): Indicates whether to consider bias in the convolution operation.
        """
        # Initialize weight parameter tensor
        self.W = nn.Parameter(torch.empty((self.K, self.in_d, self.out_d)))
        nn.init.xavier_uniform_(self.W)

        # Initialize bias parameter tensor if bias is True
        if bias:
            self.b = nn.Parameter(torch.empty(self.out_d))
            nn.init.constant_(self.b.data, 0.)
        else:
            self.b = None

    def forward(self, X, GSOs):
        """
        Performs forward pass through the DAG convolutional layer.

        Args:
            X (torch.Tensor): Input tensor of size (N, in_dim).
            GSOs (torch.Tensor): Graph Shift Operators tensor of size (K, N, N).

        Returns:
            torch.Tensor: Output tensor after convolution of size (N, out_dim).
        """
        assert len(X.shape) == 3 or len(X.shape) == 2, 'Input shape must be (M, N, F_in) or (N, F_in)'

        F_in = self.W.shape[1]
        F_out = self.W.shape[2]
        K = self.W.shape[0]
        if len(X.shape) == 3:  # XW.shape: M x N x 1
            M = X.shape[0]
            N = X.shape[1]
            
            # Shape of X after reshaping: (N, F_in*M)
            X_out = GSOs @ X.permute(1, 0, 2).contiguous().view(N, -1)  # Shape: (K, N, F_in*M)
            # Recover original shape (K, M, N, F_in) and adapt for batch multiplication
            X_out = X_out.view(K, N, M, F_in).permute(0, 2, 1, 3).reshape(K, N*M, F_in)
            # Shape after the multiplicaiton: (K, N*M, Fout)
            X_out = torch.bmm(X_out, self.W).sum(dim=0).reshape(M, N, F_out)
        else:
            # GSO: KxNxN,  X: NxFin, W: FinxFout --> X_out: NxFout
            X_out = (GSOs @ X @ self.W).sum(dim=0)

        # Add bias if available
        if self.b is not None:
            return X_out + self.b[None,:]
        else:
            return X_out
        

class ADCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, filter_coefs, hid_dim_MLP, layers_MLP, bias, act):
        super().__init__()
        assert torch.is_tensor(filter_coefs), 'Filter coefficients must be a Tensor'
        assert len(filter_coefs.shape) == 1, 'Filter coefficients must be 1D Tensor'

        self.h = filter_coefs
        self.MLP = MLP(in_dim, hid_dim_MLP, out_dim, layers_MLP, bias=bias,
                       act=act, l_act=None)

    def forward(self, X, GSOs):
        """
        Note: the ADCNLayer does not apply a non-linearity at the output of the MLP (l_act is always 
        set to None). Applying the no-linearity is responsability of the architecture using this layer.
        """
        K = GSOs.shape[0]

        assert len(X.shape) == 3 or len(X.shape) == 2, 'Input shape must be (M, N, F_in) or (N, F_in)'
        assert self.h.shape[0] == 1 or self.h.shape[0] == K, \
            "Number of GSOs s different than the number of filter coefficients"
    
        H = self.h * GSOs if self.h.shape[0] == 1 else self.h.view(K, 1, 1) * GSOs
        H = H.sum(dim=0)

        if len(X.shape) == 3:  # XW.shape: M x N x 1
            M, N, F_in = X.shape
            X_out = H @ X.permute(1, 0, 2).contiguous().view(N, -1)     # Shape: (N, F_in*M)
            X_out = X_out.view(N, M, F_in).permute(1, 0, 2)  # Shape: (M, N, F_in)
        else:
            X_out = H @ X
        return self.MLP(X_out)

#########################################################################
        

###########################   ARCHITECTURES   ###########################
class DAGConv(nn.Module):
    """
    Implementation of a DAG Convolutional Neural Network architecture.
    The model performs convolutional operations via matrix multiplication by GSOs that
    are tailored to DAGs.

    Args:
        in_dim (int): Dimensionality of the input features.
        hid_dim (int): Dimensionality of the hidden features.
        out_dim (int): Dimensionality of the output features.
        n_layers (int): Number of convolutional layers.
        GSOs (torch.Tensor): Graph Shift Operators tensor of shape (K, N, N), where K is the number of GSOs and N is the number of nodes.
        bias (bool, optional): Indicates whether to include bias in the convolutional layers. Defaults to True.
        act (function, optional): Activation function to apply after each convolutional layer. Defaults to torch.nn.functional.relu.
        l_act (function, optional): Activation function to apply after the last convolutional layer. Defaults to None.

    Returns:
        torch.Tensor: Output tensor after passing through DAGConv layers.
    """
    def __init__(self, in_dim: int, hid_dim: int, out_dim: int, K: int, n_layers: int,
                 bias: bool = True, act = F.relu,
                 l_act = None):
        super(DAGConv, self).__init__()

        self.in_d = in_dim
        self.hid_d = hid_dim
        self.out_d = out_dim
        self.act = act
        self.l_act =  l_act
        self.n_layers = n_layers
        self.bias = bias

        self.convs = self._create_conv_layers(n_layers, K, bias)
        self.n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _create_conv_layers(self, n_layers: int, K: int, bias: bool) -> nn.ModuleList:
        """
        Create convolutional layers for DAGs based on the provided parameters.

        Args:
            n_layers (int): Number of layers.
            K (int): Number of Graph Shift Operators (GSOs) used.
            bias (bool): Indicates whether to consider bias in the convolution operation.

        Returns:
            nn.ModuleList: List of DAGConvLayer instances.
        """
        convs = nn.ModuleList()
        
        if n_layers > 1:
            convs.append(DAGConvLayer(self.in_d, self.hid_d, K, bias))
            for _ in range(n_layers - 2):
                convs.append(DAGConvLayer(self.hid_d, self.hid_d, K, bias))
            convs.append(DAGConvLayer(self.hid_d, self.out_d, K, bias))
        else:
            convs.append(DAGConvLayer(self.in_d, self.out_d, K, bias))

        return convs


    def forward(self, X, GSOs):
        """
        Forward pass through the DAGConv layers.

        Args:
            X (torch.Tensor): Input tensor of size (N, in_dim).

        Returns:
            torch.Tensor: Output tensor after passing through DAGConv layers.
        """
        assert len(GSOs.shape) == 3 and GSOs.shape[1] == GSOs.shape[2], \
            "GSOs tensor must be of shape (K, N, N)"

        for _, conv in enumerate(self.convs[:-1]):
            X = self.act(conv(X, GSOs))

        X_out = self.convs[-1](X, GSOs)
        return self.l_act(X_out) if self.l_act else X_out
    

class SF_DAGConv(DAGConv):
    """
    Implementation of a DAG Convolutional Neural Network architecture using a single filter
    per layer. The model performs convolutional operations via matrix multiplication by GSOs
    that are tailored to DAGs.
    """
    def __init__(self, in_dim: int, out_dim: int, K: int, n_layers: int,
                 bias: bool = True, act = F.relu, l_act = None):
        assert in_dim == out_dim, 'Input and output dimensions must be the same'

        super(SF_DAGConv, self).__init__(in_dim, in_dim, out_dim, K, n_layers, bias, act,
                                         l_act)
        

    def _create_conv_layers(self, n_layers: int, K: int, bias: bool) -> nn.ModuleList:
        """
        Create convolutional layers for DAGs based on the provided parameters.
        """
        convs = nn.ModuleList()
        
        if n_layers > 1:
            convs.append(SF_DAGConvLayer(self.out_d, K, bias))
            for _ in range(n_layers - 2):
                convs.append(SF_DAGConvLayer(self.out_d, K, bias))
            convs.append(SF_DAGConvLayer(self.out_d, K, bias))
        else:
            convs.append(SF_DAGConvLayer(self.out_d, K, bias))

        return convs
    
class FB_DAGConv(DAGConv):
    """
    Implementation of a DAG Convolutional Neural Network architecture using a bank of filters
    per layer. The model performs convolutional operations via matrix multiplication by GSOs
    that are tailored to DAGs.
    """        

    def _create_conv_layers(self, n_layers: int, K: int, bias: bool) -> nn.ModuleList:
        """
        Create convolutional layers for DAGs based on the provided parameters.
        """
        convs = nn.ModuleList()
        
        if n_layers > 1:
            convs.append(FB_DAGConvLayer(self.in_d, self.hid_d, K, bias))
            for _ in range(n_layers - 2):
                convs.append(FB_DAGConvLayer(self.hid_d, self.hid_d, K, bias))
            convs.append(FB_DAGConvLayer(self.hid_d, self.out_d, K, bias))
        else:
            convs.append(FB_DAGConvLayer(self.in_d, self.out_d, K, bias))

        return convs
    

class ADCN(DAGConv):
    def __init__(self, in_dim: int, hid_dim: int, out_dim: int, n_layers: int, mlp_layers: int  = 2,
                 filter_coefs = torch.ones(1), bias: bool = True, act = F.relu, l_act = None, mlp_act = F.relu):

        self.h = filter_coefs if torch.is_tensor(filter_coefs) else torch.Tensor(filter_coefs)
        self.mlp_layers = mlp_layers
        self.mlp_act = mlp_act

        super(ADCN, self).__init__(in_dim, hid_dim, out_dim, self.h.shape[0], n_layers, bias,
                                         act, l_act)

    def _create_conv_layers(self, n_layers: int, K: int, bias: bool) -> nn.ModuleList:
        """
        Create convolutional layers for DAGs based on the provided parameters.
        """
        convs = nn.ModuleList()
        
        if n_layers > 1:
            convs.append(ADCNLayer(self.in_d, self.hid_d, self.h, self.hid_d, self.mlp_layers,
                                     bias, self.mlp_act))
            for _ in range(n_layers - 2):
                convs.append(ADCNLayer(self.hid_d, self.hid_d, self.h, self.hid_d, self.mlp_layers,
                                       bias, self.mlp_act))
            convs.append(ADCNLayer(self.hid_d, self.out_d, self.h, self.hid_d, self.mlp_layers,
                                   bias, self.mlp_act))
        else:
            convs.append(ADCNLayer(self.in_d, self.out_d, self.h, self.hid_d, self.mlp_layers,
                                   bias, self.mlp_act))
        return convs
    
    def to(self, dev):
        for conv in self.convs:
            conv.h = conv.h.to(dev)
        return super().to(dev)



class FB_altDAGConv(DAGConv):
    """
    Implementation of a DAG Convolutional Neural Network architecture using a bank of filters
    per layer. The model performs convolutional operations via matrix multiplication by GSOs
    that are tailored to DAGs.
    """        

    def _create_conv_layers(self, n_layers: int, K: int, bias: bool) -> nn.ModuleList:
        """
        Create convolutional layers for DAGs based on the provided parameters.
        """
        convs = nn.ModuleList()
        
        if n_layers > 1:
            convs.append(FB_AltenrnateConvLayer(self.in_d, self.hid_d, K, bias))
            for _ in range(n_layers - 2):
                convs.append(FB_AltenrnateConvLayer(self.hid_d, self.hid_d, K, bias))
            convs.append(FB_AltenrnateConvLayer(self.hid_d, self.out_d, K, bias))
        else:
            convs.append(FB_AltenrnateConvLayer(self.in_d, self.out_d, K, bias))

        return convs

