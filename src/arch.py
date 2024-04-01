import torch.nn as nn
import torch.nn.functional as F
import torch


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
        assert len(X.shape) == 3, 'Input shape must be (M, N, F_in)'
        # XW = X @ self.W        # Shape: (K, N, F_out )
        # Expand GSOs and W to match the shape of X:
        XW = X @ self.W.unsqueeze(1)     # Shape: (K, M, N, F_out)
        GSOs_exp = GSOs.unsqueeze(1)     # Shape: (K, 1, N, N)
        X_out = GSOs_exp @ XW            # Shape: (K, M, N, F_out)
        X_out = torch.sum(X_out, dim=0)  # Shape: (M, N, F_out)

        # Add bias if available
        if self.b is not None:
            return X_out + self.b[None,:]
        else:
            return X_out
##########################################################################
        

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
        last_act (function, optional): Activation function to apply after the last convolutional layer. Defaults to None.

    Returns:
        torch.Tensor: Output tensor after passing through DAGConv layers.
    """
    def __init__(self, in_dim: int, hid_dim: int, out_dim: int, K: int, n_layers: int,
                 bias: bool = True, act = F.relu,
                 last_act = None):
        super(DAGConv, self).__init__()

        self.in_d = in_dim
        self.hid_d = hid_dim
        self.out_d = out_dim
        self.act = act
        self.l_act =  last_act
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
                 bias: bool = True, act = F.relu, last_act = None):
        assert in_dim == out_dim, 'Input and output dimensions must be the same'

        super(SF_DAGConv, self).__init__(in_dim, in_dim, out_dim, K, n_layers, bias, act,
                                         last_act)
        

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