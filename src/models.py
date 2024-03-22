import torch
from torch.utils.data import TensorDataset, DataLoader

import numpy as np
from copy import deepcopy



class Model():
    """
    Model class for learning the weights of a neural network architecture in the context of regression problems.

    This class provides functionality to train and evaluate a neural network model using the provided data.

    Args:
        arch (torch.nn.Module): Neural network architecture to be trained.
        loss (torch.nn.modules.loss._Loss, optional): Loss function for training the model. Defaults to torch.nn.CrossEntropyLoss(reduction='sum').
        device (str or torch.device, optional): Device to run the model on ('cpu' or 'cuda'). Defaults to 'cpu'.

    Attributes:
        arch (torch.nn.Module): Architecture to be trained.
        loss_fn (torch.nn.modules.loss._Loss): Loss function for training the model.
        dev (str or torch.device): Device on which the model is running.

    Methods:
        _get_loss(X, Y): Computes the loss for a given input-output pair.
        fit(X, Y, lr, n_epochs, bs=100, wd=0, optim=torch.optim.Adam, eval_freq=10, patience=100, verb=False): Trains the model using the provided data.

    """
    def __init__(self, arch, loss=torch.nn.MSELoss(reduction='sum'),
                 device='cpu'):
        self.arch = arch.to(device)
        self.loss_fn = loss
        self.dev = device
    
    def _get_loss(self, X, Y):
        self.arch.eval()
        with torch.no_grad():
            Y_hat = self.arch(X)
            return self.loss_fn(Y_hat, Y).item()

    def fit(self, X, Y, lr, n_epochs, bs=100, wd=0, optim=torch.optim.Adam, eval_freq=10,
              patience=100, verb=False):
        """
        Trains the model using the provided data.

        Args:
            X (dict): Dictionary containing input data. It should contain keys 'train', 'val', and optionally 'test'.
            Y (dict): Dictionary containing target data corresponding to input data. It should contain keys 'train', 'val', and optionally 'test'.
            lr (float): Learning rate for optimization.
            n_epochs (int): Number of epochs for training.
            bs (int, optional): Batch size for training. Defaults to 100.
            wd (float, optional): Weight decay (L2 regularization) parameter. Defaults to 0.
            optim (torch.optim.Optimizer, optional): PyTorch optimizer class. Defaults to torch.optim.Adam.
            eval_freq (int, optional): Frequency of evaluation (in epochs). Defaults to 20.
            patience (int, optional): Number of epochs to wait for improvement in validation loss before early stopping. Defaults to 100.
            verb (bool, optional): Verbosity flag to print progress. Defaults to False.

        Returns:
            dict: Dictionary containing losses for training, validation, and test data (if available).
        """
        # Checks input
        assert X.keys() == Y.keys(), 'X and Y contain different keys' 
        assert 'val' in X, 'Missing validation data'

        # Extract data
        train_ds = TensorDataset(X['train'].to(self.dev), Y['train'].to(self.dev))
        train_dl = DataLoader(train_ds, batch_size=bs)
        X_val, Y_val = X['val'].to(self.dev), Y['val'].to(self.dev)
        X_test, Y_test = X['test'].to(self.dev), Y['test'].to(self.dev) if 'test' in X else None

        # Initialize variables
        best_val_loss = float('inf')
        cont_stop = 0
        best_weights = deepcopy(self.arch.state_dict())
        opt = optim(self.arch.parameters(), lr=lr, weight_decay=wd)
        losses_train, losses_val, losses_test = [np.zeros(n_epochs) for _ in range(3)]

        # Training loop
        for epoch in range(n_epochs):
            self.arch.train()

            for Xb, Yb in train_dl:
                Y_hat = self.arch(Xb)
                loss_train = self.loss_fn(Y_hat, Yb)

                loss_train.backward()
                opt.step()
                opt.zero_grad()

            # losses_train[i] = loss_train.detach().cpu().item()
            losses_train[epoch] = loss_train.item()
            losses_val[epoch] = self._get_loss(X_val, Y_val)
            
            if X_test is not None:
                losses_test[epoch] = self._get_loss(X_test, Y_test)

            if (epoch == 0 or (epoch+1) % eval_freq == 0) and verb:
                print(f"Epoch {epoch+1}/{n_epochs} - Loss Train: {losses_train[epoch]:.6f} - Val Loss: {losses_val[epoch]:.6f}", flush=True)

            # Early stopping based on validation loss
            if losses_val[epoch] < best_val_loss:
                best_val_loss = losses_val[epoch]
                best_weights = deepcopy(self.arch.state_dict())
                cont_stop = 0
            else:
                cont_stop += 1

            if cont_stop >= patience:
                if verb:
                    print(f'\tConvergence achieved at iteration: {epoch}')
                break

        self.arch.load_state_dict(best_weights)
        return {'train': losses_train, 'val': losses_val, 'test': losses_test}

    def test(self, X, Y):
        X = X.to(self.dev)
        Y = Y.to(self.dev)

        Y_hat = self.arch(X).cpu().detach().numpy().squeeze().T
        Y = Y.cpu().detach().numpy().squeeze().T
        norm_Y = np.linalg.norm(Y, axis=0)
        err = (np.linalg.norm(Y_hat - Y, axis=0)/norm_Y)**2
        return err.mean(), err.std()