import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np
import cupy as cp

# collect all models in a dictionary for easy configuration
models = dict()

class MLP(chainer.Chain):
    '''
    Multi-Layer Perceptron with variable number of layers
    and hidden units.

    Parameters
    ----------
    layers: list of int
        The number of hidden units per layer
    n_out: int
        The number of outputs of the network
    '''

    def __init__(self, layers, n_out):
        super(MLP, self).__init__()
        with self.init_scope():

            layers = [None] + layers + [n_out]

            # All the hidden layers
            self.weights = chainer.ChainList(
                    *[ L.Linear(l_in, l_out) for l_in, l_out in zip(layers[:-1],layers[1:]) ]
                    )

    def __call__(self, x):
        h = x
        for W in self.weights[:-1]:
            h = W(h)
            h = F.relu(h)
        return self.weights[-1](h)

models['MLP'] = MLP


class ResBlock(chainer.Chain):
    '''
    A two layers residual block

    Parameters
    ----------
    n: int
        the width of the block
    '''

    def __init__(self, n, n_hidden):
        super(ResBlock, self).__init__()
        with self.init_scope():

            self.hidden1 = L.Linear(n, n_hidden)
            self.hidden2 = L.Linear(n_hidden, n)

    def __call__(self, x):
        h = x
        h = F.relu(self.hidden1(h))
        h = self.hidden2(h)

        return x + h

models['ResBlock'] = ResBlock


class ResReg(chainer.Chain):

    def __init__(self, n_res, n_res_in, n_hidden, n_out, dropout=None):
        super(ResReg, self).__init__()
        with self.init_scope():

            self.dropout = dropout
            self.input = L.Linear(None, n_res_in)
            self.res_blocks = chainer.ChainList(
                    *[ResBlock(n_res_in, n_hidden) for n in range(n_res)]
                    )
            self.output = L.Linear(n_res_in, n_out)

    def __call__(self, x):

        h = F.relu(self.input(x))

        if self.dropout is not None:
            h = F.dropout(h, ratio=self.dropout)

        for R in self.res_blocks:
            h = F.relu(R(h))

        return self.output(h)

models['ResReg'] = ResReg


class BlinkNet(chainer.Chain):
    '''
    Parameters
    ----------
    locations: ndarray (n_sensors, n_dim)
        The locations of the sensors
    net_name: str
        The name of the network model to use
    *net_args:
        The positional arguments of the network
    **net_kwargs: 
        The keyword arguments of the network
    '''

    def __init__(self, locations, net_name, *net_args, **net_kwargs):
        super(BlinkNet, self).__init__()
        with self.init_scope():

            self.locations = chainer.Parameter(np.array(locations, dtype=np.float32)[None,:,:])
            self.network_x = models[net_name](*net_args, **net_kwargs)
            self.network_y = models[net_name](*net_args, **net_kwargs)
            self.eye = chainer.Parameter(np.eye(self.locations.shape[1], dtype=np.float32))

    def __call__(self, x):

        if x.ndim == 3:
            x = np.squeeze(x)

        max_loc = np.argmax(x, axis=1)
        loc = self.eye[max_loc,:]
        x = F.concat((x, loc), axis=1)

        h_x = self.network_x(x)
        h_y = self.network_y(x)

        h = F.concat((h_x[:,:,None], h_y[:,:,None]), axis=2)
        loc_bc = F.broadcast_to(self.locations, h.shape)

        return F.sum(h * loc_bc, axis=1)

models['BlinkNet'] = BlinkNet


class MaxLocNet(chainer.Chain):
    '''
    Parameters
    ----------
    locations: ndarray (n_sensors, n_dim)
        The locations of the sensors
    net_name: str
        The name of the network model to use
    *net_args:
        The positional arguments of the network
    **net_kwargs: 
        The keyword arguments of the network
    '''

    def __init__(self, locations, net_name, *net_args, **net_kwargs):
        super(MaxLocNet, self).__init__()
        with self.init_scope():

            self.locations = chainer.Parameter(np.array(locations, dtype=np.float32))
            self.network = models[net_name](*net_args, **net_kwargs)
            self.eye = chainer.Parameter(np.eye(self.locations.shape[0], dtype=np.float32))

    def __call__(self, x):
        # find blinky most likely closest to source
        if x.ndim == 3:
            x = np.squeeze(x)

        max_loc = np.argmax(x, axis=1)
        loc = self.locations[max_loc,:]
        one_hot = self.eye[max_loc,:]

        h = F.concat((x, one_hot), axis=1)
        h = self.network(h)

        return loc + h

models['MaxLocNet'] = MaxLocNet
