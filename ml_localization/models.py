import chainer
import chainer.functions as F
import chainer.links as L

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


class ResBlock(chainer.Chain):
    '''
    A two layers residual block

    Parameters
    ----------
    n: int
        the width of the block
    '''

    def __init__(self, n):
        super(ResBlock, self).__init__()
        with self.init_scope():

            self.hidden1 = L.Linear(n, n)
            self.hidden2 = L.Linear(n, n)

    def __call__(self, x):
        h = x
        h = F.relu(self.hidden1(h))
        h = self.hidden2(h)

        return x + h

class ResReg(chainer.Chain):

    def __init__(self, n_res, n_hidden, n_out):
        super(ResReg, self).__init__()
        with self.init_scope():

            self.input = L.Linear(None, n_hidden)
            self.res_blocks = chainer.ChainList(
                    *[ResBlock(n_hidden) for n in range(n_res)]
                    )
            self.output = L.Linear(n_hidden, n_out)

    def __call__(self, x):

        h = F.relu(self.input(x))

        for R in self.res_blocks:
            h = F.relu(R(h))

        return self.output(h)


models = dict(
        MLP=MLP,
        ResReg=ResReg,
        )
