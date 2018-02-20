import chainer
import chainer.functions as F
import chainer.links as L

# Network definition
class MLP1(chainer.Chain):

    def __init__(self, n_units, n_out):
        super(MLP1, self).__init__()
        with self.init_scope():
            # the size of the inputs to each layer will be inferred
            self.l1 = L.Linear(None, n_units)  # n_in -> n_units
            self.l2 = L.Linear(n_units, n_units)  # n_units -> n_units
            self.l3 = L.Linear(n_units, n_out)  # n_units -> n_out

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        return self.l3(h2)

# Network definition
class MLP2(chainer.Chain):

    def __init__(self, n_units, n_out):
        super(MLP2, self).__init__()
        with self.init_scope():
            # the size of the inputs to each layer will be inferred
            self.l1 = L.Linear(None, n_units)  # n_in -> n_units
            self.l2 = L.Linear(n_units, n_units // 2)  # n_units -> n_units
            self.l3 = L.Linear(n_units // 2, n_units // 4)  # n_units -> n_units
            self.l4 = L.Linear(n_units // 4, n_out)  # n_units -> n_out

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        h3 = F.relu(self.l3(h2))
        return self.l4(h3)

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
                    [ L.Linear(l_in, l_out) for l_in, l_out in zip(layers[:-1],layers[1:]) ]
                    )
                self.weights.append()

    def __call__(self, x):
        h = x
        for W in self.weights[:-1]:
            h = W(h)
            h = F.relu(h)
        return self.weights[-1](h)

    def to_gpu(self):
        '''
        We need to overload this to let chainer
        know about all the Linear objects in the list
        '''
        for W in self.weights:
            W.to_gpu()


models = dict(
        model1=MLP1,
        model2=MLP2,
        MLP=MLP,
        )
