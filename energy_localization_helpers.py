import numpy as np
import pyroomacoustics as pra
from scipy.optimize import least_squares

class VarPacker(object):
    '''
    This class is a helper to pack several variables of
    various sizes and shapes in a linear array.

    This is useful to run optimization on several variables
    '''

    def __init__(self, shapes):
        self.shapes = shapes

        lengths = [np.prod(shape) for shape in self.shapes]
        self.length = np.sum(lengths)
        self.bounds = np.r_[0, np.cumsum(lengths)]

    def new_vector(self):
        return np.zeros(self.length)

    def pack(self, *args):

        if len(args) != len(self.shapes):
            raise ValueError('Number of variables mismatch')

        for lo, hi, arg in zip(self.bounds[:-1], self.bounds[1:], args):
            self._vector[lo:hi] = arg

        return self._vector

    def unpack(self, vector):

        out = []
        for lo, hi, shape in zip(self.bounds[:-1], self.bounds[1:], self.shapes):
            out.append(vector[lo:hi].reshape(shape))
        
        return out


def objective(x, A, sigma, varobj, *args, **kwargs):
    '''
    This is the loss function for the energy based localization from the Microsoft group
    Paper by Chen et al.
    '''

    m, s, R, X = varobj.unpack(x)

    F = (A - m[:,None] - s[None,:] + np.log(pra.distance(R, X))) / sigma

    return F.ravel()

def jacobian(x, A, sigma, varobj, *args, **kwargs):
    '''
    This is the Jacobian function for the energy based localization from the Microsoft group
    Paper by Chen et al.
    '''

    m, s, R, X = varobj.unpack(x)
    dif = R[:,:,None] - X[:,None,:]
    D = np.sum(dif**2, axis=0)

    J = np.zeros(A.shape + (varobj.length,))

    # these two are independent from the i,j
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):

            # this is just a split and reshape trick
            dm, ds, dR, dX = varobj.unpack(J[i,j,:])

            # fill in where not zero
            dm[i] = -1 / sigma[i,j]
            ds[j] = -1 / sigma[i,j]
            dR[:,i] = dif[:,i,j] / D[None,i,j] / sigma[i,j]
            dX[:,j] = - dif[:,i,j] / D[None,i,j] / sigma[i,j]

            if 'fix_mic_gain' in kwargs and kwargs['fix_mic_gain']:
                dm[:] = 0

            if 'fix_src_gain' in kwargs and kwargs['fix_src_gain']:
                ds[:] = 0

            if 'fix_mic' in kwargs and kwargs['fix_mic']:
                dR[:,:] = 0

            if 'fix_src' in kwargs and kwargs['fix_src']:
                dX[:,:] = 0


            dm[0] = 0.
            dR[:,0] = 0.
            dR[1,1] = 0.

    return J.reshape((-1,J.shape[-1]))

def squared_loss(x, A, sigma, varobj, objective, jacobian, *args, **kwargs):
    '''
    This a squared loss wrapper around the objective and jacobian above
    '''

    obj = objective(x, A, sigma, varobj, *args, **kwargs)
    f_val = np.sum(obj**2)

    grad = 2. * np.sum(jacobian(x, A, sigma, varobj, *args, **kwargs) * obj[:,None], axis=0)

    return f_val, grad


def test_jacobian(eps, x, A, sigma, packer):

    jacobian_empirical = np.zeros(A.shape + x.shape)

    for i in range(x.shape[0]):
        x_up = x.copy()
        x_up[i] += eps
        x_down = x.copy()
        x_down[i] -= eps
        dif = (objective(x_up, A, sigma, packer) - objective(x_down, A, sigma, packer)).reshape(A.shape)
        jacobian_empirical[:,:,i] = dif / (2 * eps)

    jac_emp = jacobian_empirical.reshape((-1, x.shape[0]))
    jac_theory = jacobian(x, A, sigma, packer)
    err = np.linalg.norm(jac_emp - jac_theory) / np.prod(jacobian_empirical.shape)
    err /= np.linalg.norm(jac_theory)

    return err, jac_theory - jac_emp

def alternating_opt(A, sigma, R, m, s, X, n_iter=100, track_error=False, alpha=1):

    m = m.copy()
    s = s.copy()
    X = X.copy()
    D2 = pra.distance(R, X)**2

    def cost(A, sigma, m, s, D2, alpha):
        return np.linalg.norm((A + alpha * np.log(D2) - m[:,None] - s[None,:]) / sigma)

    from pylocus.lateration import SRLS

    if track_error:
        error = np.zeros(n_iter+1)
        error[0] = cost(A, sigma, m, s, D2, alpha)

    for epoch in range(n_iter):

        S = A + alpha * np.log(D2)
        m, s = cdm_unfolding(1. / sigma, S, sum_matrix=True)

        D2 = np.exp((m[:,None] + s[None,:] - A) / alpha)

        # Use SRLS to recompute locations of speakers
        for i in range(A.shape[1]):
            X[:,i] = SRLS(R.T, 1. / sigma[:,i], D2[:,i])

        D2 = pra.distance(R, X)**2

        if track_error:
            error[epoch+1] = cost(A, sigma, m, s, D2, alpha)

    if track_error:
        import matplotlib.pyplot as plt
        plt.plot(error)
        plt.show()


    return m, s, X, cost(A, W, m, s, D2, alpha)

def alternating_opt_2(A, sigma, R0, m0, s0, X0, step=0.001, n_iter=100, track_error=False, alpha=1):

    packer = VarPacker([m0.shape, s0.shape, R0.shape, X0.shape])
    x = packer.new_vector()
    m, s, R, X = packer.unpack(x)

    W = 1 / sigma
    m[:] = m.copy()
    s[:] = s.copy()
    X[:,:] = X.copy()
    R[:,:] = R0.copy()
    D2 = pra.distance(R, X)**2

    def cost(A, W, m, s, D2, alpha):
        return np.linalg.norm((A + alpha * np.log(D2) - m[:,None] - s[None,:]) * W)

    from pylocus.lateration import SRLS

    if track_error:
        error = np.zeros(n_iter+1)
        error[0] = cost(A, W, m, s, D2, alpha)

    for epoch in range(n_iter):

        # zero gradient for m and s
        S = A + alpha * np.log(D2)
        m[:], s[:] = cdm_unfolding(W, S, sum_matrix=True)

        # NNLS

        res_1 = least_squares(objective, x, jac=jacobian, 
                args=(A, sigma, packer),
                kwargs={'fix_mic' : True, 'fix_mic_gain' : True, 'fix_src_gain' : True},
                #ftol=1e-15, 
                #max_nfev=1000,
                method='lm', 
                #loss='huber',
                #verbose=1
                )
        m[:], s[:], R[:,:], X[:,:] = packer.unpack(res_1.x)


        '''
        # gradient step for X
        for loop in range(10):
            S = m[None,:,None] + s[None,None,:]
            direction_vector = (X[:,None,:] - R[:,:,None]) / D2[None,:,:]
            grad_X = 4 * np.sum(W[None,:,:] * direction_vector * (A[None,:,:] - S + alpha * np.log(D2[None,:,:])), axis=1)

            X -= step * grad_X

            D2 = pra.distance(R, X)**2
        '''

        D2 = pra.distance(R, X)**2

        if track_error:
            error[epoch+1] = cost(A, W, m, s, D2, alpha)

    if track_error:
        import matplotlib.pyplot as plt
        plt.plot(error)
        plt.show()


    return m, s, X, cost(A, sigma, m, s, D2, alpha)


def cdm_unfolding(W, S, sum_matrix=False):
    '''
    Coordinate Difference Matrix Unfolding

    Parameters
    ----------
    W : array_like
        weight matrix
    S : array_like
        Coordinate Difference Matrix
    sum_matrix : bool, optional
        If true, S is a coordinate sum matrix
    '''
    m,n = W.shape

    W2 = np.block( [ [ np.zeros((m,m)), W ],
                     [ W.T, np.zeros((n,n))] ] )[1:,1:]

    S2 = np.block( [ [ np.zeros((m,m)), S ],
                     [ -S.T, np.zeros((n,n))] ] )[1:,1:]

    numerator = 1 / np.sum(W2, axis=1)
    d = np.sum(S2 * W2, axis=1) * numerator

    A = np.eye(m+n-1) - numerator[:,None] * W2

    s = np.r_[0, np.linalg.lstsq(A, d)[0]]  # padded with zero in front

    # Fix the first coordinate
    s[0] = np.mean(S[0,:] + s[m:])

    if sum_matrix:
        return s[:m], -s[m:]
    else:
        return s[:m], s[m:]


