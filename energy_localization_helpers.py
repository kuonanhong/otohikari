import numpy as np
import pyroomacoustics as pra
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from rescaled_srls import rescaled_SRLS
from pylocus.lateration import SRLS

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

    m, s, R, X, alpha, scale = varobj.unpack(x)

    F = (A - m[:,None] - s[None,:] + alpha[0] * np.log(pra.distance(R, X)**2) + scale[0]) / sigma

    return F.ravel()

def jacobian(x, A, sigma, varobj, *args, **kwargs):
    '''
    This is the Jacobian function for the energy based localization from the Microsoft group
    Paper by Chen et al.
    '''

    m, s, R, X, alpha, scale = varobj.unpack(x)
    dif = R[:,:,None] - X[:,None,:]
    D2 = np.sum(dif**2, axis=0)

    J = np.zeros(A.shape + (varobj.length,))

    # these two are independent from the i,j
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):

            # this is just a split and reshape trick
            dm, ds, dR, dX, dalpha, dscale = varobj.unpack(J[i,j,:])

            # fill in where not zero
            dm[i] = -1 / sigma[i,j]
            ds[j] = -1 / sigma[i,j]
            dR[:,i] = 2 * alpha * dif[:,i,j] / D2[None,i,j] / sigma[i,j]
            dX[:,j] = - 2 * alpha * dif[:,i,j] / D2[None,i,j] / sigma[i,j]
            dalpha[:] = np.log(D2[i,j])
            dscale[:] = 1.

            if 'fix_mic_gain' in kwargs and kwargs['fix_mic_gain']:
                dm[:] = 0

            if 'fix_src_gain' in kwargs and kwargs['fix_src_gain']:
                ds[:] = 0

            if 'fix_mic' in kwargs and kwargs['fix_mic']:
                dR[:,:] = 0

            if 'fix_src' in kwargs and kwargs['fix_src']:
                dX[:,:] = 0

            if 'fix_alpha' in kwargs and kwargs['fix_alpha']:
                dalpha[:] = 0

            if 'noisy_gradient' in kwargs and kwargs['noisy_gradient']:
                J[i,j,:] += np.random.randn(varobj.length) * 0.01 * np.std(J[i,j,:])

            if 'fix_scale' in kwargs and kwargs['fix_scale']:
                dscale[:] = 0

            dm[0] = 0.
            dR[:,0] = 0.
            dR[1,1] = 0.

    return J.reshape((-1,J.shape[-1]))

def squared_loss(x, A, sigma, varobj, objective, jacobian, *args, **kwargs):
    '''
    This a squared loss wrapper around the objective and jacobian above
    '''

    obj = objective(x, A, sigma, varobj, *args, **kwargs)
    f_val = 0.5 * np.sum(obj**2)

    grad = np.sum(jacobian(x, A, sigma, varobj, *args, **kwargs) * obj[:,None], axis=0)

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

def energy_localization(A, sigma, mics_locations, n_iter=100, verbose=False):
    '''
    Energy based localization

    Parameters
    ----------
    A : array_like (n_microphones, n_sources)
        A matrix containing the log-energy of the sources at given sensors.
        A[m,k] contains the energy of the k-th source at the m-th microphone.
    sigma : array_like (n_microphones, n_sources)
        A matrix containing the noise standard deviations.
    R : array_like (n_dim, n_microphones)
        The location of the microphones
    verbose : bool, optional
        Printout stuff

    Returns
    -------
    gains : The gains of the microphones
    powers : The source powers
    source_locations : The location of the sources
    '''

    n_sources, n_mics = A.shape
    assert A.shape == sigma.shape, 'A and sigma should have the same shape'
    assert mics_locations.shape[1] == A.shape[0], 'The number of rows in A should be the same as the number of microphones'

    n_dim = mics_locations.shape[0]
    assert n_dim == 2 or n_dim == 3, 'Only 2D and 3D support'

    # Step 1 : Initialization
    #########################
    var_shapes = [(n_mics), (n_sources), (n_dim, n_mics), (n_dim, n_sources), (1,), (1,)] 
    packer = VarPacker(var_shapes)
    x0 = packer.new_vector()
    m0, s0, R0, X0, alpha0, scale0 = packer.unpack(x0)

    alpha0[:] = 0.5

    C = np.zeros((n_mics, n_sources))  # log attenuation of speaker at microphone
    D2 = np.zeros((n_mics, n_sources))  # squared distance between speaker and microphone
    for i in range(n_mics):
        for j in range(n_sources):
            if i < n_sources:
                C[i,j] = 0.5 * (A[i,j] + A[j,i] - A[i,i] - A[j,j])
            else:
                C[i,j] = A[i,j] - A[j,j]
            D2[i,j] = np.exp(-C[i,j] / alpha0[0])  # In practice, we'll need to find a way to fix the scale

    # log gain of device
    m0[0] = 0.  # i.e. m[0] = log(1)
    for i in range(1,n_mics):
        m0[i] = A[i,0] - A[0,0] - C[i,0] + m0[0]

    # energy of speaker
    for j in range(n_sources):
        s0[j] = np.mean(A[:,j] - m0[:] - C[:,j])

    # STEP 2 : SRLS for intial estimate of microphone locations
    ###########################################################

    # Fix microphone locations
    R0[:,:] = mics_locations

    # We can do some alternating optimization here
    # by increasing the number of loops
    scale = 1.
    pre_n_iter = 3
    for i in range(pre_n_iter):
        # Use SRLS to find intial guess of locations
        scalings = np.zeros(n_sources)
        for j in range(n_sources):
            X0[:,j], scalings[j] = rescaled_SRLS(R0.T, np.ones(n_mics), D2[:,j,None])
            A[:,:] += alpha0[0] * np.log(scalings[j])

        scale = np.sqrt(np.median(scalings))

        # Reinitialize gains from the SRLS distances
        S = A + alpha0[0] * np.log(pra.distance(R0, X0)**2)
        m0[:], s0[:] = cdm_unfolding(1 / sigma**2, S, sum_matrix=True)

        D2 = np.exp((- A + m0[:,None] + s0[None,:]) )

    scale0[:] = 0.

    # STEP 4 : Non-linear least-squares
    ###################################

    # Create a variable to loop over
    x = packer.new_vector()
    m, s, R, X, alpha, scale = packer.unpack(x)
    x[:] = x0

    # keep track of minimum
    x_opt = packer.new_vector()
    cost_opt = np.inf

    for i in range(n_iter):
        # noise injection
        if i > 0:
            m[:] += np.random.randn(*m.shape) * 0.01 * np.std(m)
            s[:] += np.random.randn(*s.shape) * 0.01 * np.std(s)
            X[:,:] += np.random.randn(*X.shape) * 0.30 * np.std(X)

        # Non-linear least squares solver
        res_1 = least_squares(objective, x, jac=jacobian, 
                args=(A, sigma, packer),
                kwargs={'fix_mic' : True, 'fix_alpha' : False, 'fix_scale' : True},
                xtol=1e-15,
                ftol=1e-15, 
                max_nfev=100,
                method='lm', 
                verbose=verbose,
                )

        if res_1.cost < cost_opt:
            x_opt[:] = res_1.x
            cost_opt = res_1.cost

        # use result as next initialization
        x[:] = res_1.x

    m, s, R, X, alpha, scale = packer.unpack(x_opt)

    return m, s, X, X0
