import scipy
from scipy.linalg import eigvals
import numpy as np
from pylocus.lateration import SRLS

def rescaled_SRLS(anchors, W, r2, print_out=False):
    '''Squared range least squares (SRLS)
    Algorithm written by A.Beck, P.Stoica in "Approximate and Exact solutions of Source Localization Problems".

    Orignal implementation by F. Duembgen in PyLocus
    Modification for rescaled range by R. Scheibler, Dec 2017

    :param anchors: anchor points
    :param r2: squared distances from anchors to point x.
    :return: estimated position of point x.
    '''

    def y_hat(_lambda):
        lhs = ATA + _lambda * D
        assert A.shape[0] == b.shape[0]
        assert A.shape[1] == f.shape[0], 'A {}, f {}'.format(A.shape, f.shape)
        rhs = (np.dot(A.T, b) - _lambda * f).reshape((-1,))
        assert lhs.shape[0] == rhs.shape[0], 'lhs {}, rhs {}'.format(lhs.shape, rhs.shape)
        try:
            return np.linalg.solve(lhs, rhs)
        except:
            return np.zeros((lhs.shape[1],))

    def phi(_lambda):
        yhat = y_hat(_lambda).reshape((-1, 1))
        return np.dot(yhat.T, np.dot(D, yhat)) + 2 * np.dot(f.T, yhat)

    def phi_prime(_lambda):
        # TODO: test this.
        B = np.linalg.inv(ATA + _lambda * D)
        C = A.T.dot(b) - _lambda*f
        y_prime = -B.dot(D.dot(B.dot(C)) - f)
        y = y_hat(_lambda)
        return 2*y.T.dot(D).dot(y_prime) + 2*f.T.dot(y_prime)

    from scipy import optimize
    from scipy.linalg import sqrtm

    # Set up optimization problem
    n = anchors.shape[0]
    d = anchors.shape[1]

    if n < 3:
        raise ValueError('At least 3 anchors are needed for rescaled SRLS.')

    A = np.c_[-2 * anchors, np.ones((n, 1)), -r2]
    Sigma = np.diagflat(np.power(W, 0.5))
    A = Sigma.dot(A)
    ATA = np.dot(A.T, A)
    b = - np.power(np.linalg.norm(anchors, axis=1), 2).reshape(r2.shape)
    b = Sigma.dot(b)
    D = np.zeros((d + 2, d + 2))
    D[:d, :d] = np.eye(d)
    f = np.c_[np.zeros((1, d)), -0.5, 0].T
    eig = np.sort(np.real(eigvals(a=D, b=ATA)))
    if (print_out):
        print('ATA:', ATA)
        print('rank:', np.linalg.matrix_rank(A))
        print('eigvals:', eigvals(ATA))
        print('condition number:', np.linalg.cond(ATA))
        print('generalized eigenvalues:',eig)
    eps = 0.01
    if eig[-1] > 1e-10:
        I_orig = -1.0 / eig[-1] + eps
    else:
        print('Warning: biggest eigenvalue is zero!')
        I_orig = -1e-5
    #assert phi(I_orig) < 0 and phi(inf) > 0 
    inf = 1e5
    xtol = 1e-12
    try:
        lambda_opt = optimize.bisect(phi, I_orig, inf, xtol=xtol)
    except:
        print('Bisect failed. Trying Newton...')
        lambda_opt = I_orig
        try:
            lambda_opt = optimize.newton(phi, I_orig, fprime=phi_prime, maxiter=1000, tol=xtol)
            assert phi(lambda_opt) < xtol, 'did not find solution of phi(lambda)=0:={}'.format(phi(lambda_opt))
        except:
            print('SRLS ERROR: Did not converge. Setting lambda to 0.')
            lambda_opt = 0

    if (print_out):
        print('phi I_orig', phi(I_orig))
        print('phi inf', phi(inf))
        print('phi opt', phi(lambda_opt))

    # Compute best estimate
    yhat = y_hat(lambda_opt)

    return yhat[:d], yhat[-1]

if __name__ == '__main__':

    d = 3
    n = 10
    sigma = 3.

    x = np.ones(3) * 4
    anchors = np.ones((n,d))
    anchors[:,0] = np.arange(n)
    anchors[:,1] = np.arange(n)[::-1] + np.random.randn(n)
    anchors[:,2] = np.random.randn(n)

    r2 = np.linalg.norm(anchors - x[None,:], axis=1)**2

    # add noise
    #r2 += np.random.randn(*r2.shape) * 0.01 * np.std(r2)

    # rescale
    r2 *= sigma

    print(x)

    x_hat_srls = SRLS(anchors, np.ones(n), r2)
    print('Output from SRLS without rescaling :', x_hat_srls)

    x_hat_rsrls, sigma_hat = rescaled_SRLS(anchors, np.ones(n), r2)
    print('Output of SRLS with rescaling :', x_hat_rsrls)
    print('  The scale that was found :', sigma_hat)
