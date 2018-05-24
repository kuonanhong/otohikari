import json
import numpy as np
from scipy.optimize import least_squares

import matplotlib.pyplot as plt

def min_max_normalization(x):
    x_min = x.min()
    x_max = x.max()
    x_norm = (x - x_min) / ( x_max - x_min)
    return x_norm

def f_log(p, x):
    return np.log(1 + p[0] * x) / np.log(p[1])

def error(p, x, y):

    return  f_log(p, x) - y

def error_jac(p, x, y):
    ''' Jacobian of `f_log` '''

    log_arg = 1 + p[0] * x

    J = np.zeros((x.shape[0], p.shape[0]))
    J[:,0] = x / ( log_arg * np.log(p[1]) )
    J[:,1] = -np.log(log_arg) / ( p[1] * np.log(p[1]) ** 2 )

    return J

def error2(p, x, y):
    return (1 + p[0] * x) * p[1] ** (-y) - 1

def error2_jac(p, x, y):
    J = np.zeros((x.shape[0], p.shape[0]))
    J[:,0] = x * (p[1] ** (-y))
    J[:,1] = (1 + p[0] * x) * (-y) * (p[1] ** -(y + 1.))
    
    return J

def f_log3(p, x):
    return np.log(1 + (p[0] - 1) * x) / np.log(p[0])

def error3(p, x, y):

    return (1 + (p[0] - 1) * x) * p[0] ** (-y) - 1

def error3_jac(p, x, y):
    ''' Jacobian of `f_log` '''

    log_arg = 1 + p[0] * x

    J = np.zeros((x.shape[0], p.shape[0]))
    J[:,0] = x * p[0] ** (-y) + (1 + (p[0] - 1) * x) * (-y) * p[0]**(-y-1)

    return J

def log_fit(x, y):

    p0 = np.array([8.])

    #ret = least_squares(error, p0, jac=error_jac, method='lm', args=[x, y])
    ret = least_squares(error3, p0, jac=error3_jac, method='lm', args=[x, y])

    fit_curve = f_log3(ret.x, x)

    return ret.x, fit_curve


def const_poly_fit(x, y, deg):
    x = np.array(x)

    A = x[:,None] ** np.arange(1, deg+1)

    # constrain poly(0) = 0 and poly(1) = 1
    ATA_i = np.linalg.inv(np.dot(A.T, A))
    ls = np.dot(ATA_i, np.dot(A.T, y))
    lmbd = (np.sum(ls) - 1) / np.sum(ATA_i)
    c = ls - lmbd * np.sum(ATA_i, axis=1)
    
    fit_curve = np.dot(A, c)

    return c, fit_curve

def polyfit(x, y, deg):
    x = np.array(x)
    c = np.polyfit(x, y, deg)
    y_fit = np.dot(x[:,None] ** np.arange(deg, -1, -1), c)

    return c, y_fit

if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(description='Extract the LED intensity from the video.')
    parser.add_argument('parameter_file', metavar='FILE', help='The file that contains segmentation information')
    parser.add_argument('-s', '--save', type=str, help='Save the values to a file')
    parser.add_argument('-d', '--deg', default=2, type=int, help='Degree of the polynomial')
    parser.add_argument('-p', '--plot', action='store_true', help='Plot the calibration curves')
    parser.add_argument('-i', '--inspect', type=float, help='Displays a single frame from the video, then exits')
    args = parser.parse_args()

    with open(args.parameter_file, 'r') as f:
        calib = json.load(f)

    for color, param in calib.items():

        data = np.array(param['data'])
        data = data[:,0,:]

        x = np.linspace(0, 1, data.shape[0])

        if 'color_num' in param:
            y = np.array(data[:,param['color_num']])
        else:
            y = np.mean(data, axis=1)

        y = min_max_normalization(y)

        N = y.shape[0]

        #param['coef'], param['fit'] = const_poly_fit(x, y)
        param['coef'], param['fit'] = log_fit(x, y)
        #param['coef'], param['fit'] = piecewise_linear_fit(x, y, 20)
        #param['coef'], param['fit'] = polyfit(x, y, 10)

        if color == 'white':
            col = 'k'
        else:
            col = color

        plt.plot(x, param['fit'], label=color, c=col)
        plt.plot(x, y, label=color, c=col)

        print(color, param['coef'])

    plt.xlim(0, 1)
    plt.legend()
    plt.show()


