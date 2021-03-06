import json, os
import numpy as np
from scipy.optimize import least_squares

import matplotlib.pyplot as plt
import seaborn as sns

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
    parser.add_argument('-p', '--plot', action='store_true', help='Plot the calibration curves')
    parser.add_argument('-i', '--inspect', type=float, help='Displays a single frame from the video, then exits')
    parser.add_argument('-c', '--color', choices=['red', 'green', 'blue', 'white'],
            help='LED color to process')
    parser.add_argument('--pwm', type=int,
            help='The number of bits of the PWM range')
    parser.add_argument('--no_fit', action='store_true',
            help='Do not plot the fit')
    parser.add_argument('--save_plot', type=str,
            help='The directory where to save the plot')
    args = parser.parse_args()

    with open(args.parameter_file, 'r') as f:
        calib = json.load(f)

    # Plot setting
    sns.set(style='white', context='paper', font_scale=1.,
            rc={
                'axes.facecolor': (0, 0, 0, 0),
                'figure.figsize':(3.38649 * (3.7 / 8), (3 / 8) * 3.338649),
                'lines.linewidth':1.,
                'text.usetex': False,
                })

    x_max = 0
    
    for color, param in calib.items():

        if args.color is not None and color != args.color:
            continue

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

        label = color.capitalize() + ' channel'

        plt.plot(y, c=col)

        if not args.no_fit:
            plt.plot(x * y.shape[0], param['fit'], '--', c=col)

        plt.ylabel('Norm. Intensity')

        if y.shape[0] > x_max:
            x_max = y.shape[0]

        print(color, param['coef'])

    if args.pwm is not None:
        if args.pwm > 0:
            plt.xlabel('PWM Duty Cycle')
            #plt.xticks([0, x_max], ['$0$', '$2^{' + str(args.pwm) + '}-1$'])
            plt.xticks([])
        else:
            plt.xlabel('Norm. Audio Power')
            plt.xticks([])


    plt.xlim(0, x_max)
    plt.ylim([0,1])
    plt.yticks([])

    sns.despine(left=False, bottom=False, offset=5)
    plt.legend()
    plt.tight_layout(pad=0.1)

    if args.save_plot is not None:
        fn = os.path.split(args.parameter_file)[1]
        fn = os.path.splitext(fn)[0] + '.pdf'
        fn = os.path.join(args.save_plot, fn)
        plt.savefig(fn)

    plt.show()


