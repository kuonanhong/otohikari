import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns

with open('figures/results_experiment_sir.json', 'r') as f:
    results = json.load(f)

sns.set(style='white', context='paper', font_scale=0.9,
        rc={
            'axes.facecolor': (0, 0, 0, 0),
            'figure.figsize':(3.38649,3.338649),
            'lines.linewidth':1.,
            'text.usetex': False,
            })

xticks = results['pyramic']['SIR_in']
xmin = np.min(results['pyramic']['SIR_in'])
xmax = np.max(results['pyramic']['SIR_in'])

equal = np.linspace(xmin-2, xmax+2)

plt.figure()

labels = {'pyramic': 'Pyramic - 24 ch.', 'olympus': 'Olympus E-PL2 - 2 ch.'}

sp = 1
for metric in ['SIR', 'SDR']:
    for mic in ['pyramic', 'olympus']:

        ax = plt.subplot(2,2,sp)

        # force integer tick labels
        ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
        ax.set_xticks(xticks)

        plt.bar(results[mic]['SIR_in'], results[mic]['{}_out'.format(metric)], label='_nolegend_', width=3)

        if sp == 1 or sp == 3:
            plt.ylabel('Output {} [dB]'.format(metric))

        if metric == 'SIR':

            plt.title('{}'.format(labels[mic]))

            plt.plot(equal, equal+5, 'g--', label='+5dB')
            plt.plot(equal, equal, 'k-', label='+0dB')

            leg = plt.legend(frameon=True, loc='lower right', framealpha=0.5)
            leg.get_frame().set_linewidth(0)
            leg.get_frame().set_facecolor('white')

            plt.ylim([0, 42])

        else:
            plt.ylim([0, 13])
            plt.xlabel('Input SIR [dB]')

        sns.despine()

        sp += 1

plt.tight_layout(pad=0.1)

plt.savefig('figures/experiment_sir.pdf', dpi=300)

plt.show()

