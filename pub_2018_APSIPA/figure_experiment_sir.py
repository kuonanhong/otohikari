import json, argparse, os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns
from tools import natural_keys


parser = argparse.ArgumentParser()
parser.add_argument('file', help='The JSON file containing the results to display')
parser.add_argument('output_dir', help='The output directory')
args = parser.parse_args()

with open(args.file, 'r') as f:
    data = json.load(f)
    results = data['results']
    parameters = data['parameters']

metrics = ['SDR', 'SIR']
mics = ['pyramic_48', 'pyramic_24', 'pyramic_4', 'pyramic_2', 'camera']


sns.set(style='whitegrid', context='paper', font_scale=1.,
        rc={
            'axes.facecolor': (0, 0, 0, 0),
            #'figure.figsize':(3.38649, 3.338649 / 4 * len(metrics)),
            'lines.linewidth':1.,
            'text.usetex': False,
            })

labels = {
        'pyramic_48': '48 ch',
        'pyramic_24': '24 ch',
        'pyramic_4': '4 ch',
        'pyramic_2': '2 ch',
        'camera' : '2 ch (camera)',
        }
algorithms = {
        'Max-SINR' : 'Max-SINR',
        'BSS' : 'AuxIVA',
        'Mix' : 'Mix',
        }

# create dataframe
df = pd.DataFrame(**data['results'])
df = df.rename(columns={'SIR_in' : 'Target Input SIR [dB]'})

# remove the camera measurements
df = df[df['Microphones'] != 'camera']

# remove the mix measurements for more than 2 channels
I = np.logical_or(df['Algorithm'] != 'Mix', df['Microphones'] == 'pyramic_2')
df = df[I]

df['Algorithms'] = (df['Microphones'].map(lambda x : labels[x])
        + ' - ' + df['Algorithm'].map(lambda x : algorithms[x]))

# For the mix, we want to drop the number of channels
df['Algorithms'] = df['Algorithms'].map(lambda x : 'Mix' if x == '2 ch - Mix' else x)

df2 = df.melt(
        id_vars=['Algorithms', 'Target Input SIR [dB]'],
        value_vars=['SDR', 'SIR'],
        value_name='[dB]', 
        var_name='Metric',
        )

hue_order = sorted(df['Algorithms'].unique(), key=natural_keys)
hue_order.remove('Mix')
hue_order.insert(0, 'Mix')

n_colors = len(hue_order)
sns.set_palette(sns.cubehelix_palette(n_colors, start=.5, rot=-.75))

g = sns.factorplot(
        data=df2,
        x='Target Input SIR [dB]',
        y='[dB]',
        hue='Algorithms',
        row='Metric',
        kind='bar',
        sharey=False,
        hue_order=hue_order,
        size=4,
        aspect=1.3,
        legend_out=False,
        legend=False,
        )
#plt.ylim([-10, 37.5])
g.set_titles('')
g.axes.flat[0].set_ylabel('SDR [dB]')
g.axes.flat[1].set_ylabel('SIR [dB]')
ax = g.axes.flat[0]
leg = ax.legend(loc='upper left', frameon=True, facecolor='w', framealpha=0.7, edgecolor='w')

sns.despine(left=True, bottom=True, offset=5)

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

fig_fn = os.path.join(args.output_dir, 
        '{}_{}_experiment_sir.pdf'.format(data['date'], data['git_commit']))
plt.savefig(fig_fn, dpi=300)

plt.show()

