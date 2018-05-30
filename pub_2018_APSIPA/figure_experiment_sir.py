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
mics = ['pyramic_48', 'pyramic_24', 'pyramic_4', 'camera']


sns.set(style='whitegrid', context='paper', font_scale=0.9,
        rc={
            'axes.facecolor': (0, 0, 0, 0),
            'figure.figsize':(3.38649, 3.338649 * len(metrics)),
            'lines.linewidth':1.,
            'text.usetex': False,
            })

labels = {
        'pyramic_48': '48 ch',
        'pyramic_24': '24 ch',
        'pyramic_4': '4 ch',
        'camera': '2 ch',
        }

df = pd.DataFrame(**data['results'])
df = df.rename(columns={'SIR_in' : 'Input SIR'})
df['Algorithms'] = df['Microphones'].map(lambda x : labels[x]) + ' - ' + df['Algorithm']
df2 = df.melt(
        id_vars=['Algorithms', 'Input SIR'],
        value_vars=['SDR', 'SIR'],
        value_name='[dB]', 
        var_name='Metric',
        )

hue_order = sorted(df['Algorithms'].unique(), key=natural_keys)

sns.factorplot(
        data=df2,
        x='Input SIR',
        y='[dB]',
        hue='Algorithms',
        col='Metric',
        kind='bar',
        sharey=True,
        hue_order=hue_order,
        )

sns.despine(left=True, bottom=True, offset=5)

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

fig_fn = os.path.join(args.output_dir, 
        '{}_{}_experiment_sir.pdf'.format(data['date'], data['git_commit']))
plt.savefig(fig_fn, dpi=300)

plt.show()

