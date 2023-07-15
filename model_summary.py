import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pdb

# Generate bar plot only for full model
scatter = pd.read_csv('data_outputs/bayes_model_summary_1.csv')

values = scatter['mean']
x_val = range(len(values))

y_top = abs(scatter['mean'] - scatter['conf_2.5'])
y_bot = abs(scatter['conf_97.5'] - scatter['mean'])
y_error = [y_top, y_bot]

colors = ['#7393B3','#7393B3','#7393B3','#7393B3','#7393B3','#7393B3','#AAAAAA','#AAAAAA','gold','gold',
'navy','cornflowerblue','cornflowerblue','yellowgreen','yellowgreen','yellowgreen','lightcoral','lightcoral']
# next, plot these results as an attractive barchart. 
plt.figure(figsize=(10,5))
plt.rc('font', size=10) 

# plot post-reg bars with color
pdb.set_trace()
plt.errorbar(x_val, values, yerr=y_error, fmt='none', ecolor=colors)
plt.scatter(x_val, values, color=colors)


plt.axhline(0, color='black', linestyle = '--', linewidth=0.5)
plt.xticks(rotation=80, fontweight='bold')
tick_space = np.arange(-0.5, 17.5, 1)
plt.xticks(tick_space, minor=True)
plt.grid(axis='x', which='minor', linestyle = '--')
plt.ylabel('Model predictor significance', fontweight='bold')
plt.xticks(ticks=x_val, labels=scatter['name'])
plt.tight_layout()
plt.savefig('data_outputs/bayes_bar.png', dpi=1200)
# pdb.set_trace()
