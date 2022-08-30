import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pdb

# Generate bar plot only for full model
bars = pd.read_csv('data_outputs/bayes_model_summary_1.csv')

bars_pre = bars['pre_sig']
bars_post = bars['post_sig']

colors = ['#C1C1C1','#7393B3','#7393B3','#7393B3','#AAAAAA','#AAAAAA','gold','gold','navy','cornflowerblue','cornflowerblue','cornflowerblue',
'yellowgreen','yellowgreen','yellowgreen','yellowgreen','lightcoral','lightcoral','lightcoral','lightcoral']
# next, plot these results as an attractive barchart. 
plt.figure(figsize=(10,7))
# plot post-reg bars with color
plt.bar(x=bars['name'], height=bars_post, color=colors)
# plot pre-reg bars as transparent with hatches over colored bars
plt.bar(x=bars['name'], height=bars_pre, facecolor='none', hatch='/', edgecolor='black')
plt.xticks(rotation=80, fontweight='bold')
plt.ylabel('Model predictor significance (%)', fontweight='bold')
plt.tight_layout()
plt.savefig('data_outputs/bayes_bar.png', dpi=1200)
pdb.set_trace()
