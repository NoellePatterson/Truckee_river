import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pdb

# Take in model processing outputs for each model run
model_outputs = glob.glob('data_outputs/Bayes_model_summaries/*.csv')
# average together all model outputs to get the final average stat across all models for each predictor/metric

metric_sig_counts = {}
for model in model_outputs:
    model = pd.read_csv(model)
    for index, metric in enumerate(model['metric']):
        if metric in metric_sig_counts.keys():
            val = (model['pre_sig'][index] + model['post_sig'][index])/2
            metric_sig_counts[metric].append(val)
        else:
            metric_sig_counts[metric] = []
            val = (model['pre_sig'][index] + model['post_sig'][index])/2
            metric_sig_counts[metric].append(val)

for metric in metric_sig_counts.keys():
    metric_sig_counts[metric] = np.nanmean(metric_sig_counts[metric])

output = pd.DataFrame(metric_sig_counts.items())
output.to_csv('data_outputs/bayes_model_summary.csv')

bars = pd.read_csv('data_outputs/bayes_model_summary_1.csv')
bars = bars.reindex([0,2,3,9,1,4,10,11,16,18,19,17,6,7,8,5,13,14,15,12])
colors = ['#C1C1C1','#7393B3','#7393B3','#7393B3','#AAAAAA','#AAAAAA','gold','gold','navy','cornflowerblue','cornflowerblue','cornflowerblue',
'yellowgreen','yellowgreen','yellowgreen','yellowgreen','lightcoral','lightcoral','lightcoral','lightcoral']
# next, plot these results as an attractive barchart. 
plt.figure(figsize=(10,7))
plt.bar(x=bars['name'], height=bars['val'], color=colors)
plt.xticks(rotation=80, fontweight='bold')
plt.ylabel('Model predictor significance (%)', fontweight='bold')
plt.tight_layout()
plt.savefig('data_outputs/bayes_bar.png', dpi=1200)
pdb.set_trace()
