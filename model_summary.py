import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pdb

# Take in model processing outputs for each model run
full_model_output = pd.read_csv('data_outputs/Bayes_model_summaries/bayes_model_siteprepost_summarymodel_parameters_all_params_8_06.csv')
# Generate bar plot only for full model

bars_pre = full_model_output['pre_sig']

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
