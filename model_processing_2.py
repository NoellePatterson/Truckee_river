import numpy as np
import pandas as pd
import glob
import pdb

# Now put the individual model summaries into on big table
# pull back in all the model summaries
models = glob.glob('data_outputs/Bayes_model_summaries/*.csv')
# Each model will be an ordered dict with same keys (metrics as appear in outputs, one for pre one for post) and blank values to start...
all_results_ever = {}

for model in models:
    summary_dict_pre = {'alpha':np.nan,'bavg':np.nan,'bprecip':np.nan,'btemp':np.nan,'bv':np.nan,'bcv':np.nan,'bfm':np.nan,
    'bft':np.nan,'bwd':np.nan,'bwm':np.nan,'bwt':np.nan,'bpm':np.nan,'bsd':np.nan,'bsm':np.nan,'bsr':np.nan,'bst':np.nan,
    'bddur':np.nan,'bdm50':np.nan,'bdm90':np.nan,'bdtim':np.nan}

    name = model[83:] # make sure file naming stays consistent
    model = pd.read_csv(model)
    for key in summary_dict_pre:
        for index, val in enumerate(model['metric']):
            if val == key:
                summary_dict_pre[key] = model['pre_sig'][index]
                continue
    all_results_ever[name+'_pre'] = summary_dict_pre

    summary_dict_post = {'alpha':np.nan,'bavg':np.nan,'bprecip':np.nan,'btemp':np.nan,'bv':np.nan,'bcv':np.nan,'bfm':np.nan,
    'bft':np.nan,'bwd':np.nan,'bwm':np.nan,'bwt':np.nan,'bpm':np.nan,'bsd':np.nan,'bsm':np.nan,'bsr':np.nan,'bst':np.nan,
    'bddur':np.nan,'bdm50':np.nan,'bdm90':np.nan,'bdtim':np.nan}
    
    for key in summary_dict_post:
        for index, val in enumerate(model['metric']):
            if val == key:
                summary_dict_post[key] = model['post_sig'][index]
                continue
    all_results_ever[name+'_post'] = summary_dict_post

df = pd.DataFrame(all_results_ever)
df.to_csv('data_outputs/Bayes_summary_full.csv')
