# perform a time period comparison analysis using simple Tukey's HSD test

import pandas as pd
from functools import reduce
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import numpy as np
import pdb
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("white")
import matplotlib.ticker as ticker

# import all relevant datasets from csv
isotopes = pd.read_csv('data_inputs/isotopes.csv')
isotopes = isotopes.rename(columns={'Year':'year'})

bb_bai = pd.read_csv('data_inputs/all_trees_bai/BB_bai.csv') 
n_bai = pd.read_csv('data_inputs/all_trees_bai/N_bai.csv') 
mre_bai = pd.read_csv('data_inputs/all_trees_bai/MRE_bai.csv') 
mrw_bai = pd.read_csv('data_inputs/all_trees_bai/MRW_bai.csv') 
r_bai = pd.read_csv('data_inputs/all_trees_bai/R_bai.csv') 

# combine all data inputs into a giant table to iterate through
tree_timeseries = [n_bai, bb_bai, mre_bai, mrw_bai, r_bai, isotopes]
tree_timeseries = reduce(lambda  left,right: pd.merge(left,right,on=['year'], how='outer'), tree_timeseries)
tree_timeseries = tree_timeseries.set_index('year')

# before loop: create list for results to append with dicts
results = {'name':[], 'tukey_73':[], 'mean_diff_73':[], 'perc_diff_73':[], 'mean_diff_82':[],'tukey_82':[], 'perc_diff_82':[]}
def tukey_test(breakpoint_year, dict_name, mean_dict_name, perc_name):
    period = []
    for index, year in enumerate(timeseries.index):
        if year < breakpoint_year:
            period.append('1-pre')
        else:
            period.append('2-post')
    tukey_dict = {'data':timeseries.tolist(), 'period':period}
    tukey_df = pd.DataFrame(tukey_dict)
    tukey_df = tukey_df.dropna()
    if len(np.where(tukey_df['period']=='1-pre')[0]) < 15: # require at least 15 years in each comparison group
        results[dict_name].append(np.nan)
        results[mean_dict_name].append(np.nan)
        results[perc_name].append(np.nan)
    else:
        tukey = pairwise_tukeyhsd(endog=tukey_df['data'], groups=tukey_df['period'], alpha=0.05)
        results[dict_name].append(tukey.summary().data[1][-1])
        results[mean_dict_name].append(tukey.summary().data[1][2])
        pre_mean = np.nanmean(tukey_df['data'][tukey_df['period']=='1-pre'])
        results[perc_name].append(tukey.summary().data[1][2]/pre_mean*100)
    return
# loop through each columns (each timeseries) in big table, performing tukey test
for (name, timeseries) in tree_timeseries.iteritems():
    results['name'].append(name)
    
    output = tukey_test(1973, 'tukey_73', 'mean_diff_73', 'perc_diff_73')
    output = tukey_test(1982, 'tukey_82', 'mean_diff_82', 'perc_diff_82')
results_df = pd.DataFrame(results)
results_df.to_csv('data_outputs/Tukey_hsd_alltrees.csv')


# make bar plot of results.
# bar_data = pd.read_csv('data_inputs/tukey_bar_plot.csv') 
names = ['Upstream', 'Downstream']
inc = np.array([8, 10])
dec = np.array([2, 1])
no_change = np.array([6, 3])
plt.rcParams.update({'font.size': 16})
fig, ax = plt.subplots(figsize=(7,6))
for axis in [ax.xaxis, ax.yaxis]:
    axis.set_major_locator(ticker.MaxNLocator(integer=True))

ax.bar(names, inc, color='green')
ax.bar(names, dec, bottom=inc, color='#AB784E')
ax.bar(names, no_change, bottom=inc+dec, color='grey')
ax.legend(['increasing', 'decreasing', 'no change'], fontsize=11)
plt.ylabel('Number of trees')
fig.tight_layout() 
plt.savefig('data_outputs/tukey_growth.jpeg', dpi=1200)
pdb.set_trace()
    
                          
                          

    


