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


# isotopes = pd.read_csv('data_inputs/isotopes.csv')
# isotopes = isotopes.rename(columns={'Year':'year'})

# bb_bai = pd.read_csv('data_inputs/all_trees_bai/BB_bai.csv') 
# n_bai = pd.read_csv('data_inputs/all_trees_bai/N_bai.csv') 
# mre_bai = pd.read_csv('data_inputs/all_trees_bai/MRE_bai.csv') 
# mrw_bai = pd.read_csv('data_inputs/all_trees_bai/MRW_bai.csv') 
# r_bai = pd.read_csv('data_inputs/all_trees_bai/R_bai.csv') 

# # combine all data inputs into a giant table to iterate through
# tree_timeseries = [n_bai, bb_bai, mre_bai, mrw_bai, r_bai, isotopes]
# tree_timeseries = reduce(lambda  left,right: pd.merge(left,right,on=['year'], how='outer'), tree_timeseries)
# tree_timeseries = tree_timeseries.set_index('year')

# # before loop: create list for results to append with dicts
# results = {'name':[], 'tukey_73':[], 'mean_diff_73':[], 'perc_diff_73':[], 'mean_diff_82':[],'tukey_82':[], 'perc_diff_82':[]}
# def tukey_test(breakpoint_year, dict_name, mean_dict_name, perc_name):
#     period = []
#     for index, year in enumerate(timeseries.index):
#         if year < breakpoint_year:
#             period.append('1-pre')
#         else:
#             period.append('2-post')
#     tukey_dict = {'data':timeseries.tolist(), 'period':period}
#     tukey_df = pd.DataFrame(tukey_dict)
#     tukey_df = tukey_df.dropna()
#     if len(np.where(tukey_df['period']=='1-pre')[0]) < 15: # require at least 15 years in each comparison group
#         results[dict_name].append(np.nan)
#         results[mean_dict_name].append(np.nan)
#         results[perc_name].append(np.nan)
#     else:
#         tukey = pairwise_tukeyhsd(endog=tukey_df['data'], groups=tukey_df['period'], alpha=0.05)
#         results[dict_name].append(tukey.summary().data[1][-1])
#         results[mean_dict_name].append(tukey.summary().data[1][2])
#         pre_mean = np.nanmean(tukey_df['data'][tukey_df['period']=='1-pre'])
#         results[perc_name].append(tukey.summary().data[1][2]/pre_mean*100)
#     return
# # loop through each columns (each timeseries) in big table, performing tukey test
# for (name, timeseries) in tree_timeseries.iteritems():
#     results['name'].append(name)
    
#     output = tukey_test(1973, 'tukey_73', 'mean_diff_73', 'perc_diff_73')
#     output = tukey_test(1982, 'tukey_82', 'mean_diff_82', 'perc_diff_82')
# results_df = pd.DataFrame(results)
# results_df.to_csv('data_outputs/Tukey_hsd_alltrees.csv')


# make bar plot of results.
# bar_data = pd.read_csv('data_inputs/tukey_bar_plot.csv') 
names = ['Upstream', 'Downstream']
inc_bai = np.array([8, 10])
dec_bai = np.array([2, 1])
no_change_bai = np.array([6, 3])
inc_c13 = np.array([1,1]) 
dec_c13 = np.array([1,0])
plt.rcParams.update({'font.size': 16})
fig, axs = plt.subplots(2, 1,figsize=(7,7.3), gridspec_kw={'height_ratios': [4, 1]})
for axis in [axs[0].xaxis, axs[0].yaxis]:
    axis.set_major_locator(ticker.MaxNLocator(integer=True))

axs[0].bar(names, inc_bai, color='green', label='increasing')
axs[0].bar(names, dec_bai, bottom=inc_bai, color='#AB784E', label='decreasing')
axs[0].bar(names, no_change_bai, bottom=inc_bai+dec_bai, color='grey', label='no change')
axs[1].bar(names, inc_c13, color='green')
axs[1].bar(names, dec_c13, bottom=inc_c13, color='#AB784E')
plt.axhline(y=0, color='black')
axs[0].legend(fontsize=11)
plt.ylabel('Number of trees')
fig.tight_layout() 
plt.savefig('data_outputs/tukey_growth.jpeg', dpi=1200)
plt.show()
pdb.set_trace()

                        