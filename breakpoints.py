# Identify breakpoints in hydrologic data, to confirm change in hydrology based on flow regulation periods

import pandas as pd
import ruptures as rpt
import jenkspy
import numpy as np
import matplotlib.pyplot as plt
import pdb

# import data: ffc metrics for vista and blw derby
vista_upstream_ffc = pd.read_csv('data_inputs/streamflow/ffc_outputs/vista_pearson_vals.csv') # 113 yr POR mb start 1930 to skip over many nas
derby_downstream_ffc = pd.read_csv('data_inputs/streamflow/ffc_outputs/Blw_derby_pearson_vals.csv') # 102 yr POR
vista_upstream_ffc = vista_upstream_ffc.apply(pd.to_numeric, errors='coerce')
derby_downstream_ffc = derby_downstream_ffc.apply(pd.to_numeric, errors='coerce')

# set year as index
derby_downstream_ffc = derby_downstream_ffc.set_index('year')
vista_upstream_ffc = vista_upstream_ffc.set_index('year')


# make loop function, 
breakpoint_yrs = []
breakpoint_mag = []
metric_cols = vista_upstream_ffc.columns
window_len = 30
for metric in metric_cols:
    data = np.array(vista_upstream_ffc[metric])[23:] # for vista, skip over many NAs 
    breakpoint_results = pd.Series(np.nan, index=list(range(1,len(data)-(window_len*2)+1,1)))
    for index in range(len(data) - (window_len*2)):
        window1 = np.nanmean(data[index:index+window_len])
        window2 = np.nanmean(data[index+window_len:index+(window_len*2)])
        diff = abs(window1 - window2)
        breakpoint_results.iloc[index] = diff
    
    final = max(breakpoint_results)

    year_index = breakpoint_results[breakpoint_results==final].index[0] + window_len #+23 for upstream
    year = vista_upstream_ffc.index[year_index + 23]
    breakpoint_yrs.append(year)
    mag = (np.nanmean(data[year_index:]) - np.nanmean(data[:year_index]))/np.nanmean(data[:year_index])
    breakpoint_mag.append(mag)
    if metric == 'SP_ROC':
        pdb.set_trace()
    # calc diff as a percentage
blw_derby_output = pd.DataFrame(list(zip(breakpoint_yrs, breakpoint_mag)), columns=['years', 'percent diff'], index=metric_cols) 
blw_derby_output.to_csv('data_outputs/metric_diffs_vista.csv')
# pdb.set_trace()
