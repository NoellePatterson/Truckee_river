# Identify breakpoints in hydrologic data, to confirm change in hydrology based on flow regulation periods

import pandas as pd
import ruptures as rpt
import jenkspy
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import pdb
plt.rcParams["axes.grid.axis"] ="y"
plt.rcParams["axes.grid"] = True

# import data: ffc metrics for vista and blw derby
vista_upstream_ffc = pd.read_csv('data_inputs/streamflow/ffc_outputs/vista_pearson_vals.csv') # 113 yr POR mb start 1930 to skip over many nas
derby_downstream_ffc = pd.read_csv('data_inputs/streamflow/ffc_outputs/Blw_derby_pearson_vals.csv') # 102 yr POR
vista_upstream_ffc = vista_upstream_ffc.apply(pd.to_numeric, errors='coerce')
derby_downstream_ffc = derby_downstream_ffc.apply(pd.to_numeric, errors='coerce')

# set year as index
derby_downstream_ffc = derby_downstream_ffc.set_index('Year')
vista_upstream_ffc = vista_upstream_ffc.set_index('Year')


# make loop function, 
breakpoint_yrs = []
breakpoint_mag = []
breakpoint_perc = []
significance = []
# metric_cols = derby_downstream_ffc.columns
metric_cols = vista_upstream_ffc.columns
window_len = 30
for metric in metric_cols:
    # data = np.array(derby_downstream_ffc[metric])
    data = np.array(vista_upstream_ffc[metric])[23:] # for vista, skip over many NAs 
    breakpoint_results = pd.Series(np.nan, index=list(range(1,len(data)-(window_len*2)+1,1)))
    for index in range(len(data) - (window_len*2)):
        window1 = np.nanmean(data[index:index+window_len])
        window2 = np.nanmean(data[index+window_len:index+(window_len*2)])
        diff = abs(window1 - window2)
        breakpoint_results.iloc[index] = diff
    
    final = max(breakpoint_results)

    # year_index = breakpoint_results[breakpoint_results==final].index[0] + window_len 
    year_index = breakpoint_results[breakpoint_results==final].index[0] + window_len +23 # +23 for upstream
    # year = derby_downstream_ffc.index[year_index]
    year = vista_upstream_ffc.index[year_index]
    breakpoint_yrs.append(year)
    year_index = year_index - 23 # for upstream need to remove 23 from calcs
    perc = (np.nanmean(data[year_index:]) - np.nanmean(data[:year_index]))/np.nanmean(data[:year_index])
    mag = np.nanmean(data[year_index:]) - np.nanmean(data[:year_index])
    tuk_df = pd.DataFrame({'data': data, 'groups':np.repeat(['one', 'two'], repeats=[len(data)-year_index, year_index])})
    tuk_df = tuk_df.dropna()
    tukey_results = pairwise_tukeyhsd(endog=tuk_df['data'], groups=tuk_df['groups'], alpha=0.05)
    tukey_sig = tukey_results.summary().data[1][-1]

    breakpoint_mag.append(mag)
    breakpoint_perc.append(perc)
    significance.append(tukey_sig)
    # if metric == 'SP_ROC':
    #     pdb.set_trace()
    # calc diff as a percentage
# blw_derby_output = pd.DataFrame(list(zip(breakpoint_yrs, breakpoint_mag, breakpoint_perc, significance)), columns=['years', 'change mag', 'percent diff', 'Tukey hsd'], index=metric_cols) 
vista_output = pd.DataFrame(list(zip(breakpoint_yrs, breakpoint_mag, breakpoint_perc, significance)), columns=['years', 'change mag', 'percent diff', 'Tukey hsd'], index=metric_cols) 
# blw_derby_output.to_csv('data_outputs/metric_diffs_blw_derby.csv')
vista_output.to_csv('data_outputs/metric_diffs_vista.csv')
pdb.set_trace()

# Make plot printout of notable breakpoints

# Arrange data into same PORs/starting dates. Try starting at 1930 for Vista start date

vista_plot = vista_upstream_ffc.iloc[23:]
derby_plot = derby_downstream_ffc.iloc[12:]

# fill gaps in DG mag for plotting
derby_flow = pd.read_csv('data_inputs/streamflow/ffc_outputs/Blw_derby_annual_flow_matrix.csv')
derby_flow = derby_flow.iloc[:,12:]
derby_flow = derby_flow.apply(pd.to_numeric, errors='coerce')
ds_mag_plot = derby_plot['DS_Mag_50']
# loop through DS mag output, which should have same index as flox matrix columns
for index, val in enumerate(ds_mag_plot):
    if np.isnan(val) == True:
        dry_val = np.nanmean(derby_flow.iloc[276:366,index])
        ds_mag_plot.iloc[index] = dry_val

# Create paneled figure
plt.style.use('seaborn-white')
plt.figure(figsize=(10,8))
plt.subplots(ncols=1, nrows=4, sharex=True)
plt.subplot(4, 1, 1)
plt.plot(ds_mag_plot*0.0283168) # convert to cms
plt.axvline(1972, color='black', label = 'regime shift')
plt.axvline(1973, color='grey', linestyle='--')
plt.axvline(1982, color='grey', linestyle='--')
plt.title('Dry Season Median Magnitude', loc='left', fontweight='bold')
plt.ylabel('Flow (cms)')
plt.grid(axis='x')
plt.subplot(4, 1, 2)
plt.plot(derby_plot['Avg']*0.0283168) # convert to cms
plt.axvline(1970, color='black')
plt.axvline(1973, color='grey', linestyle='--')
plt.axvline(1982, color='grey', linestyle='--')
plt.title('Average Annual Flow', loc='left', fontweight='bold')
plt.ylabel('Flow (cms)')
plt.grid(axis='x')
plt.subplot(4, 1, 3)
plt.plot(derby_plot['CV'])
plt.axvline(1965, color='black')
plt.axvline(1973, color='grey', linestyle='--')
plt.axvline(1982, color='grey', linestyle='--')
plt.title('Coefficient of Variation', loc='left', fontweight='bold')
plt.ylabel('Percent')
plt.grid(axis='x')

plt.subplot(4, 1, 4)
plt.plot(vista_plot['SP_ROC']*100)
plt.axvline(1961, color='black')
plt.axvline(1973, color='grey', linestyle='--')
plt.axvline(1982, color='grey', linestyle='--')
plt.title('Spring Recession Rate-of-Change', loc='left', fontweight='bold')
plt.ylabel('Percent')
# plt.gca().xaxis.grid(True)
plt.grid(axis='x')
plt.xticks(range(1930, 2020, 10))
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5)
# plt.savefig('data_outputs/flow_breakpoints.png', dpi=1200)
plt.show()
pdb.set_trace()

# use breakpoints vals to draw vline on each plot
