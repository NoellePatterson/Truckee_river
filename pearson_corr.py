import pandas as pd
import numpy as np
from functools import reduce
import seaborn as sns
import matplotlib.pyplot as plt
import pdb
# Calculate Pearson correlation coefficients for site-averaged growth data and environmental variables

# import all relevant datasets from csv
isotopes = pd.read_csv('data_inputs/isotopes.csv')
isotopes = isotopes.rename(columns={'Year':'year'})

bb_bai = pd.read_csv('data_inputs/site_avg_bai/BB_bai_site_avg.csv') # 109 yr POR
n_bai = pd.read_csv('data_inputs/site_avg_bai/N_bai_site_avg.csv') # 124 yr POR
mre_bai = pd.read_csv('data_inputs/site_avg_bai/MRE_bai_site_avg.csv') # 38 yr POR
mrw_bai = pd.read_csv('data_inputs/site_avg_bai/MRW_bai_site_avg.csv') # 73 yr POR
r_bai = pd.read_csv('data_inputs/site_avg_bai/R_bai_site_avg.csv') # 68 yr POR

r_bai_all = pd.read_csv('data_inputs/all_trees_bai/R_bai.csv')
n_bai_all = pd.read_csv('data_inputs/all_trees_bai/N_bai.csv')
mrw_bai_all = pd.read_csv('data_inputs/all_trees_bai/MRW_bai.csv')

vpd = pd.read_csv('data_inputs/annual_max_vpd.csv')
precip = pd.read_csv('data_inputs/annual_precip.csv')
summer_temps = pd.read_csv('data_inputs/prism_summer_temps.csv')
vista_upstream_ffc = pd.read_csv('data_inputs/streamflow/ffc_outputs/vista_pearson_vals.csv') # 113 yr POR
derby_downstream_ffc = pd.read_csv('data_inputs/streamflow/ffc_outputs/Blw_derby_pearson_vals.csv') # 102 yr POR
vista_upstream_ffc = vista_upstream_ffc.apply(pd.to_numeric, errors='coerce')

# shorten all columns to smallest col size so matrix will have equal shape. Also remove yr 2020 from tree records (incomplete growth)
n_bai = n_bai[21:-1].reset_index(drop=True)
bb_bai = bb_bai[6:-1].reset_index(drop=True)
mre_bai = mre_bai[:-1].reset_index(drop=True)
mrw_bai = mrw_bai[:-1].reset_index(drop=True)
r_bai = r_bai[:-1].reset_index(drop=True)

# try offsetting streamflow metrics by one year (lag effect) to see if corrs are stronger
# for index, value in enumerate(derby_downstream_ffc['year']):
#     derby_downstream_ffc['year'].iloc[index] = value + 1
#     vista_upstream_ffc['year'].iloc[index] = vista_upstream_ffc['year'].iloc[index] + 1

# run pearson corr on matrix. 
# create input correlation matrix for BB
def create_correlation_matrix(site_bai, site_flow_data, output_name):
    variable_dfs = [vpd, precip, summer_temps, site_bai, site_flow_data]
    variable_df = reduce(lambda  left,right: pd.merge(left,right,on=['year'], how='outer'), variable_dfs)
    # variable_df = variable_df[87:].reset_index(drop=True)
    # consider removing first rows of df before sflow or tree data
    correlation_mat = variable_df.corr()
    plt.figure(figsize=(12,12))
    plt.subplot(1,1,1)
    sns.heatmap(correlation_mat, annot = True)
    plt.title(output_name + ' Pearson correlations')
    plt.tight_layout()
    # import pdb; pdb.set_trace()
    plt.savefig('data_outputs/{}_corr_matrix_flow_offset.jpeg'.format(output_name), bbox_inches='tight')
corr = create_correlation_matrix(r_bai, vista_upstream_ffc, 'Ranch 102')

upstream_isotopes = isotopes.drop(['N'], axis=1)
downstream_isotopes = isotopes[['year', 'N']]

r_rings_isotopes =  r_bai_all[['year', 'R_isotopes']] # pull out ring averages for same trees used in isotope analysis
n_rings_isotopes =  n_bai_all[['year', 'N_isotopes']]
mrw_rings_isotopes =  mrw_bai_all[['year', 'MRW_isotopes']]

# create input correlation matrix for isotopes
def create_correlation_matrix(isotopes_input, output_name):
    variable_dfs = [r_rings_isotopes, n_rings_isotopes, mrw_rings_isotopes, isotopes_input]
    variable_df = reduce(lambda  left,right: pd.merge(left,right,on=['year'], how='outer'), variable_dfs)
    # variable_df = variable_df[87:].reset_index(drop=True)
    # consider removing first rows of df before sflow or tree data
    correlation_mat = variable_df.corr()
    plt.figure(figsize=(12,12))
    plt.subplot(1,1,1)
    sns.heatmap(correlation_mat, annot = True)
    plt.title(output_name + ' Pearson correlations')
    plt.tight_layout()
    plt.savefig('data_outputs/{}_corr_matrix.jpeg'.format(output_name), bbox_inches='tight')
# corr = create_correlation_matrix(isotopes,  'Isotopes_to_measured_rings')


# Try a time lagged cross correlation by adapting this code:
def crosscorr(datax, datay, lag=0, wrap=False):
    """ Lag-N cross correlation. 
    Shifted data filled with NaNs 
    
    Parameters
    ----------
    lag : int, default 0
    datax, datay : pandas.Series objects of equal length
    Returns
    ----------
    crosscorr : float
    """
    if wrap:
        shiftedy = datay.shift(lag)
        shiftedy.iloc[:lag] = datay.iloc[-lag:].values
        return datax.corr(shiftedy)
    else: 
        return datax.corr(datay.shift(lag))

def align_data(d1, d2):
    dfs = reduce(lambda  left,right: pd.merge(left,right,on=['year'], how='outer'), [d1, d2])
    dfs = dfs.dropna()
    d1 = dfs.iloc[:,1]
    d2 = dfs.iloc[:,2]
    return(d1, d2)

# get each pair of isotope data and bai to align by years and limit to same POR (write function?)
# figure out which axis to move time lag on
# fit datasets into code below.
d1 = isotopes[['year', 'N']]
d2 = n_rings_isotopes

d1, d2 = align_data(d1, d2)

shift_len = 15
rs = [crosscorr(d1,d2, lag) for lag in range(-int(shift_len),int(shift_len+1))]
offset = np.floor(len(rs)/2)-np.argmax(rs)
offset = shift_len
f,ax=plt.subplots(figsize=(14,5))
ax.plot(rs)
ax.axvline(offset, color='k',linestyle='--',label='Center')
ax.axvline(np.argmax(rs),color='r',linestyle='--',label='Peak synchrony')
ax.set(title=f'Numana trees N 5, 7, 13, 14\nIsotope leads <> BAI leads     ', xlabel='Offset',ylabel='Pearson r')
ax.set_xticks([0, 5, 10, 15, 20, 25, 30])
ax.set_xticklabels([-15, -10, -5, 0, 5, 10, 15]);
plt.legend()
# plt.savefig('data_outputs/Cross correlations/Cross_correlation_N_iso_trees_only.jpeg', bbox_inches='tight')
pdb.set_trace()
