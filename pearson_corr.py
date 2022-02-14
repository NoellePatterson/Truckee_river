import pandas as pd
from functools import reduce
import seaborn as sns
import matplotlib.pyplot as plt
# Calculate Pearson correlation coefficients for site-averaged growth data and environmental variables

# import all relevant datasets from csv
bb_bai = pd.read_csv('data_inputs/site_avg_bai/BB_bai_site_avg.csv') # 109 yr POR
n_bai = pd.read_csv('data_inputs/site_avg_bai/N_bai_site_avg.csv') # 124 yr POR
mre_bai = pd.read_csv('data_inputs/site_avg_bai/MRE_bai_site_avg.csv') # 38 yr POR
mrw_bai = pd.read_csv('data_inputs/site_avg_bai/MRW_bai_site_avg.csv') # 73 yr POR
r_bai = pd.read_csv('data_inputs/site_avg_bai/R_bai_site_avg.csv') # 68 yr POR

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
for index, value in enumerate(derby_downstream_ffc['year']):
    derby_downstream_ffc['year'].iloc[index] = value + 1
    vista_upstream_ffc['year'].iloc[index] = vista_upstream_ffc['year'].iloc[index] + 1

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