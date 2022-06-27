from matplotlib.pyplot import prism
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from sklearn.linear_model import LinearRegression
import glob
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning) # to ignore append depracation warning


def fill_nas(streamflow_loc):
    flow_file = pd.read_csv(streamflow_loc)
    orig_dates = flow_file['date']
    date_start = datetime.strptime(orig_dates.iloc[0], '%m-%d-%Y')
    date_end = datetime.strptime(orig_dates.iloc[-1], '%m-%d-%Y')
    delta = date_end - date_start
    complete_dates = []
    complete_dates_dashed = []
    for i in range(delta.days + 1):
        day = date_start + timedelta(days=i)
        day_s = day.strftime('%m/%d/%Y')
        day_d = day.strftime('%m-%d-%Y')
        complete_dates.append(day_s)
        complete_dates_dashed.append(day_d)
    new_flow = [None] * len(complete_dates)
    for date_index, date in enumerate(complete_dates_dashed):
        for match_date_index, match_date in enumerate(orig_dates):
            if date == match_date:
                new_flow[date_index] = flow_file['flow'][match_date_index]

    output = pd.DataFrame(zip(complete_dates, new_flow), columns = ['date', 'flow'])
    output.to_csv('data_inputs/streamflow/renocomplete.csv')

def fill_flow_gaps(vista, reno):
    # fill gaps in vista data with daily flow from Reno station
    vista_start_index = vista['date'][vista['date']=='10/01/1906'].index[0] # shorten vista data to match Reno POR
    new_date = vista['date'][vista_start_index:]
    new_date = new_date.reset_index(drop=True)
    vista_cfs = vista['flow'][vista_start_index:] # need to reset indices otherwise df will not build properly
    vista_cfs = vista_cfs.reset_index(drop=True)
    reno_cfs = reno['flow'] 
    frame = {'date':new_date, 'vista_filled':vista_cfs, 'reno':reno_cfs}
    df = pd.DataFrame(frame)
    # relate vista to reno: vista = 1.0742(Reno) + 69.496 (from Excel linear fit)
    for index, vista_flow in enumerate(vista_cfs): 
        if np.isnan(vista_flow) == True:
            if np.isnan(reno_cfs[index]) == False:
                # vista value will remain nan if there is no reno value to replace it
                df.loc[index, 'vista_filled'] = 1.0742 * reno_cfs[index] + 69.496
    df.to_csv('data_inputs/streamflow/vista_gap_filled.csv')

def combine_prism_grids():
    prism_grids = glob.glob('data_inputs/PRISM_monthly_all_grids/*')
    grid_ls = []
    for grid in prism_grids:
        prism_data = pd.read_csv(grid, header=10)
        grid_ls.append(prism_data)
    date_col = prism_data['Date']
    prism_grids = pd.concat(grid_ls)
    prism_means = prism_grids.groupby(prism_grids.index)
    prism_means = prism_means.mean()
    prism_means['date'] = date_col
    prism_means.to_csv('data_inputs/prism_averaged_grids.csv', index=False)
  
def prism_annual_vals():
    prism_monthly = pd.read_csv('data_inputs/prism_averaged_grids.csv')
    dates = prism_monthly['date']
    prism_monthly['year'] = np.nan * len(dates)
    prism_monthly['month'] = np.nan * len(dates)
    for index, row in prism_monthly.iterrows():
        prism_monthly.loc[index, 'year'] = datetime.strptime(row['date'], '%Y-%m').year
        prism_monthly.loc[index, 'month'] = datetime.strptime(row['date'], '%Y-%m').month
    years = np.unique(prism_monthly['year'])

    summer_temps = {}
    annual_temps = {}
    precip_cum = {}
    vpd_max = {}
    for year_index, year in enumerate(years):
        summer_temps[str(int(year))] = []
        annual_temps[str(int(year))] = []
        precip_cum[str(int(year))] = []
        vpd_max[str(int(year))] = []
    # populate summer temp dict with monthly data
    for index, row in prism_monthly.iterrows():
        if row['month'] == 6 or row['month'] == 7 or row['month'] == 8:
            summer_temps[str(int(row['year']))].append(row['tmax (degrees C)'])
    summer_output = []
    for year in years:
        summer_output.append(np.nanmean(summer_temps[str(int(year))]))
    summer_output = pd.DataFrame(list(zip(years,summer_output,)),columns =['year','summer_max_temp'])
    # summer_output.to_csv('data_inputs/prism_summer_temps.csv', index=False)
    
    # populate annual temp dict with monthly data based on water years
    for index, row in prism_monthly.iterrows():
        if row['month'] == 10 or row['month'] == 11 or row['month'] == 12: # start indexing at 10 for Oct-start water year
            annual_temps[str(int(row['year'])+1)].append(row['tmean (degrees C)'])
        else:
            annual_temps[str(int(row['year']))].append(row['tmean (degrees C)'])
    mean_temp_output = []
    for year in years:
        mean_temp_output.append(np.nanmean(annual_temps[str(int(year))]))
    mean_temp_output = pd.DataFrame(list(zip(years, mean_temp_output,)),columns = ['year', 'annual_mean_temp'])
    mean_temp_output.to_csv('data_inputs/annual_mean_temp.csv', index=False)

    # populate cumulative precip dict with monthly data based on water years
    for index, row in prism_monthly.iterrows():
        if row['month'] == 10 or row['month'] == 11 or row['month'] == 12: # start indexing at 10 for Oct-start water year
            precip_cum[str(int(row['year'])+1)].append(row['ppt (mm)'])
        else:
            precip_cum[str(int(row['year']))].append(row['ppt (mm)'])
    precip_output = []
    for year in years:
        precip_output.append(np.sum(precip_cum[str(int(year))]))
    precip_output = pd.DataFrame(list(zip(years,precip_output,)),columns =['year','annual_precip'])
    precip_output.to_csv('data_inputs/annual_precip.csv', index=False)

    # populate VPD dict with cumulative monthly data based on water years
    for index, row in prism_monthly.iterrows():
        if row['month'] == 10 or row['month'] == 11 or row['month'] == 12:
            vpd_max[str(int(row['year'])+1)].append(row['vpdmax (hPa)'])
        else:
            vpd_max[str(int(row['year']))].append(row['vpdmax (hPa)'])
    vpd_output = []
    for year in years:
        vpd_output.append(np.nanmean(vpd_max[str(int(year))]))
    vpd_output = pd.DataFrame(list(zip(years,vpd_output,)),columns =['year','annual_max_vpd'])
    vpd_output.to_csv('data_inputs/annual_max_vpd.csv', index=False)
# output = prism_annual_vals()

def prism_spring_vals():
    prism_monthly = pd.read_csv('data_inputs/prism_averaged_grids.csv')
    dates = prism_monthly['date']
    prism_monthly['year'] = np.nan * len(dates)
    prism_monthly['month'] = np.nan * len(dates)
    for index, row in prism_monthly.iterrows():
        prism_monthly.loc[index, 'year'] = datetime.strptime(row['date'], '%Y-%m').year
        prism_monthly.loc[index, 'month'] = datetime.strptime(row['date'], '%Y-%m').month
    years = np.unique(prism_monthly['year'])

    spring_temps = {}
    spring_precip_cum = {}
    spring_vpd_max = {}
    for year_index, year in enumerate(years):
        spring_temps[str(int(year))] = []
        spring_precip_cum[str(int(year))] = []
        spring_vpd_max[str(int(year))] = []
    # populate spring temp dict with monthly data
    for index, row in prism_monthly.iterrows():
        if row['month'] == 4 or row['month'] == 5 or row['month'] == 6 or row['month'] == 7:
            spring_temps[str(int(row['year']))].append(row['tmean (degrees C)'])
    summer_output = []
    for year in years:
        summer_output.append(np.nanmean(spring_temps[str(int(year))]))
    summer_output = pd.DataFrame(list(zip(years,summer_output,)),columns =['year','spring_mean_temp'])
    summer_output.to_csv('data_inputs/prism_spring_temps.csv', index=False)
    
    # populate cumulative spring precip dict with monthly data
    for index, row in prism_monthly.iterrows():
        if row['month'] == 4 or row['month'] == 5 or row['month'] == 6 or row['month'] == 7:
            spring_precip_cum[str(int(row['year']))].append(row['ppt (mm)'])
    precip_output = []
    for year in years:
        precip_output.append(np.sum(spring_precip_cum[str(int(year))]))
    precip_output = pd.DataFrame(list(zip(years,precip_output,)),columns =['year','spring_precip'])
    precip_output.to_csv('data_inputs/spring_precip.csv', index=False)

    # populate spring VPD dict with cumulative monthly data
    for index, row in prism_monthly.iterrows():
        if row['month'] == 4 or row['month'] == 5 or row['month'] == 6 or row['month'] == 7:
            spring_vpd_max[str(int(row['year']))].append(row['vpdmax (hPa)'])
    vpd_output = []
    for year in years:
        vpd_output.append(np.nanmean(spring_vpd_max[str(int(year))]))
    vpd_output = pd.DataFrame(list(zip(years,vpd_output,)),columns =['year','spring_max_vpd'])
    vpd_output.to_csv('data_inputs/spring_max_vpd.csv', index=False)
# output = prism_spring_vals()

def huge_format():
    # organize data into huge format with all trees and variables all together
    # input necessary files: all trees bai, tree metadata, prism avgd grids, flow pearson vals (derby and vista)
    bb_bai = pd.read_csv('data_inputs/all_trees_bai/BB_bai.csv', index_col=0)
    mre_bai = pd.read_csv('data_inputs/all_trees_bai/MRE_bai.csv', index_col=0)
    mrw_bai = pd.read_csv('data_inputs/all_trees_bai/MRW_bai.csv', index_col=0)
    r_bai = pd.read_csv('data_inputs/all_trees_bai/R_bai.csv', index_col=0)
    n_bai = pd.read_csv('data_inputs/all_trees_bai/N_bai.csv', index_col=0)
    sites = [bb_bai, mre_bai, mrw_bai, r_bai, n_bai]

    tree_metadata = pd.read_csv('data_inputs/All_tree_metadata.csv')
    vpd = pd.read_csv('data_inputs/annual_max_vpd.csv')
    mean_t = pd.read_csv('data_inputs/annual_mean_temp.csv')
    precip = pd.read_csv('data_inputs/annual_precip.csv')
    # try a model version with spring-based input climate data
    # spring_vpd = pd.read_csv('data_inputs/spring_max_vpd.csv')
    # spring_t = pd.read_csv('data_inputs/prism_spring_temps.csv')
    # spring_precip = pd.read_csv('data_inputs/spring_precip.csv')
    ffc_blw_derby = pd.read_csv('data_inputs/streamflow/ffc_outputs/Blw_derby_pearson_vals.csv')
    ffc_vista = pd.read_csv('data_inputs/streamflow/ffc_outputs/vista_pearson_vals.csv')
    pdo = pd.read_csv('data_inputs/pdo.csv')

    
    # make sure environmental datasets rows are all traceable by year
    ffc_blw_derby = ffc_blw_derby.set_index('year')
    ffc_vista = ffc_vista.set_index('year')
    tree_metadata = tree_metadata.drop('Dbh_cm', axis=1)
    # prism_df = pd.merge(spring_vpd, spring_t, on='year')
    # prism_df = pd.merge(prism_df, spring_precip, on='year')
    prism_df = pd.merge(vpd, mean_t, on='year')
    prism_df = pd.merge(prism_df, precip, on='year')
    prism_df = pd.merge(prism_df, pdo, on='year')

    # Create variable for regulation period: pre/post 1973 for upstream sites, pre/post 1982 for downstream sites


    # Create a bunch of lists to fill iteratively with needed data
    meta_cols = tree_metadata.columns.tolist()
    flow_cols = ffc_blw_derby.columns.tolist()
    prism_cols = ['annual_max_vpd', 'annual_mean_temp', 'annual_precip', 'pdo']
    # prism_cols = ['spring_max_vpd', 'spring_mean_temp', 'spring_precip']
    columns = ['bai', 'year'] + meta_cols + ['regulation'] + prism_cols + flow_cols
    bai_ls = []
    year_ls = []
    tree_id_ls = []
    site_id_ls  = []
    lat_ls  = []
    lon_ls  = []
    sex_ls  = []
    to_river_ls  = []
    regulation = []
    annual_max_vpd_ls  = []
    annual_mean_temp_ls  = []
    annual_precip_ls  = []
    pdo_ls = []
    fa_mag_ls = []
    fa_tim_ls = []
    wet_BFL_Mag_50_ls = []	
    wet_Tim_ls = []	
    wet_BFL_Dur_ls = []	
    sp_Mag_ls = []	
    sp_Tim_ls = []	
    sp_ROC_ls = []	
    ds_Mag_50_ls = []	
    ds_Mag_90_ls = []	
    ds_Tim_ls = []	
    ds_Dur_WS_ls = []	
    avg_ls = []	
    cv_ls = []

    # Go through each site, by each column (tree) at a time. Append data to lists
    for site in sites:
        for tree in site.iteritems(): # iterates over a tuple, 0 is name 1 is attribute list
            # ID the row in tree_metadata that aligns with this tree
            for index, value in enumerate(tree_metadata['Tree_id']):
                if value == tree[0]:
                    meta_index = index
                    tree_id = tree_metadata['Tree_id'][meta_index]
                    break
            for tree_index, bai in enumerate(tree[1]): # iterating through years
                if np.isnan(bai) == True:
                    continue # don't add data before por of tree
                else:
                    bai_ls.append(bai)
                    tree_id_ls.append(tree_metadata['Tree_id'][meta_index])
                    site_id_ls.append(tree_metadata['Site_id'][meta_index])
                    lat_ls.append(tree_metadata['Lat'][meta_index])
                    lon_ls.append(tree_metadata['Lon'][meta_index]) 
                    sex_ls.append(tree_metadata['Sex'][meta_index]) 
                    to_river_ls.append(tree_metadata['To_river_m'][meta_index]) 

                    year = tree[1].index[tree_index]
                    # find correct index in prism dataset
                    for p_index, value in enumerate(prism_df['year']):
                        if value == year:
                            prism_index = p_index
                            break

                    # import pdb; pdb.set_trace()
                    annual_max_vpd_ls.append(prism_df['annual_max_vpd'][prism_index]) 
                    annual_mean_temp_ls.append(prism_df['annual_mean_temp'][prism_index]) 
                    annual_precip_ls.append(prism_df['annual_precip'][prism_index]) 
                    pdo_ls.append(prism_df['pdo'][prism_index])
                    # find correct index in streamflow dataset
                    if tree_id[0:3] == 'MR' or tree_id == 'R':
                        flow_metrics = ffc_vista
                    else:
                        flow_metrics = ffc_blw_derby
                    for f_index, value in enumerate(flow_metrics.index):
                        if value == year:
                            flow_index = f_index
                            break
                    # assign regulation period based on breakpoint year ()
                    # import pdb; pdb.set_trace()
                    site_name = tree_metadata['Site_id'][meta_index]
                    if site_name == 'BB' or site_name == 'N':
                        breakpoint = 1973
                    else:
                        breakpoint = 1982
                    if year < breakpoint:
                        regulation.append('pre')
                    else:
                        regulation.append('post')
                    
                    fa_mag_ls.append(flow_metrics['FA_Mag'].iloc[flow_index])
                    fa_tim_ls.append(flow_metrics['FA_Tim'].iloc[flow_index])
                    ds_Tim_ls.append(flow_metrics['DS_Tim'].iloc[flow_index]) 
                    ds_Mag_90_ls.append(flow_metrics['DS_Mag_90'].iloc[flow_index]) 
                    ds_Mag_50_ls.append(flow_metrics['DS_Mag_50'].iloc[flow_index]) 
                    ds_Dur_WS_ls.append(flow_metrics['DS_Dur_WS'].iloc[flow_index])
                    cv_ls.append(flow_metrics['CV'].iloc[flow_index]) 
                    avg_ls.append(flow_metrics['Avg'].iloc[flow_index])
                    sp_Mag_ls.append(flow_metrics['SP_Mag'].iloc[flow_index]) 
                    sp_ROC_ls.append(flow_metrics['SP_ROC'].iloc[flow_index]) 
                    sp_Tim_ls.append(flow_metrics['SP_Tim'].iloc[flow_index]) 
                    wet_BFL_Dur_ls.append(flow_metrics['Wet_BFL_Dur'].iloc[flow_index]) 
                    wet_BFL_Mag_50_ls.append(flow_metrics['Wet_BFL_Mag_50'].iloc[flow_index]) 
                    wet_Tim_ls.append(flow_metrics['Wet_Tim'].iloc[flow_index]) 
                    year_ls.append(year)           

    zipped_ls = list(zip(bai_ls, year_ls, tree_id_ls, site_id_ls, lat_ls, lon_ls, sex_ls, to_river_ls, regulation, annual_max_vpd_ls,
    annual_mean_temp_ls, annual_precip_ls, pdo_ls, fa_mag_ls, fa_tim_ls, wet_BFL_Mag_50_ls, wet_Tim_ls, wet_BFL_Dur_ls, sp_Mag_ls,
    sp_Tim_ls, sp_ROC_ls, ds_Mag_50_ls, ds_Mag_90_ls, ds_Tim_ls, ds_Dur_WS_ls, avg_ls, cv_ls))
    huge_df = pd.DataFrame(zipped_ls, columns=columns)
    huge_df.to_csv('data_outputs/all_model_data.csv')
    return
output = huge_format()