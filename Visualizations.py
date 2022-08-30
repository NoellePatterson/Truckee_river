import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
import seaborn as sns
from matplotlib import gridspec
from sklearn import preprocessing
import glob
import pdb

# plot overlaid years from two different usgs gages
# import daily flow data
vista = pd.read_csv('data_inputs/streamflow/ffc_outputs/vista_annual_flow_matrix.csv')
derby = pd.read_csv('data_inputs/streamflow/ffc_outputs/Blw_derby_annual_flow_matrix.csv')
# select year for plotting (input to function)
year = '1959'

def wy_overlay(upstream, downstream, year, up_label, down_label, pt_title):
    up_flow = upstream[year]
    down_flow = downstream[year]
    up_flow = pd.to_numeric(up_flow, errors='coerce')
    down_flow = pd.to_numeric(down_flow, errors='coerce')
    up_flow = up_flow.divide(35.3147)
    down_flow = down_flow.divide(35.3147)

    plt.rc('ytick', labelsize=14) 
    plt.subplot(1,1,1)
    plt.plot(up_flow, color = 'C0', linestyle = '-', label = up_label)
    plt.plot(down_flow, color = '#CC5801', linestyle = '--', label = down_label)
    plt.hlines(y=0, xmin=0, xmax=366, color='black', linewidth=0.5)
    plt.xticks([])
    plt.tick_params(axis='y', which='major', pad=1)
    month_ticks = [0,32,60,91,121,152,182,213,244,274,305,335]
    month_labels = [ 'O', 'N', 'D', 'J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S']
    plt.xticks(month_ticks, month_labels, fontsize=14)
    plt.ylabel('Flow (cms)', fontsize=14)
    plt.ylim(-200/35, 155)
    # plt.title('{}'.format(pt_title), size=10)
    output_name = 'overlay_' + year
    plt.legend()
    # plt.savefig('data_outputs/hydrographs/'+ output_name +'.png')
    plt.clf()
    return

# plot = wy_overlay(vista, derby, '1943', 'Vista', 'Below Derby Dam', 'title')
# for i in range(1918, 2020):
#     year = str(i)
#     plot = wy_overlay(vista, derby, year, 'Vista', 'Below Derby Dam', 'title')

def q_precip_plot():
    downstream = pd.read_csv('data_inputs/streamflow/ffc_outputs/Blw_derby_supplementary_metrics.csv')
    upstream = pd.read_csv('data_inputs/streamflow/ffc_outputs/vista_supplementary_metrics.csv')
    precip = pd.read_csv('data_inputs/annual_precip.csv')
    # Now make plot over POR (start at 1960?) with precip
    upstream_avg = upstream.iloc[0,:][54:]
    upstream_avg = pd.to_numeric(upstream_avg, errors='coerce')
    upstream_avg = upstream_avg.divide(35.3147)
    downstream_avg = downstream.iloc[0,:][43:]
    downstream_avg = pd.to_numeric(downstream_avg, errors='coerce')
    downstream_avg = downstream_avg.divide(35.3147)
    precip = precip.set_index('year')
    precip = precip.iloc[64:]
    precip.index = downstream_avg.index
    fig, ax1 = plt.subplots()

    ax1.bar(precip.index, precip['annual_precip'], color='#bfe6ff', alpha=0.75)
    ax2 = ax1.twinx()
    ax2.plot(upstream_avg, linestyle='-', color='blue', label='Upstream')
    ax2.plot(downstream_avg, linestyle='-', color='orange', label='Downstream')
    ax1.set_ylabel('Annual precipitation, mm', fontsize=12)
    ax2.set_ylabel('Average annual flow, cms', fontsize=12)
    ax1.set_xticks([0, 10, 20, 30, 40, 50, 60])
    ax1.set_xticklabels(['1960','1970','1980','1990', '2000', '2010', '2020'], fontsize=12)
    ax2.legend(loc='upper left', fontsize=12)
    
    return
# plot  = q_precip_plot()  

def climate_plot():
    # try a second version of climate plot with three facets including temperature
    temp = pd.read_csv('data_inputs/annual_mean_temp.csv')
    precip = pd.read_csv('data_inputs/annual_precip.csv')
    downstream = pd.read_csv('data_inputs/streamflow/ffc_outputs/Blw_derby_supplementary_metrics.csv')
    upstream = pd.read_csv('data_inputs/streamflow/ffc_outputs/vista_supplementary_metrics.csv')
    # Align all data inputs, try 1950 as starting date
    upstream_avg = upstream.iloc[0,:][44:]
    upstream_avg = pd.to_numeric(upstream_avg, errors='coerce')
    upstream_avg = upstream_avg.divide(35.3147)
    downstream_avg = downstream.iloc[0,:][33:]
    downstream_avg = pd.to_numeric(downstream_avg, errors='coerce')
    downstream_avg = downstream_avg.divide(35.3147)

    precip = precip.set_index('year')
    temp = temp.set_index('year')
    precip = precip.iloc[54:]
    temp = temp.iloc[54:]
    downstream_avg.index = precip.index
    upstream_avg.index = precip.index
    plot_df =  pd.merge(precip, temp, left_index=True, right_index=True)
    plot_df2 = pd.merge(upstream_avg, downstream_avg, left_index=True, right_index=True)
    plot_df = pd.merge(plot_df, plot_df2,  left_index=True, right_index=True)
    plot_df.columns = ['annual_precip', 'annual_temp', 'upstream', 'downstream']
    # stacked plots
    fig = plt.figure(figsize=(8,6))
    # set height ratios for subplots
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1])
    # the first subplot
    ax0 = plt.subplot(gs[0])
    # log scale for axis Y of the first subplot
    line0, = ax0.plot(plot_df['annual_temp'], color='r', label='Annual mean temperature')
    ax1 = plt.subplot(gs[1], sharex = ax0)
    ax2 = ax1.twinx()
    ax1.bar(precip.index, precip['annual_precip'], color='#bfe6ff', alpha=0.75)
    line1, = ax2.plot(plot_df['upstream'], linestyle='-', color='blue', label='Upstream flow')
    line2, = ax2.plot(plot_df['downstream'], linestyle='-', color='purple', label='Downstream flow')
    plt.setp(ax0.get_xticklabels(), visible=False)
    yticks = ax1.yaxis.get_major_ticks()
    yticks[-1].label1.set_visible(False)
    ax0.set_ylabel('Temperature ($^\circ$C)', fontsize=12)
    ax1.set_ylabel('Precipitation (mm)', fontsize=12)
    ax2.set_ylabel('Flow (cms)', fontsize=12)
    ax2.legend(loc='upper right', fontsize=12)
    ax0.legend(loc='upper left', fontsize=12)
    plt.savefig('data_outputs/climate.png', dpi=1200)
    pdb.set_trace()
    return
# output = climate_plot()

def bai_plot():
    #import bai lines, new csv
    precip = pd.read_csv('data_inputs/annual_precip.csv')
    precip = precip.set_index('year')
    precip = precip.iloc[34:]

    all_sites = pd.read_csv('data_inputs/site_avg_bai/All_bai_site_avg.csv')
    # Try this with standardized tree growth instead
    bai_dict = {}    
    raw_sites = glob.glob('data_inputs/all_trees_bai/*.csv')
    for site in raw_sites:
        data = pd.read_csv(site)
        data = data.set_index('year')
        name = site.split('/')[2][:-4]
        bai_dict[name] = data
    bai_dict['MRW_bai'] = bai_dict['MRW_bai'].drop('MRW_isotopes', axis=1)
    bai_dict['R_bai'] = bai_dict['R_bai'].drop('R_isotopes', axis=1)
    bai_dict['N_bai'] = bai_dict['N_bai'].drop('N_isotopes', axis=1)

    def standardize(col):
        col_mean = np.nanmean(col)
        sd = np.nanstd(col)
        return (col-col_mean)/sd

    for site in bai_dict.keys():
        # check if you can do an apply function to every row to standardize instead of a dumb loop
        bai_dict[site] = bai_dict[site].apply(standardize)
        bai_dict[site] = bai_dict[site].mean(axis=1)
        bai_dict[site].name = site
    
    # take average of upstream and downstream sites
    upstream = pd.merge(bai_dict['MRW_bai'], bai_dict['MRE_bai'], how='left',  left_index=True, right_index=True)
    upstream = pd.merge(upstream, bai_dict['R_bai'], how='left', left_index=True, right_index=True)
    downstream = pd.merge(bai_dict['N_bai'], bai_dict['BB_bai'], how='left', left_index=True, right_index=True)
    upstream = upstream.mean(axis=1)
    downstream = downstream.mean(axis=1)  

    #make pretty, label, output
    fig, ax1 = plt.subplots(figsize=(9,4))
    ax1.bar(precip.index, precip['annual_precip'], color='#bfe6ff', alpha=0.75, label='Annual precipitation')
    ax1.set_ylabel('Annual precipitation (mm)')
    ax2 = ax1.twinx()   
    ax2.plot(downstream[33:], linestyle='-', color='purple', label='Downstream sites', linewidth='2.5')
    ax2.plot(upstream, linestyle='-', color='blue', label='Upstream sites', linewidth='2.5')
    ax2.set_ylabel('Basal area increment, standardized')
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    lines = lines_1 + lines_2
    labels = labels_1 + labels_2
    
    ax1.legend(lines, labels, loc='upper left')
    plt.savefig('data_outputs/bai_allsites.png', dpi=1200)
    plt.show()
    return
# output = bai_plot()

def r102_plot():
    isotopes = pd.read_csv('data_inputs/isotopes.csv')
    isotopes = isotopes.drop(['N', 'Rpooled', 'MRW'], axis=1)
    isotopes = isotopes.dropna()
    isotopes = isotopes.set_index('Year')
    r2 = isotopes['R2']
    r16 = isotopes['R16']
    plt.rcParams.update({'font.size': 16})
    plt.subplots(figsize=(9,4))
    plt.plot(r2, color='#301934', label='farther from channel', linewidth=2)
    plt.plot(r16, color='#62baac', label='near channel', linewidth=2)
    plt.axvline(1982, color='black', linestyle='dotted')
    plt.legend()
    plt.ylabel(r'$\Delta^{13}$'+'C (%)')
    plt.savefig('data_outputs/R102_isotopes.png', dpi=1200)
    plt.show()
    return 
# plot = r102_plot()

def n_plot():
    plot_d = pd.read_csv('data_inputs/N_plotting.csv')
    plot_d = plot_d.set_index('Year')
    n_bai = plot_d['N_bai_pooled']
    n_iso = plot_d['N_DC13']
    plt.rcParams.update({'font.size': 16})
    fig,ax = plt.subplots(figsize=(12,4))
    ax.plot(n_bai, color='#301934', label='Basal area increment')
    ax.set_ylabel('Basal area increment '+ r'$(mm^2)$')
    ax.set_ylim(0,17500)
    ax.legend(loc=1)
    ax2 = ax.twinx()
    ax2.plot(n_iso, color='#62baac', label=r'$\Delta^{13}$'+'C %')
    ax2.set_ylim(18, 22)
    ax2.set_ylabel(r'$\Delta^{13}$'+'C (%)')
    plt.axvline(1973, color='black', linestyle='dotted')
    ax2.legend(loc=2)
    plt.savefig('data_outputs/n_plot.png', dpi=1200)
    plt.show()
    
# plot = n_plot()

def model_sig():
    data = pd.read_csv('data_inputs/model_sig_plotter.csv')
    # first plot without random effects
    data = data.drop(axis=0, index=0)
    plt.rcParams.update({'font.size': 14})
    fig, ax = plt.subplots(figsize = (15, 7))
    ax.barh(data['Model'], data['Average'], color='teal')
    ax.set_xlabel('Number of significant model coefficients')
    fig.tight_layout()
    plt.show()
    return
# plot = model_sig()

def posterior_density():
    # bring in mcmc csv's
    mcmc = pd.read_csv('data_inputs/model_outputs/mcmc_samples/mcmc_samples_all_params_sp_precip.csv', index_col=0)

    # decide which params to plot. May need to average some cols. start w avg annual Q by reg, for downstream sites (avg'd?)
    mcmc_cols = mcmc.columns
    bb_avg_pre = []
    bb_avg_post = []
    mre_avg_pre = []
    mre_avg_post = []
    mrw_avg_pre = []
    mrw_avg_post = []
    n_avg_pre = []
    n_avg_post = []
    r_avg_pre = []
    r_avg_post = []
    metric = 'bprecip' # dry is bdm50
    for mc_col in mcmc_cols:
        # make col number an int
        nums = re.findall(r'\d+', mc_col)#[1:3] # [1:3] for dry season only
        name = re.findall('[a-z]+', mc_col)

        # name = [mc_col[0:5]] # for dry season only

        if name[0] == metric:
            if int(nums[0]) < 11: # bb indices
                if int(nums[1]) == 1:
                    bb_avg_post.append(mcmc[mc_col])
                elif int(nums[1]) == 2:
                    bb_avg_pre.append(mcmc[mc_col])
            elif int(nums[0]) > 10 and int(nums[0]) < 30: # mre indices
                if int(nums[1]) == 1:
                    mre_avg_post.append(mcmc[mc_col])
                elif int(nums[1]) == 2:
                    mre_avg_pre.append(mcmc[mc_col])
            elif int(nums[0]) > 29 and int(nums[0]) < 49: # mrw indices
                if int(nums[1]) == 1:
                    mrw_avg_post.append(mcmc[mc_col])
                elif int(nums[1]) == 2:
                    mrw_avg_pre.append(mcmc[mc_col])
            elif int(nums[0]) > 48 and int(nums[0]) < 69: # n indices
                if int(nums[1]) == 1:
                    n_avg_post.append(mcmc[mc_col])
                elif int(nums[1]) == 2:
                    n_avg_pre.append(mcmc[mc_col])
            elif int(nums[0]) > 68: # r indices
                if int(nums[1]) == 1:
                    r_avg_post.append(mcmc[mc_col])
                elif int(nums[1]) == 2:
                    r_avg_pre.append(mcmc[mc_col])
    # Concat all indiv trees at a site into one long df (retains more info than averaging)
    bb_avg_pre = pd.concat(bb_avg_pre, axis=0).to_frame()
    bb_avg_post = pd.concat(bb_avg_post, axis=0).to_frame()
    mre_avg_pre = pd.concat(mre_avg_pre, axis=0).to_frame()
    mre_avg_post = pd.concat(mre_avg_post, axis=0).to_frame()
    mrw_avg_pre = pd.concat(mrw_avg_pre, axis=0).to_frame()
    mrw_avg_post = pd.concat(mrw_avg_post, axis=0).to_frame()
    n_avg_pre = pd.concat(n_avg_pre, axis=0).to_frame()
    n_avg_post = pd.concat(n_avg_post, axis=0).to_frame()
    r_avg_pre = pd.concat(r_avg_pre, axis=0).to_frame()
    r_avg_post = pd.concat(r_avg_post, axis=0).to_frame()

    plt.rcParams.update({'font.size': 14})
    
    def plotter(data, d_label, color):
        sns.distplot(data, hist = False, kde = True,
                    kde_kws = {'shade': True, 'linewidth': 2}, 
                    label = d_label,
                    color=color)
    plot_inputs = [[bb_avg_pre, 'Historic (1918-1972)', '#663399'], [bb_avg_post, 'Modern (1973-2019)','#62baac']]

    # plot_inputs = [[mrw_avg_pre, 'US-1', '#62baac'],[mre_avg_pre, 'US-2','#005248'],[r_avg_pre, 'US-3','#A9A9A9'],[bb_avg_pre, 'DS-1', '#cf92ff'],[n_avg_pre, 'DS-2', '#663399']]
    for input in plot_inputs:
        plotter(input[0], input[1], input[2])

    plt.legend(prop={'size': 11}, loc='upper right')
    # plt.figure(figsize=(8,6))
    # plt.title(metric)
    plt.title('Site DS-1: Growing Season Precipitation', fontsize=13) #Spring Recession Rate of Change: Historic (1918-1973), Site DS-1: Dry Season Median Magnitude
    plt.xlabel('Coefficient value')
    plt.ylabel('Posterior density')
    plt.yticks(np.arange(0, 4.5, 0.5))
    plt.xlim((-1.5,1.5))
    plt.axvline(x=0, linestyle='--', color='black', linewidth='0.5')
    # plt.show()
    plt.savefig('data_outputs/density_plots/density_bb_prepost_'+ metric +'.jpeg', dpi=1200)
    
    plt.clf()
    fig, ax = plt.subplots()
    plot_inputs = [[n_avg_pre, 'Historic (1918-1972)', '#663399'], [n_avg_post, 'Modern (1973-2019)', '#62baac']]
    # plot_inputs = [[mrw_avg_post, 'US-1', '#62baac'],[mre_avg_post, 'US-2','#005248'],[r_avg_post, 'US-3','#A9A9A9'],[bb_avg_post, 'DS-1', '#cf92ff'],[n_avg_post, 'DS-2', '#663399']]
    for input in plot_inputs:
        plotter(input[0], input[1], input[2])

    plt.legend(prop={'size': 11}, loc='upper right')
    # plt.figure(figsize=(8,6))
    # plt.title(metric)
    plt.title('Site DS-2: Growing Season Precipitation', fontsize=13)
    plt.xlabel('Coefficient value')
    plt.ylabel('Posterior density')
    plt.yticks(np.arange(0, 4.5, 0.5))
    plt.xlim((-1.5,1.5))
    plt.axvline(x=0, linestyle='--', color='black', linewidth='0.5')
    # plt.show()
    plt.savefig('data_outputs/density_plots/density_n_'+ metric +'.jpeg', dpi=1200)
    # # pdb.set_trace()
    return
outout = posterior_density()

def func_flow_hydro(vista):
    # pull out water year from vista to plot (try 2006, maybe 1995?)
    plt_yr = vista['2006']
    plt_yr = pd.to_numeric(plt_yr, errors='coerce').divide(35.3147)
    plt.rc('ytick', labelsize=14) 
    plt.figure(figsize=(10,2))
    
    plt.plot(plt_yr, color = 'C0', linestyle = '-')
    
    fall = plt_yr[55:68]
    wet = plt_yr[75:232]
    peak = plt_yr[90:99]
    spring = plt_yr[232:295]
    dry = plt_yr[295:365]
    dry2 = plt_yr[0:55]
    plt.plot(fall, color='#FBB117', linestyle = '-', linewidth=2)
    plt.plot(wet, color='#3895D3', linestyle = '-', linewidth=2)
    plt.plot(peak, color='#072F5F', linestyle = '-', linewidth=2) 
    plt.plot(spring, color='#249225', linestyle = '-', linewidth=2)
    plt.plot(dry, color='#cc1100', linestyle = '-', linewidth=2)
    plt.plot(dry2, color='#cc1100', linestyle = '-', linewidth=2)
    plt.savefig("data_outputs/hydro_rainbow.png", dpi=1200, transparent=True)

    plt.clf()
    plt.figure(figsize=(10,6))
    vista_25 = []
    vista_50 = []
    vista_75 = []
    for row_index, value in enumerate(vista.iloc[:,0]): # loop through each row, 366 total
        data = pd.to_numeric(vista.iloc[row_index, :], errors='coerce')
        vista_25.append(np.nanpercentile(data, 25))
        vista_50.append(np.nanpercentile(data, 50))
        vista_75.append(np.nanpercentile(data, 75))
    # Try with full POR and color coding...
    plt.plot(vista_50, color='#072F5F', linestyle = '-', linewidth=2) 
    x = np.arange(0, 366, 1)
    plt.fill_between(x, vista_25, vista_50, color='#3895D3', alpha=.5)
    plt.fill_between(x, vista_50, vista_75, color='#3895D3', alpha=.5)
    plt.savefig("data_outputs/vista_rh.png", dpi=1200, transparent=True)
    return
# output = func_flow_hydro(vista)