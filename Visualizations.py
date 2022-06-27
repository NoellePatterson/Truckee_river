import pandas as pd
import matplotlib.pyplot as plt
import re
import seaborn as sns
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

def bai_plot():
    #import bai lines, new csv
    precip = pd.read_csv('data_inputs/annual_precip.csv')
    precip = precip.set_index('year')
    precip = precip.iloc[64:]

    all_sites = pd.read_csv('data_inputs/site_avg_bai/All_bai_site_avg.csv')
    all_sites = all_sites.set_index('year')
    # constrain years to 1960-2019
    all_sites = all_sites.iloc[63:]
    us_plot = pd.to_numeric(all_sites['mrw_r_avg'], errors='coerce')
    us_plot = us_plot.iloc[3:] # cut off first three years in 1960 bc of lack of enough data
    ds_plot = pd.to_numeric(all_sites['DS_avg'], errors='coerce')
    # pdb.set_trace()
    #make pretty, label, output
    fig, ax1 = plt.subplots()
    ax1.plot(ds_plot, linestyle='-', color='#CC5801', label='Downstream sites', linewidth='3')
    ax1.plot(us_plot, linestyle='-', color='C0', label='Upstream sites', linewidth='3')
    ax1.ylabel('Basal area increment '+ r'$(mm^2)$')
    ax1.legend()
    ax2 = ax1.twinx()
    ax2.bar(precip.index, precip['annual_precip'], color='#bfe6ff', alpha=0.75)
    ax2.set_ylabel('Annual precipitation, mm')
    ax2.legend(loc='upper left')
    plt.show()
    return
output = bai_plot()

def r102_plot():
    isotopes = pd.read_csv('data_inputs/isotopes.csv')
    isotopes = isotopes.drop(['N', 'Rpooled', 'MRW'], axis=1)
    isotopes = isotopes.dropna()
    isotopes = isotopes.set_index('Year')
    r2 = isotopes['R2']
    r16 = isotopes['R16']
    plt.subplot(1,1,1)
    plt.plot(isotopes['R2'], color='red', label='far from channel')
    plt.plot(isotopes['R16'], color='orange', label='near channel')
    plt.axvline(1982, color='black', linestyle='dotted')
    plt.legend()
    plt.ylabel(r'$\delta^{13}$'+'C (%)')
    plt.show()
    import pdb; pdb.set_trace()
    return 
# plot = r102_plot()

def n_plot():
    plot_d = pd.read_csv('data_inputs/N_plotting.csv')
    plot_d = plot_d.set_index('Year')
    n_bai = plot_d['N_bai_pooled']
    n_iso = plot_d['N_abs']
    fig,ax = plt.subplots(figsize=(12,4))
    ax.plot(n_bai, color='red', label='Basal area increment')
    ax.set_ylabel('Basal area '+ r'$(mm^2)$')
    ax.legend(loc=2)
    ax2 = ax.twinx()
    ax2.plot(n_iso, color='blue', label=r'$\delta^{13}$'+'C % (absolute value)')
    ax2.set_ylabel(r'$\delta^{13}$'+'C (%)')
    plt.axvline(1973, color='black', linestyle='dotted')
    ax2.legend(loc=1)
    plt.show()
# plot = n_plot()

def posterior_density():
    # bring in mcmc csv's
    mcmc = pd.read_csv('data_inputs/mcmc_samples_notemp.csv', index_col=0)

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
    metric = 'bprecip'
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

    
    # plot_df = pd.concat([bb_avg_pre, bb_avg_post, n_avg_pre, n_avg_post], axis=0)
    # plot_df.columns = ['param', 'label']
    # pdb.set_trace()
    # plot density plts
    # fig = sns.distplot(plot_df, x='param', hue='label', fill=True, kind='kde')
    def plotter(data, d_label):
        sns.distplot(data, hist = False, kde = True,
                    kde_kws = {'shade': True, 'linewidth': 2}, 
                    label = d_label)
    # plot_inputs = [[bb_avg_pre, 'Site #4 1973-2019'], [mre_avg_pre, 'Site #2 1973-2019'], [mrw_avg_pre, 'Site #1 1973-2019'],
    # [n_avg_pre, 'Site #5 1973-2019'], [r_avg_pre, 'Site #3 1973-2019']]
    plot_inputs = [[bb_avg_pre, 'Site #4 1918?-1973'], [bb_avg_post, 'Site #4 1973-2019']]
    for input in plot_inputs:
        plotter(input[0], input[1])

    plt.legend(prop={'size': 10}, loc='upper right')
    # plt.figure(figsize=(8,6))
    plt.title(metric)
    plt.xlabel('Coefficient value')
    plt.ylabel('Posterior density')
    # fig = sns.displot(bb_avg_pre, kind='kde', fill=True)
    # fig = sns.displot(bb_avg_post, kind='kde', fill=True)
    # plt.show()
    plt.savefig('data_outputs/density_plots/density_bb_notempmodel'+ metric +'.jpeg', dpi=300)
    # pdb.set_trace()
    return
# outout = posterior_density()