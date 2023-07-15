import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import pdb
import re

# plot tree random effect param values agains distance from the channel. 
# Also plot out all random effects. 
# RE avg vals: all_params_summary.csv
# channel proximity: All_tree_metadata.csv

# Need to line up the model output tree ID's with the ID's in tree metadata. 
# Here it is below, try to keep it simple. 

model_dat = pd.read_csv('data_inputs/model_outputs/all_params.csv')

bb = ('BB', 1, 11)
mre = ('MRE', 11, 29)
mrw = ('MRW', 30, 48)
n = ('N', 49, 68)
r = ('R', 69, 82)

maps = [bb, mre, mrw, n, r]
tree_site_mapping = {}
def mapper(sitemap):
    for step in range(sitemap[1], sitemap[2]+1):
        tree_site_mapping[str(step)] = sitemap[0]
for site_map in maps:
    output = mapper(site_map)

indiv_tree_mapping = {"1":"BB1", "2":"BB11", "3":"BB14", "4":"BB15", "5":"BB18", "6":"BB2",  
"7":"BB3", "8":"BB4", "9":"BB5", "10":"BB6", "11":"MRE1", "12":"MRE10",
"13":"MRE11", "14":"MRE13", "15":"MRE14", "16":"MRE15", "17":"MRE16", "18":"MRE17",
"19":"MRE18", "20":"MRE19", "21":"MRE2", "22":"MRE20", "23":"MRE3", "24":"MRE4", 
"25":"MRE5", "26":"MRE6", "27":"MRE7", "28":"MRE8", "29":"MRE9", "30":"MRW1", 
"31":"MRW11", "32":"MRW13", "33":"MRW14", "34":"MRW15", "35":"MRW16", "36":"MRW17",
"37":"MRW18", "38":"MRW19", "39":"MRW20", "40":"MRW21", "41":"MRW22", "42":"MRW23",
"43":"MRW4", "44":"MRW5", "45":"MRW6", "46":"MRW7", "47":"MRW8",  "48":"MRW9", 
"49":"N1", "50":"N10", "51":"N11", "52":"N12", "53":"N13",  "54":"N14",  
"55":"N15", "56":"N16", "57":"N17", "58":"N18", "59":"N19", "60":"N2",   
"61":"N20", "62":"N3", "63":"N4", "64":"N5", "65":"N6", "66":"N7",   
"67":"N8", "68":"N9", "69":"R1", "70":"R10", "71":"R12", "72":"R13",  
"73":"R14", "74":"R15", "75":"R16", "76":"R17", "77":"R18", "78":"R19",  
"79":"R2", "80":"R20", "81":"R6", "82":"R7"}

tree_dat = {}
tree_avg = []
tree_2p5 = []
tree_97p5 = [] 
for col in model_dat.columns:
    if col.find('alphaT') != -1:
        id = re.findall(r'\d+', col)[0]
        for key in indiv_tree_mapping.keys():
            if key == id:
                tree_name = indiv_tree_mapping[key]
                tree_dat[tree_name] = model_dat[col]
                tree_avg.append(np.nanmean(model_dat[col]))
                tree_2p5.append(np.nanpercentile(model_dat[col], 2.5))
                tree_97p5.append(np.nanpercentile(model_dat[col], 97.5))

re_df = pd.DataFrame(list(zip(tree_dat.keys(), tree_avg, tree_2p5, tree_97p5)), columns=["name", "average", "p_2.5", "p_97.5"])   
# re_df.to_csv("data_outputs/re_plotting.csv", index=False)

# Print out outputs, order them manually in a csv, then read back in to make plot. 

tree_re = pd.read_csv('data_outputs/re_plotting_ordered.csv')
sites_re = pd.read_csv('data_outputs/Site_re_plotter.csv')

site_vals = sites_re['average']
site_x_val = [5, 20, 39, 58.5, 75.5] # plot site sig in middle of tree-level zones
site_y_top = abs(site_vals - sites_re['p_2.5'])
site_y_bot = abs(sites_re['p_97.5'] - site_vals)
site_y_error = [site_y_top, site_y_bot]

site_sig_vals = [site_vals[0], site_vals[1], site_vals[3], site_vals[4]]
site_sig_x = [site_x_val[0], site_x_val[1], site_x_val[3], site_x_val[4]]
site_notsig_vals = site_vals[2]
site_notsig_x = site_x_val[2]

values = tree_re['average']
x_val = range(len(values))

y_top = abs(values - tree_re['p_2.5'])
y_bot = abs(tree_re['p_97.5'] - values)
y_error = [y_top, y_bot]

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True) # ax2 for tree-level, ax1 for site-level
# plt.subplot(figsize=(10,5))
plt.rc('font', size=10) 

sig_vals = []
sig_x = []
notsig_vals = []
notsig_x = []
# Split scatter points by sig/notsig and fill in color on sig points only
for index, val in enumerate(values):
    if tree_re['p_2.5'][index] < 0 and tree_re['p_97.5'][index] > 0:
        notsig_vals.append(values[index])
        notsig_x.append(x_val[index])
    else:
        sig_vals.append(values[index])
        sig_x.append(x_val[index])


# plot sig and not sig circles separately
ax2.errorbar(x_val, values, yerr=y_error, fmt='none', color='darkgrey', zorder=1)
ax2.scatter(sig_x, sig_vals, color='#505050', edgecolor='darkgrey', zorder=2)
ax2.scatter(notsig_x, notsig_vals, color='white', edgecolor='darkgrey', zorder=2)

ax1.errorbar(site_x_val, site_vals, yerr=site_y_error, fmt='none', color='darkgrey', zorder=1)
ax1.scatter(site_sig_x, site_sig_vals, color='#505050', edgecolor='darkgrey', zorder=2)
ax1.scatter(site_notsig_x, site_notsig_vals, color='white', edgecolor='darkgrey', zorder=2)

ax2.axhline(0, color='black', linestyle = '--', linewidth=0.5)
[ax2.axvline(x=i, color="grey", linestyle='--', linewidth=1) for i in [-1.5, 10.5, 29.5, 48.5, 68.5, 82.5]]
ax1.axhline(0, color='black', linestyle = '--', linewidth=0.5)
[ax1.axvline(x=i, color="grey", linestyle='--', linewidth=1) for i in [-1.5, 10.5, 29.5, 48.5, 68.5, 82.5]]

# ax2.grid(axis='x', which='minor', linestyle = '--')
plt.tick_params(bottom = False)
ax1.tick_params('x', length=0)
ax2.set_ylabel('Tree-level significance', fontweight='bold')
ax1.set_ylabel('Site-level significance', fontweight='bold')
plt.tight_layout()
# plt.savefig('data_outputs/rand_eff_summary_1.png', dpi=1200)

# Plot regulation period random effect as separate plot and combine later
re_mod = model_dat['gamma.star[1]']
re_hist = model_dat['gamma.star[2]']

hist_avg = np.nanmean(re_hist)
hist_25 = np.nanpercentile(re_hist, 2.5)
hist_975 = np.nanpercentile(re_hist, 97.5)
hist_error = [abs(hist_avg-hist_25), abs(hist_avg+hist_975)]

mod_avg = np.nanmean(re_mod)
mod_25 = np.nanpercentile(re_mod, 2.5)
mod_975 = np.nanpercentile(re_mod, 97.5)
mod_error = [abs(mod_avg-mod_25), abs(mod_avg+mod_975)]

fig, ax = plt.subplots(1,1)
# plt.figure(figsize=(5,5))
x_cat = ['historic', 'modern']
ax.errorbar([0,0.01], [hist_avg,mod_avg], yerr=[hist_error,mod_error], fmt='none', color='darkgrey', zorder=1)
ax.scatter([0,0.01], [hist_avg,mod_avg], color='#505050', edgecolor='darkgrey', zorder=2)
ax.axhline(0, color='black', linestyle = '--', linewidth=0.5)
plt.xticks([0,0.01], x_cat)
plt.margins(x=0.5, y=0.5)

pdb.set_trace()
