# Interpret and visualize outputs from several model runs testing different predictors of cottonwood growth. 

import glob
import pandas as pd
import numpy as np
import pdb

# Match outputs to site #, tree #, regulation type
# For each model output table, tally up the total number of Sig outputs by site and regulation period. 
# (For now) - make a table for each model, with sig results split by regulation and sites. 

models_loc = glob.glob('data_inputs/model_outputs/*.csv')
models_dict = {}
for model in models_loc:
    model_dat = pd.read_csv(model)
    name = model.split('/')[2][:-4]
    models_dict[name] = model_dat

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

reg_mapping = {"1":"post", "2":"pre"}

for key in models_dict:
    model_dat = models_dict[key]
    metrics = []
    sites = []
    trees = []
    reg_period = []
    for index, row in enumerate(model_dat.iloc[:,0]):
        metrics.append(row.split('.')[0])
        sites.append(tree_site_mapping[row.split('.')[1]])
        trees.append(indiv_tree_mapping[row.split('.')[1]])
        reg_period.append(reg_mapping[row.split('.')[2]])
    model_dat['metric'] = metrics
    model_dat['site'] = sites
    model_dat['tree'] = trees
    model_dat['reg_period'] = reg_period


    # Get list of unique metrics, track which row numbers correspond to it from dataset and create a subset table,
    # loop through that table tallying up all the counts, then save all those counts with the associated metric name.
    # And THEN you can create the final output table. 
    all_outputs = []
    metric_ls = np.unique(metrics)
    #create subsets of data containing each metric separately. 
    metric_subsets = []
    for current_metric in metric_ls:
        data = model_dat[model_dat['metric']==current_metric]
        tempdict = {current_metric:data}
        metric_subsets.append(tempdict)

    for metric_subset in metric_subsets:
        metric_name = list(metric_subset.items())[0][0]
        dat = list(metric_subset.items())[0][1]

        bb_rows = dat[dat['site']=='BB']
        mre_rows = dat[dat['site']=='MRE']
        mrw_rows = dat[dat['site']=='MRW']
        n_rows = dat[dat['site']=='N']
        r_rows = dat[dat['site']=='R']
        data_subsets = [['BB', bb_rows], ['MRE', mre_rows], ['MRW', mrw_rows], ['N', n_rows], ['R', r_rows]]
        pre_sig = 0
        post_sig = 0
        bb_sig = 0
        mre_sig = 0
        mrw_sig = 0
        n_sig = 0
        r_sig = 0

        for site in data_subsets:
            total_count = 0
            check = 0
            for index, row in site[1].iterrows(): # loop through again, counting up significance per metric
                if row['significance'] == True:
                    if row['reg_period'] == 'post':
                        post_sig += 1
                    elif row['reg_period'] == 'pre':
                        pre_sig += 1
            for index, row in site[1].iterrows(): # loop through again, counting up significance per metric
                total_count += 1
                if row['significance'] == True:
                    check += 1
                    if row['site'] == 'BB':
                        bb_sig += 1
                    elif row['site'] == 'MRE':
                        mre_sig += 1
                    elif row['site'] == 'MRW':
                        mrw_sig += 1
                    elif row['site'] == 'R':
                        r_sig += 1
                    elif row['site'] == 'N':
                        n_sig += 1
        # Build output table: convert counts into percentages
        all_outputs.append([metric_name, pre_sig, post_sig, mre_sig, mrw_sig, r_sig, bb_sig, n_sig])
    df = pd.DataFrame(all_outputs, columns=['metric', 'pre_sig', 'post_sig', 'mre_sig_19', 'mrw_sig_19', 'r_sig_14', 'bb_sig_10', 'n_sig_20'])
    df.to_csv('data_outputs/bayes_model_summary'+key+'.csv')


