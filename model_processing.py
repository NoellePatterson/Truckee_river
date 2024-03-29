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

        bbpre_rows = dat[(dat['site']=='BB') & (dat['reg_period']=='pre')]
        bbpost_rows = dat[(dat['site']=='BB') & (dat['reg_period']=='post')]
        mrepre_rows = dat[(dat['site']=='MRE') & (dat['reg_period']=='pre')]
        mrepost_rows = dat[(dat['site']=='MRE') & (dat['reg_period']=='post')]
        mrwpre_rows = dat[(dat['site']=='MRW') & (dat['reg_period']=='pre')]
        mrwpost_rows = dat[(dat['site']=='MRW') & (dat['reg_period']=='post')]
        npre_rows = dat[(dat['site']=='N') & (dat['reg_period']=='pre')]
        npost_rows = dat[(dat['site']=='N') & (dat['reg_period']=='post')]
        rpre_rows = dat[(dat['site']=='R') & (dat['reg_period']=='pre')]
        rpost_rows = dat[(dat['site']=='R') & (dat['reg_period']=='post')]

        data_subsets = [['BBpre', 0, bbpre_rows], ['MREpre', 0, mrepre_rows], ['MRWpre', 0, mrwpre_rows], ['Npre', 0, npre_rows], ['Rpre', 0, rpre_rows],
        ['BBpost', 0, bbpost_rows], ['MREpost', 0, mrepost_rows], ['MRWpost', 0, mrwpost_rows], ['Npost', 0, npost_rows], ['Rpost', 0, rpost_rows]]
        data_subsets_pre = data_subsets[0:5]
        data_subsets_post = data_subsets[5:10]

        def data_counting(data_subset):
            for index, row in data_subset[2].iterrows():
                if row['significance'] == True:
                    data_subset[1] += 1
            return
        
        for data_subset in data_subsets:
            counts = data_counting(data_subset)
        
        pre_sig = 0
        for data_subset in data_subsets_pre:
            for index, row in data_subset[2].iterrows():
                if row['significance'] == True:
                    pre_sig += 1

        post_sig = 0
        for data_subset in data_subsets_post:
            for index, row in data_subset[2].iterrows():
                if row['significance'] == True:
                    post_sig += 1
        
        # Build output table: convert counts into percentages
        all_outputs.append([metric_name, pre_sig, post_sig, data_subsets[1][1], data_subsets[2][1], data_subsets[4][1], data_subsets[0][1], data_subsets[3][1], data_subsets[6][1], data_subsets[7][1], data_subsets[9][1], data_subsets[5][1], data_subsets[8][1]])
    df = pd.DataFrame(all_outputs, columns=['metric', 'pre_sig', 'post_sig', 'mre_pre_sig_19', 'mrw_pre_sig_19', 'r_pre_sig_14', 'bb_pre_sig_10', 'n_pre_sig_20', 'mre_post_sig_19', 'mrw_post_sig_19', 'r_post_sig_14', 'bb_post_sig_10', 'n_post_sig_20'])
    df.to_csv('data_outputs/Bayes_model_summaries/bayes_model_siteprepost_summary'+key+'_8_09.csv')

