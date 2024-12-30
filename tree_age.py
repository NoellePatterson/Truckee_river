# Revisions analysis 2024
# Estimate tree age via missing core based on both radial increment and BAI. 
# Calculate per site, pulling in necessary data. 

import numpy as np
import pandas as pd
import pdb

# Bring in new meta file, tree_age which has DBH
tree_age = pd.read_csv('data_inputs/Tree_age.csv')
tree_age = tree_age.set_index('Tree_id')
tree_age['ring_count'] = ''
tree_age['radial_estimate_age'] = ''
tree_age['bai_estimate_age'] = ''
tree_age['final_age_estimate'] = ''
sites = ['BB', 'MRE', 'MRW', 'N', 'R']

for site in sites:
    # Bring in radial increments
    radials = pd.read_csv('data_inputs/Final_radial/Site_{}.csv'.format(site))
    radials = radials.set_index('year')
    # Bring in BAI measurements
    bais = pd.read_csv('data_inputs/all_trees_bai/{}_bai.csv'.format(site))
    tree_ids = radials.columns
    # For each tree in site: # BB1 for tree_id in tree_ids:
    # ID the DBH from meta file, convert to radius
    for tree_id in tree_ids:
        dbh_cm = tree_age.loc[tree_id, 'Dbh_cm'] 
        dbh_mm = dbh_cm*10 # convert cm to mm
        radius_mm = dbh_mm/2
        # Calc bark from regression, calc bark BAI and increment. Bark is a one-sided length (radius not diameter)
        bark_len_mm = (dbh_cm*0.0535 - 0.4353)*10
        # ID total num of rings measured (min age count) - tree timeseries
        age_baseline = len(radials[tree_id].dropna())
        # ID total core length from from radial measurements total (to correct for angle offset) - final radial
        core_len_mm = np.nansum(radials[tree_id])
        # Calc avg radial increment len
        ave_radial_mm = np.nanmean(radials[tree_id])
        # Estimate # remaining rings, radial method, add to age file
        len_diff = radius_mm - bark_len_mm - core_len_mm
        if len_diff < 0:
            add_yrs_rad = age_baseline
        else:
            add_yrs_rad = round(len_diff/ave_radial_mm)

        # Calc total basal area from DBH, in mm2 - tree timeseries
        bai_tot = np.pi * ((dbh_mm - bark_len_mm*2)/2)**2
        # Add up basal increments to calc difference (remove negative vals)
        bai_inc_vals = bais[tree_id]
        for index, val in enumerate(bai_inc_vals):
            if val < 0:
                bai_inc_vals = bai_inc_vals.drop(index)
        bai_inc = np.nansum(bai_inc_vals)
        # Calc avg basal increment
        ave_bai_mm2 = np.nanmean(bais[tree_id])
        # estimate # remaining basal increments, BAI method, add to age file
        bai_diff = bai_tot - bai_inc
        if bai_diff < 0:
            add_yrs_bai = age_baseline
        else:
            add_yrs_bai = round(bai_diff/ave_bai_mm2)

        tree_age.loc[tree_id, 'ring_count'] = age_baseline
        tree_age.loc[tree_id, 'radial_estimate_age'] = age_baseline + add_yrs_rad
        tree_age.loc[tree_id, 'bai_estimate_age'] = age_baseline + add_yrs_bai
        tree_age.loc[tree_id, 'final_age_estimate'] = np.floor((age_baseline + add_yrs_bai + age_baseline + add_yrs_rad)/2)
tree_age.to_csv('data_outputs/tree_ages.csv')
pdb.set_trace()


