# Analysis required to address major revisions in manuscript
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import plotly.express as px
import pdb

# Juvenile effect analysis

# bring in tree age file
tree_age = pd.read_csv('data_outputs/tree_ages.csv') # generated from tree_age.py script
tree_age = tree_age.set_index('Tree_id')
sites = ['BB', 'MRE', 'MRW', 'N', 'R']

# for each site:
for site in sites:
    # bring in tree BAIs 
    bais = pd.read_csv('data_inputs/all_trees_bai/{}_bai.csv'.format(site))
    bais = bais.set_index('year')
    # set up plot and add each tree as a new line
    tree_ids = bais.columns
    plt.subplots(figsize=(12,4))
    plt.ylabel('BAI (mm2)')
    plt.title('Site {} BAI'.format(site))
    for tree_id in tree_ids:
        if len(tree_id) > 7: # some isotope data mixed in with bai, pass it using id len as a proxy
            continue
        dbh_cm = tree_age.loc[tree_id, 'final_age_estimate'] 
        # plot year vs. BAI, black line
        plt.plot(bais[tree_id], color='grey')
        # go back, check for juvenile phase. If missing rings < 25, juv = 25 - missing ring#
        missing_rings = tree_age.loc[tree_id, 'missing_rings'] 
        juv_phase = 25 - missing_rings
        if juv_phase > 0:
            juv_phase = int(juv_phase)
            bai = bais[tree_id].dropna()
            plt.plot(bai[0:juv_phase], color='red')
    plt.savefig('data_outputs/growth_plots/site_{}'.format(site), dpi=400)

# When all trees finished up, save plot. interactive? Or just big. 