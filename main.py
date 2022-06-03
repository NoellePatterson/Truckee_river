from utils import huge_format, prism_spring_vals
import warnings
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


# output = prism_spring_vals()
# output = huge_format()

prism_monthly = pd.read_csv('data_inputs/prism_averaged_grids.csv')
dates = prism_monthly['date']
prism_monthly['year'] = np.nan * len(dates)
prism_monthly['month'] = np.nan * len(dates)
for index, row in prism_monthly.iterrows():
    prism_monthly.loc[index, 'year'] = datetime.strptime(row['date'], '%Y-%m').year
    prism_monthly.loc[index, 'month'] = datetime.strptime(row['date'], '%Y-%m').month
years = np.unique(prism_monthly['year'])

precip_cum = {}
for year_index, year in enumerate(years):
    precip_cum[str(int(year))] = []

# populate cumulative spring precip dict with monthly data
for index, row in prism_monthly.iterrows():
    if row['month'] == 10 or row['month'] == 11 or row['month'] == 12: # start indexing at 10 for Oct-start water year
        precip_cum[str(int(row['year'])+1)].append(row['ppt (mm)'])
    else:
        # import pdb; pdb.set_trace()
        precip_cum[str(int(row['year']))].append(row['ppt (mm)'])
precip_output = []
for year in years:
    precip_output.append(np.sum(precip_cum[str(int(year))]))
precip_output = pd.DataFrame(list(zip(years,precip_output,)),columns =['year','annual_precip'])
precip_output.to_csv('data_inputs/annual_precip.csv', index=False)
