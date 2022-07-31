# process and plot grounwater and surface water elevation data
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, timedelta
from dateutil.parser import parse
import numpy as np
from climata.usgs import InstantValueIO
import pdb

# # Import gage height tot from USGS - only need to do this once
# start_day = pd.to_datetime("2021-01-15") # start: 1/15/21 14:00
# end_day = pd.to_datetime("2022-06-24") # end: 06/24/22 10:15am
# num_days = (end_day-start_day).days
# station_id = '10351650' # "10351650" wadsworth gage
# param_id = "00065"
# # flow is 00060, temperature is 00010, gage height is 00065
# data = InstantValueIO(
#     start_date=start_day,
#     end_date=end_day,
#     station=station_id,
#     parameter=param_id,
# )
# # pdb.set_trace()
# for series in data:
#     dates = [r[0] for r in series.data]
#     date_print = [r[0].strftime("%m-%d-%Y %H:%M") for r in series.data]
#     height_ft = [r[1] for r in series.data]
#     height_m = [r[1]*0.3048 for r in series.data]
# wadsworth_height = {'date':date_print, 'height_m':height_m, 'height_ft':height_ft}
# wadsworth_height = pd.DataFrame(wadsworth_height)

# start_day = pd.to_datetime("2021-01-15") # start: 1/15/21 14:00
# end_day = pd.to_datetime("2022-06-20") # end: 06/20/22 14:45
# num_days = (end_day-start_day).days
# station_id = '10350340' # "10350340" Tracy gage
# param_id = "00065"
# data = InstantValueIO(
#     start_date=start_day,
#     end_date=end_day,
#     station=station_id,
#     parameter=param_id,
# )
# for series in data:
#     dates = [r[0] for r in series.data]
#     date_print = [r[0].strftime("%m-%d-%Y %H:%M") for r in series.data]
#     height_ft = [r[1] for r in series.data]
#     height_m = [r[1]*0.3048 for r in series.data]

# tracy_height = {'date':date_print, 'height_m':height_m, 'height_ft':height_ft}
# tracy_height = pd.DataFrame(tracy_height)

header = ['date_time', 'seconds', 'depth_ft', 'nan']
bb_gw = pd.read_csv('data_inputs/groundwater/Big_bend_gw_corrected.csv', index_col=False, skiprows=[i for i in range(0, 73)], names=header)
mre_gw = pd.read_csv('data_inputs/groundwater/MRE_gw_corrected.csv', index_col=False, skiprows=[i for i in range(0, 74)], names=header)

tracy_height = pd.read_csv('data_inputs/groundwater/tracy_gage_height.csv')
wadsworth_height = pd.read_csv('data_inputs/groundwater/wadsworth_gage_height.csv')

# figure out why data from the MRE logger is unuseable (negative??)
# plt.plot(mre_gw['date_time'], mre_gw['depth_ft'])
# plt.show()

# figure out why data sources aren't aligning over time. The 15-min interval is off in the data loggers?

# figure out an averaging that will line up the two data sources. hourly average?
bb_gw['date_time'] = pd.to_datetime(bb_gw['date_time'], infer_datetime_format=True)
bb_gw = bb_gw.set_index('date_time')
bb_hourly = bb_gw.resample('H').mean()
# clean up well data a little. Replace bad vals (zero or negative) with prior good val
for index, val in enumerate(bb_hourly['depth_ft']):
    if val < 0:
        bb_hourly['depth_ft'][index] =  bb_hourly['depth_ft'][index - 1]

# transform vals so they are in same direction as gage heights 
bb_hourly['depth_ft'] = (4.5 - bb_hourly['depth_ft'])*0.3048

wadsworth_height['date'] = pd.to_datetime(wadsworth_height['date'], infer_datetime_format=True)
wadsworth_hourly = wadsworth_height.set_index('date')
wadsworth_hourly = wadsworth_hourly.resample('H').mean()

ds_merge = pd.merge(bb_hourly, wadsworth_hourly, left_index=True, right_index=True)
plt.plot(ds_merge['height_m'], label='surface water level')
plt.plot(ds_merge['depth_ft'], label='shallow groundwater')
plt.ylabel('water level, meters')
plt.legend()
# plt.show()
ds_merge_sum = ds_merge.iloc[4000:6000]
plt.plot(ds_merge_sum['height_m'], label='surface water level')
plt.plot(ds_merge_sum['depth_ft'], label='shallow groundwater')
plt.ylim(top=3)
plt.ylabel('water level, meters')
plt.legend()
# plt.show()


mre_gw['date_time'] = pd.to_datetime(mre_gw['date_time'], infer_datetime_format=True)
mre_gw = mre_gw.set_index('date_time')
mre_hourly = mre_gw.resample('H').mean()
# clean up well data a little. Replace bad vals (zero or negative) with prior good val
for index, val in enumerate(mre_hourly['depth_ft']):
    if val < 0:
        mre_hourly['depth_ft'][index] =  mre_hourly['depth_ft'][index - 1]

# transform vals so they are in same direction as gage heights 
mre_hourly['depth_ft'] = (4.5 - mre_hourly['depth_ft'])*0.3048

tracy_height['date'] = pd.to_datetime(tracy_height['date'], infer_datetime_format=True)
tracy_hourly = tracy_height.set_index('date')
tracy_hourly = tracy_hourly.resample('H').mean()

us_merge = pd.merge(mre_hourly, tracy_hourly, left_index=True, right_index=True)
plt.plot(us_merge['height_m'], label='surface water level')
plt.plot(us_merge['depth_ft'], label='shallow groundwater')
plt.ylabel('water level, meters')
plt.legend()
# plt.show()
plt.clf()
us_merge_sum = us_merge.iloc[12000:19000]
plt.plot(us_merge_sum['height_m'], label='surface water level')
plt.plot(us_merge_sum['depth_ft'], label='shallow groundwater')
plt.ylim(top=4)
plt.ylabel('water level, meters')
plt.legend()
plt.show()

# Decide units/axes for plotting

# plot it, make it pretty