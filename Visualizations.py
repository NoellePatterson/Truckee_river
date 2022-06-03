import pandas as pd
import matplotlib.pyplot as plt
import pdb

# plot overlaid years from two different usgs gages
# import daily flow data
vista = pd.read_csv('data_inputs/streamflow/ffc_outputs/vista_annual_flow_matrix.csv')
derby = pd.read_csv('data_inputs/streamflow/ffc_outputs/Blw_derby_annual_flow_matrix.csv')
# select year for plotting (input to function)
year = '1952'

def wy_overlay(upstream, downstream, year, up_label, down_label, pt_title):
    up_flow = upstream[year]
    down_flow = downstream[year]
    up_flow = pd.to_numeric(up_flow, errors='coerce')
    down_flow = pd.to_numeric(down_flow, errors='coerce')
    plt.rc('ytick', labelsize=9) 
    plt.subplot(1,1,1)
    plt.plot(up_flow, color = 'C0', linestyle = '-', label = up_label)
    plt.plot(down_flow, color = 'C1', linestyle = '--', label = down_label)
    plt.xticks([])
    plt.tick_params(axis='y', which='major', pad=1)
    month_ticks = [0,32,60,91,121,152,182,213,244,274,305,335]
    month_labels = [ 'O', 'N', 'D', 'J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S']
    plt.xticks(month_ticks, month_labels)
    plt.ylabel('Flow (cfs)', fontsize=10)
    # plt.title('{}'.format(pt_title), size=10)
    output_name = 'overlay_' + year
    plt.legend(loc='upper right', bbox_to_anchor=(0.5, -0.05), fancybox=True, ncol=7, borderaxespad = .9, fontsize='small', labelspacing=.2, columnspacing=1, markerscale=.5)
    plt.savefig('data_outputs/hydrographs/'+ output_name +'.png')
    plt.clf()
    return

for i in range(1918, 2020):
    year = str(i)
    plot = wy_overlay(vista, derby, year, 'Vista', 'Below Derby Dam', 'title')
# get one water year of flow data (year number including Jan - check w 2017)
# plot together in one plt
# make it pretty
# consider: FFC metric overlays?