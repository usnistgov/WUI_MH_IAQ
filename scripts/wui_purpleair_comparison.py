# -*- coding: utf-8 -*-
#%% RUN 
# import needed modules
print('Importing needed modules')
import os
import numpy as np
import pandas as pd
from datetime import datetime
from bokeh.palettes import Spectral11
from bokeh.plotting import figure, show, output_file, output_notebook
from bokeh.models.tools import HoverTool, CrosshairTool, Span
from bokeh.layouts import gridplot
from bokeh.models import CrosshairTool, Span, Legend, LegendItem

#%% RUN User defines directory path for datset, dataset used, and dataset final location
# User set absolute_path
absolute_path = 'C:/Users/nml/OneDrive - NIST/Documents/NIST/WUI_smoke/burn_data/purpleair/' #USER ENTERED PROJECT PATH
os.chdir(absolute_path)

# use only one dataset at a time
dataset = 'garage-kitchen' #USER ENTERED selected
print('dataset selected: '+dataset)

# Set up Datasets that are used
print(dataset + ' dataset selected and final file path defined')
if dataset == 'garage-kitchen':
    dfk = pd.read_excel("./garage-kitchen.xlsx",sheet_name='(P2)kitchen')
    dfg = pd.read_excel("./garage-kitchen.xlsx",sheet_name='(P1)garage')
    
elif dataset == 'ahs21':
    df = pd.read_csv("./ahs_data/2021/household.csv") #USER ENTERED FINAL PATH FOR AHS

dfk['DateTime']=pd.to_datetime(dfk['DateTime'])
dfk = dfk.set_index('DateTime')

dfg['DateTime']=pd.to_datetime(dfg['DateTime'])
dfg = dfg.set_index('DateTime')


dfk = dfk.drop(columns=['PA2-30::3D A', 'PA2-30::3D B'])
dfg = dfg.drop(columns=['P1:30::5 A','P1:30::5 B'])

df = pd.merge(dfk,dfg,left_index=True, right_index=True,how='outer')
df.rename(columns = {'Average_x':'kitchen','Average_y':'garage'}, inplace = True)
df = df.interpolate('index')

df_5min = pd.DataFrame()
df_5min['kitchen']=df.kitchen.resample('5min').mean()
df_5min['garage']=df.garage.resample('5min').mean()

#%%
# For visualization in Jupyter Notebook
#output_notebook()
# For visualization to static HTML
output_file('C:/Users/nml/OneDrive - NIST/Documents/NIST/WUI_smoke/Paper_figures/PurpleAirComparison.html')

p = figure(
    y_range=[0, 20], 
    max_width=800, 
    height=500,
    x_axis_type="datetime"
    )

numlines=len(df_5min.columns)

r = p.multi_line(xs=[df_5min.index.values]*numlines,
                ys=[df_5min[name].values for name in df_5min],
                color=['green','blue'],
                line_width=1)

#p.title.text = "Purple Air PM2.5 Concentration"
p.title.text_font_size = '12pt'
p.title.align = 'center'

p.xaxis.axis_label = "Date (month/day)"

p.yaxis.axis_label = "5 minute average PM2.5 concentration (µg/m3)"

legend = Legend(items=[
    LegendItem(label="Morning Room", renderers=[r], index=0),
    LegendItem(label="Garage", renderers=[r], index=1),
])

# Adjust the legend position to the lower-left corner
p.legend.label_text_font = 'Calibri'
p.legend.label_text_font_size = '12pt'
p.legend.orientation = 'vertical'

p.axis.axis_label_text_font = 'Calibri'
p.axis.axis_label_text_font_size = '12pt'
p.axis.axis_label_text_font_style = 'normal'

p.add_layout(legend)
p.legend.location = "top_right"

show(p)

# %%
