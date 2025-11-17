# Import necessary libraries
import pandas as pd
import os
from bokeh.plotting import figure, show
from bokeh.io import output_file
from bokeh.models import  LegendItem, Span

# Set paths and load data
absolute_path = 'C:/Users/nml/OneDrive - NIST/Documents/NIST/WUI_smoke/'
burn_log_path = os.path.join(absolute_path, 'burn_log.xlsx')
burn_log = pd.read_excel(burn_log_path, sheet_name='Sheet2')

quantaq_b_path = os.path.join(absolute_path, './burn_data/quantaq/MOD-PM-00194-b0fc215029fa4852b926bc50b28fda5a.csv')
quantaq_k_path = os.path.join(absolute_path, './burn_data/quantaq/MOD-PM-00197-a6dd467a147a4d95a7b98a8a10ab4ea3.csv')

# Load QuantAQ data
quantaq_b_data = pd.read_csv(quantaq_b_path)
quantaq_k_data = pd.read_csv(quantaq_k_path)

# Convert timestamp to datetime and filter for burn8
quantaq_b_data['timestamp_local'] = pd.to_datetime(quantaq_b_data['timestamp_local'].str.replace('T', ' ').str.replace('Z', ''), errors='coerce').dt.tz_localize(None)
quantaq_k_data['timestamp_local'] = pd.to_datetime(quantaq_k_data['timestamp_local'].str.replace('T', ' ').str.replace('Z', ''), errors='coerce').dt.tz_localize(None)

# Get burn8 date and garage closed time
burn8_date_row = burn_log[burn_log['Burn ID'] == 'burn8']
burn8_date = pd.to_datetime(burn8_date_row['Date'].iloc[0])
garage_closed_time_str = burn8_date_row['garage closed'].iloc[0]
garage_closed_time = pd.to_datetime(f"{burn8_date.strftime('%Y-%m-%d')} {garage_closed_time_str}")

# Filter data for burn8
quantaq_b_data['Date'] = quantaq_b_data['timestamp_local'].dt.date
quantaq_k_data['Date'] = quantaq_k_data['timestamp_local'].dt.date

filtered_b_data = quantaq_b_data[quantaq_b_data['Date'] == burn8_date.date()].copy()
filtered_k_data = quantaq_k_data[quantaq_k_data['Date'] == burn8_date.date()].copy()

# Apply time shift
time_shifts = {
    'QuantAQB': -2.97,
    'QuantAQK': 0
}

filtered_b_data['timestamp_local'] += pd.Timedelta(minutes=time_shifts['QuantAQB'])
filtered_k_data['timestamp_local'] += pd.Timedelta(minutes=time_shifts['QuantAQK'])

# Calculate time since garage closed
filtered_b_data['Time Since Garage Closed (hours)'] = (filtered_b_data['timestamp_local'] - garage_closed_time).dt.total_seconds() / 3600
filtered_k_data['Time Since Garage Closed (hours)'] = (filtered_k_data['timestamp_local'] - garage_closed_time).dt.total_seconds() / 3600

# Create figure
p = figure(
    x_axis_label='Time Since Garage Closed (hours)',
    y_axis_label='PM2.5 Concentration (µg/m³)',
    x_axis_type='linear',
    y_axis_type='log',
    width=800,
    height=500,
    y_range=(10**-1, 10**3),  # Set y-axis range
    title="QuantAQ PM2.5 Concentration for Burn 8"
)

# Plot data
p.line(filtered_b_data['Time Since Garage Closed (hours)'], filtered_b_data['pm25'], legend_label='Bedroom', line_color='blue')
p.line(filtered_k_data['Time Since Garage Closed (hours)'], filtered_k_data['pm25'], legend_label='Morning Room', line_color='green')

# Set x-axis range
p.x_range.start = -1
p.x_range.end = 3

# Add vertical line at Time = 0 (garage closed)
garage_closed_line = Span(location=0, dimension='height', line_color='black', line_width=1, line_dash='dashed')
p.add_layout(garage_closed_line)
p.legend.items.append(LegendItem(label='Garage Closed', renderers=[p.renderers[-1]]))

# Create directory if it doesn't exist
output_dir = os.path.join(absolute_path, 'Paper_figures')
os.makedirs(output_dir, exist_ok=True)

# Save to file
output_file(os.path.join(output_dir, 'QuantAQ_PM25_Burn8.html'))
show(p)