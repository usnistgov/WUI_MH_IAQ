#%% RUN 
# import needed modules
print('Importing needed modules')
import os
import numpy as np
import pandas as pd
from datetime import datetime as dt
from scipy import optimize
from bokeh.plotting import figure, show, output_file, output_notebook
from bokeh.models import LinearAxis, Range1d

#%% RUN User defines directory path for dataset
# User set directory path
directory_path = 'C:/Users/nml/OneDrive - NIST/Documents/NIST/WUI_smoke/'

# Define file names for different burn data
file_name_burn1 = 'burn_data/vaisalaht/20240421-MH_Task_Logger_Data.xlsx'
file_name_burns_2_to_9 = 'burn_data/vaisalaht/20240429-MH_Data_Processed.xlsx'
file_name_burn10 = 'burn_data/vaisalaht/20240531-MH_Data_Processed.xlsx'

# Define sheet names
sheet_name_burn1 = 'T_RH'
sheet_name_default = 'Sheet1'

# Function to read data for each Burn based on file names and sheet names
def read_burn_data(burn):
    if burn == 'Burn 1':
        file_path = os.path.join(directory_path, file_name_burn1)
        df = pd.read_excel(file_path, sheet_name=sheet_name_burn1)
        df['datetime'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Time'].astype(str), errors='coerce')
        df.drop(['Date', 'Time'], axis=1, inplace=True)
    else:
        file_path = os.path.join(directory_path, file_name_burns_2_to_9 if burn != 'Burn 10' else file_name_burn10)
        df = pd.read_excel(file_path, sheet_name=sheet_name_default)
        # Create datetime column for Burn 2 through Burn 10
        if 'Date' in df.columns and 'Time' in df.columns:
            df['datetime'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Time'].astype(str), errors='coerce')
        elif 'datetime' not in df.columns:
            print(f"Missing 'Date' or 'Time' columns in file for {burn}.")
    return df

# Define burn dates
burn_dates = {
    'Burn 1': '2024-04-26',       
    'Burn 2': '2024-05-02',
    'Burn 3': '2024-05-06',
    'Burn 4': '2024-05-09',
    'Burn 5': '2024-05-13',    
    'Burn 6': '2024-05-17',
    'Burn 7': '2024-05-20',
    'Burn 8': '2024-05-23',
    'Burn 9': '2024-05-28',
    'Burn 10': '2024-05-31'
}

# Function to filter data by whole day
def filter_by_date(df, date):
    date = pd.to_datetime(date).date()
    if 'datetime' in df.columns:
        df['date_only'] = df['datetime'].dt.date
        filtered_df = df[df['date_only'] == date].copy()
        df.drop(columns=['date_only'], inplace=True)
        if filtered_df.empty:
            print(f"No data found for date {date}.")
        return filtered_df
    else:
        print("Column 'datetime' is missing.")
        return pd.DataFrame()

# Create a dictionary to hold filtered and processed data for each Burn
burn_data = {burn: pd.DataFrame() for burn in burn_dates.keys()}

# Filter and process data for each Burn
for burn, date in burn_dates.items():
    print(f"Processing {burn} for date {date}...")
    df = read_burn_data(burn)
    burn_data[burn] = filter_by_date(df, date)

# Define colors for RH series
colors = {
    'RH_Bed1_M3_C0': 'blue',
    'RH_Bed2_M3_C1': 'green',
    'RH_Bed3_M3_C2': 'red',
    'RH_Liv_M3_C3': 'purple',
    'RH_Fam_M3_C4': 'orange'
}

legend_labels = {
    'RH_Bed1_M3_C0': 'BR1 RH',
    'RH_Bed2_M3_C1': 'BR2 RH',
    'RH_Bed3_M3_C2': 'BR3 RH',
    'RH_Liv_M3_C3': 'Liv.R RH',
    'RH_Fam_M3_C4': 'Fam.R RH'
}
#%% RUN 
# import needed modules
print('Importing needed modules')
import os
import numpy as np
import pandas as pd
from datetime import datetime as dt
from scipy import optimize
from bokeh.plotting import figure, show, output_file, output_notebook
from bokeh.models import LinearAxis, Range1d

#%% RUN User defines directory path for dataset
# User set directory path
directory_path = 'C:/Users/nml/OneDrive - NIST/Documents/NIST/WUI_smoke/'

# Define file names for different burn data
file_name_burn1 = 'burn_data/vaisalaht/20240421-MH_Task_Logger_Data.xlsx'
file_name_burns_2_to_9 = 'burn_data/vaisalaht/20240429-MH_Data_Processed.xlsx'
file_name_burn10 = 'burn_data/vaisalaht/20240531-MH_Data_Processed.xlsx'

# Define sheet names
sheet_name_burn1 = 'T_RH'
sheet_name_default = 'Sheet1'

# Function to read data for each Burn based on file names and sheet names
def read_burn_data(burn):
    if burn == 'Burn 1':
        file_path = os.path.join(directory_path, file_name_burn1)
        df = pd.read_excel(file_path, sheet_name=sheet_name_burn1)
        df['datetime'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Time'].astype(str), errors='coerce')
        df.drop(['Date', 'Time'], axis=1, inplace=True)
    else:
        file_path = os.path.join(directory_path, file_name_burns_2_to_9 if burn != 'Burn 10' else file_name_burn10)
        df = pd.read_excel(file_path, sheet_name=sheet_name_default)
        # Create datetime column for Burn 2 through Burn 10
        if 'Date' in df.columns and 'Time' in df.columns:
            df['datetime'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Time'].astype(str), errors='coerce')
        elif 'datetime' not in df.columns:
            print(f"Missing 'Date' or 'Time' columns in file for {burn}.")
    return df

# Define burn dates
burn_dates = {
    'Burn 1': '2024-04-26',       
    'Burn 2': '2024-05-02',
    'Burn 3': '2024-05-06',
    'Burn 4': '2024-05-09',
    'Burn 5': '2024-05-13',    
    'Burn 6': '2024-05-17',
    'Burn 7': '2024-05-20',
    'Burn 8': '2024-05-23',
    'Burn 9': '2024-05-28',
    'Burn 10': '2024-05-31'
}

# Function to filter data by whole day
def filter_by_date(df, date):
    date = pd.to_datetime(date).date()
    if 'datetime' in df.columns:
        df['date_only'] = df['datetime'].dt.date
        filtered_df = df[df['date_only'] == date].copy()
        df.drop(columns=['date_only'], inplace=True)
        if filtered_df.empty:
            print(f"No data found for date {date}.")
        return filtered_df
    else:
        print("Column 'datetime' is missing.")
        return pd.DataFrame()

# Create a dictionary to hold filtered and processed data for each Burn
burn_data = {burn: pd.DataFrame() for burn in burn_dates.keys()}

# Filter and process data for each Burn
for burn, date in burn_dates.items():
    print(f"Processing {burn} for date {date}...")
    df = read_burn_data(burn)
    burn_data[burn] = filter_by_date(df, date)

# Define colors for RH series
colors = {
    'RH_Bed1_M3_C0': 'blue',
    'RH_Bed2_M3_C1': 'green',
    'RH_Bed3_M3_C2': 'red',
    'RH_Liv_M3_C3': 'purple',
    'RH_Fam_M3_C4': 'orange'
}

legend_labels = {
    'RH_Bed1_M3_C0': 'BR1 RH',
    'RH_Bed2_M3_C1': 'BR2 RH',
    'RH_Bed3_M3_C2': 'BR3 RH',
    'RH_Liv_M3_C3': 'Liv.R RH',
    'RH_Fam_M3_C4': 'Fam.R RH'
}

#%%
# Loop through each burn to create and save a figure
for burn in burn_dates.keys():
    print(f"Generating plot for {burn}...")
    
    # Prepare the data for plotting
    data = burn_data[burn]
    
    # Define output file for each burn
    output_file_path = os.path.join(directory_path, f'Paper_figures/{burn}_Vaisala_Temperature_and_RH.html')
    output_file(output_file_path)
    
    # Create a Bokeh figure
    p = figure(title=f'WUI: {burn}', width=800, height=500, x_axis_type='datetime')
    
    # Plot Temperature on the left y-axis
    if not data.empty:
        p.line(data['datetime'], data['T_HVAC-S_M5_C7'], color="black", legend_label="Temperature", line_width=1)
        p.yaxis.axis_label = 'Temperature [°C]'
        p.y_range.start = 0  # Set the start of the temperature axis
        p.y_range.end = 30    # Set the end of the temperature axis
    
    # Create a second y-axis for Relative Humidity
    p.extra_y_ranges = {"RH": Range1d(start=20, end=70)}  # Adjust range if needed
    p.add_layout(LinearAxis(y_range_name="RH", axis_label="Relative Humidity [%]"), 'right')
    
    # Plot Relative Humidity from Vaisala dataframe on the right y-axis
    for rh_col in colors:
        if rh_col in data.columns:
            p.line(data['datetime'], data[rh_col], color=colors[rh_col], legend_label=legend_labels.get(rh_col, rh_col), line_width=1, y_range_name="RH")
    
    # Customize the plot
    p.xaxis.axis_label = 'Datetime'
    
    # Configure legend to run horizontally across the top
    p.legend.location = 'bottom_center'
    p.legend.orientation = 'horizontal'
    p.legend.label_text_font = 'Calibri'
    p.legend.label_text_font_size = '12pt'
    p.legend.border_line_color = None
    
    p.axis.axis_label_text_font = 'Calibri'
    p.axis.axis_label_text_font_size = '12pt'
    p.axis.axis_label_text_font_style = 'normal'
    
    # Center the title text
    p.title.align = 'center'
    
    # Display the plot
    show(p)
    
    print(f"Plot saved to {output_file_path}")
# %%
