#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%%
# Example DataFrame
data = {
    'x': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'y': [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
}
df = pd.DataFrame(data)

# Define the start and end values for x
start_value = 3
end_value = 7

# Select the section of data where x values are between start_value and end_value
subset_df = df[(df['x'] >= start_value) & (df['x'] <= end_value)]

# Perform linear regression on the selected portion
x_subset = subset_df['x'].values
y_subset = subset_df['y'].values

# Calculate necessary parameters
x_mean = np.mean(x_subset)
y_mean = np.mean(y_subset)

numerator = np.sum((x_subset - x_mean) * (y_subset - y_mean))
denominator = np.sum((x_subset - x_mean) ** 2)

slope = numerator / denominator
intercept = y_mean - slope * x_mean

print(f"Slope: {slope}, Intercept: {intercept}")

# Plot the data and the fitted line
plt.scatter(subset_df['x'], subset_df['y'], color='blue', label='Data')
plt.plot(x_subset, slope * x_subset + intercept, color='red', label='Linear fit')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
#%%
# Plot the full DataFrame
plt.scatter(df['x'], df['y'], color='blue', label='Full DataFrame')

# Plot the selected portion of the DataFrame
plt.scatter(subset_df['x'], subset_df['y'], color='green', label='Subset for Linear Fit')

# Plot the linear fit line
plt.plot(x_subset, slope * x_subset + intercept, color='red', label='Linear fit')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Linear Fit on Subset of DataFrame')
plt.show()
# %%
slope, intercept = np.polyfit(x_subset, y_subset, 1)  # Using numpy polyfit for simplicity
line_fit = slope * x_subset + intercept

# Calculate the error (root mean squared error)
rmse = np.sqrt(np.mean((y_subset - line_fit) ** 2))

# Plot the full DataFrame
plt.scatter(df['x'], df['y'], color='blue', label='Full DataFrame')

# Plot the selected portion of the DataFrame
plt.scatter(subset_df['x'], subset_df['y'], color='green', label='Subset for Linear Fit')

# Plot the linear fit line
plt.plot(x_subset, line_fit, color='red', label='Linear fit')

# Add the equation of the linear fit line and the error next to the linear fit line
equation_text = f'y = {slope:.2f}x + {intercept:.2f}'
error_text = f'RMSE: {rmse:.2f}'

# Annotate the plot with the equation of the linear fit line
plt.annotate(equation_text, xy=(x_subset.mean(), line_fit.mean()), xytext=(x_subset.mean() + 1, line_fit.mean() + 5),
             fontsize=12, color='black', arrowprops=dict(facecolor='black', arrowstyle='->'), ha='left', va='center')

# Annotate the plot with the RMSE
plt.annotate(error_text, xy=(x_subset.mean(), line_fit.mean()), xytext=(x_subset.mean() + 1, line_fit.mean() + 3),
             fontsize=12, color='black', arrowprops=dict(facecolor='black', arrowstyle='->'), ha='left', va='center')

plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Linear Fit on Subset of DataFrame')
plt.show()





# %%
import numpy as np
import pandas as pd
from scipy import optimize
from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource, Band, Label, Arrow, OpenHead
from bokeh.io import output_notebook
from bokeh.palettes import Spectral11
import datetime as dt

# For visualization in Jupyter Notebook
output_notebook()

# Generate synthetic decay data
np.random.seed(0)

# Generate time series in hours using datetime
start_time = dt.datetime(2024, 1, 1, 0, 0)  # Starting at midnight
time_hours = [start_time + dt.timedelta(hours=i * 0.24) for i in range(100)]  # Every 0.24 hours

# Convert to numpy arrays for calculations
time_np = np.array(time_hours)
time_hours_np = np.array([t.hour + t.minute / 60.0 for t in time_np])  # Convert to hours of the day

# Decay data
decay_data = {
    'Series 1': np.exp(-time_hours_np / 10) + np.random.normal(scale=0.05, size=len(time_hours)),
    'Series 2': np.exp(-time_hours_np / 15) + np.random.normal(scale=0.1, size=len(time_hours))
}

# Define different datetime ranges for fitting each series
fit_ranges = {
    'Series 1': (dt.datetime(2024, 1, 1, 3, 0), dt.datetime(2024, 1, 1, 8, 0)),
    'Series 2': (dt.datetime(2024, 1, 1, 6, 0), dt.datetime(2024, 1, 1, 12, 0))
}

# Define specific label positions for each series
label_positions = {
    'Series 1': (30, -0.3),  # x_offset in minutes, y_offset in units above the line
    'Series 2': (60, 0.15)    # x_offset in minutes, y_offset in units above the line
}

def exponential_decay(x, a, b):
    return a * np.exp(-b * x)

def fit_exponential_curve(x, y):
    # Fit the exponential decay model
    popt, pcov = optimize.curve_fit(exponential_decay, x, y, p0=(1, 1))
    a, b = popt
    # Calculate standard error for the parameter 'b'
    std_err = np.sqrt(np.diag(pcov))
    y_fit = exponential_decay(x, *popt)
    return popt, y_fit, std_err

# Create a Bokeh figure with logarithmic y-axis and datetime on x-axis
p = figure(title="Decay Data with Exponential Curve Fit", x_axis_label='Time', y_axis_label='Value', y_axis_type='log', x_axis_type='datetime')

# Plot decay data
colors = Spectral11  # Color palette for differentiating lines

for i, (label, data) in enumerate(decay_data.items()):
    p.line(time_np, data, legend_label=label, line_width=2, color=colors[i])
    
    # Define the datetime range for fitting
    fit_start_time, fit_end_time = fit_ranges[label]
    
    # Convert datetime to indices
    fit_start_index = np.searchsorted(time_np, fit_start_time)
    fit_end_index = np.searchsorted(time_np, fit_end_time)
    
    # Fit exponential curve on the specified portion of the time series
    x_fit = time_hours_np[fit_start_index:fit_end_index]
    y_fit = data[fit_start_index:fit_end_index]
    popt, y_curve_fit, std_err = fit_exponential_curve(x_fit, y_fit)
    
    # Define the fit curve and uncertainty band
    curve_fit_y = exponential_decay(time_hours_np, *popt)
    
    # Calculate uncertainty for the portion of the data used in the fit
    uncertainty = 1.96 * std_err[1] * np.exp(-popt[1] * x_fit)
    
    # Add the curve fit line for the fitting portion
    p.line(time_np[fit_start_index:fit_end_index], y_curve_fit, line_color='red', legend_label=f'{label} Exponential Fit')

    # Add the uncertainty band for the fitting portion
    source = ColumnDataSource(data=dict(x=time_np[fit_start_index:fit_end_index], lower=y_fit - uncertainty, upper=y_fit + uncertainty))
    band = Band(base='x', lower='lower', upper='upper', source=source, level='underlay', fill_alpha=0.2, fill_color='red')
    p.add_layout(band)

    # Prepare fit text with only b value and uncertainty
    fit_info = (f"{popt[1]:.2f} h^{-1} ± {1.96 * std_err[1]:.2f} h^{-1}")
    
    # Retrieve label position for the current series
    x_offset, y_offset = label_positions[label]
    label_x = time_np[fit_end_index-1] + dt.timedelta(minutes=x_offset)  # Move label to the right
    label_y = y_curve_fit[-1] + y_offset  # Move label slightly above the fit line end
    
    # Add Label
    p.add_layout(Label(x=label_x, y=label_y, text=fit_info, text_font_size='10pt', text_align='left', text_baseline='middle'))

    # Add Arrow pointing to the fit line
    arrow = Arrow(end=OpenHead(size=10, line_color='black'), line_color='black', x_start=label_x, y_start=label_y, x_end=time_np[fit_end_index-1], y_end=y_curve_fit[-1])
    p.add_layout(arrow)

# Show the plot
show(p)



# %%
import numpy as np
import pandas as pd
from scipy import optimize
from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource, Band, Label, Arrow, OpenHead
from bokeh.io import output_notebook
from bokeh.palettes import Spectral11
import datetime as dt

# For visualization in Jupyter Notebook
output_notebook()

# Generate synthetic decay data
np.random.seed(0)

# Generate time series in hours using datetime
start_time = dt.datetime(2024, 1, 1, 0, 0)  # Starting at midnight
time_hours = [start_time + dt.timedelta(hours=i * 0.24) for i in range(100)]  # Every 0.24 hours

# Convert to numpy arrays for calculations
time_np = np.array(time_hours)
time_hours_np = np.array([t.hour + t.minute / 60.0 for t in time_np])  # Convert to hours of the day

# Decay data
decay_data = {
    'Series 1': np.exp(-time_hours_np / 10) + np.random.normal(scale=0.05, size=len(time_hours)),
    'Series 2': np.exp(-time_hours_np / 15) + np.random.normal(scale=0.1, size=len(time_hours))
}

# Define different datetime ranges for fitting each series
fit_ranges = {
    'Series 1': (dt.datetime(2024, 1, 1, 3, 0), dt.datetime(2024, 1, 1, 8, 0)),
    'Series 2': (dt.datetime(2024, 1, 1, 6, 0), dt.datetime(2024, 1, 1, 12, 0))
}

# Define specific label positions for each series
label_positions = {
    'Series 1': (10, -0.3),  # x_offset in minutes, y_offset in units above the line
    'Series 2': (60, 0.2)    # x_offset in minutes, y_offset in units above the line
}

# Define custom line colors and line types for each series
line_properties = {
    'Series 1': {'color': 'blue', 'line_dash': 'solid'},
    'Series 2': {'color': 'green', 'line_dash': 'dashed'}
}

# Define the x-axis and y-axis ranges
x_range_start = dt.datetime(2024, 1, 1, 0, 0)
x_range_end = dt.datetime(2024, 1, 2, 0, 0)  # Cover the full next day for a full day's view
y_range_start = 0.01  # Avoid log scale issues by not starting at zero
y_range_end = 1.5     # Example upper limit for the y-axis

def exponential_decay(x, a, b):
    return a * np.exp(-b * x)

def fit_exponential_curve(x, y):
    # Fit the exponential decay model
    popt, pcov = optimize.curve_fit(exponential_decay, x, y, p0=(1, 1))
    a, b = popt
    # Calculate standard error for the parameter 'b'
    std_err = np.sqrt(np.diag(pcov))
    y_fit = exponential_decay(x, *popt)
    return popt, y_fit, std_err

# Create a Bokeh figure with logarithmic y-axis and datetime on x-axis
p = figure(
    title="Decay Data with Exponential Curve Fit",
    x_axis_label='Time (hours)',
    y_axis_label='-Ln(C_t1 / C_t0)',
    y_axis_type='log',
    x_axis_type='datetime',
    x_range=(x_range_start, x_range_end),
    y_range=(y_range_start, y_range_end)
)

# Plot decay data
for label, data in decay_data.items():
    # Get line properties for the current series
    color = line_properties[label]['color']
    line_dash = line_properties[label]['line_dash']
    
    # Plot the original data with legend
    p.line(time_np, data, legend_label=label, line_width=2, color=color, line_dash=line_dash)
    
    # Define the datetime range for fitting
    fit_start_time, fit_end_time = fit_ranges[label]
    
    # Convert datetime to indices
    fit_start_index = np.searchsorted(time_np, fit_start_time)
    fit_end_index = np.searchsorted(time_np, fit_end_time)
    
    # Fit exponential curve on the specified portion of the time series
    x_fit = time_hours_np[fit_start_index:fit_end_index]
    y_fit = data[fit_start_index:fit_end_index]
    popt, y_curve_fit, std_err = fit_exponential_curve(x_fit, y_fit)
    
    # Define the fit curve and uncertainty band
    curve_fit_y = exponential_decay(time_hours_np, *popt)
    
    # Calculate uncertainty for the portion of the data used in the fit
    uncertainty = 1.96 * std_err[1] * np.exp(-popt[1] * x_fit)
    
    # Add the curve fit line for the fitting portion without legend
    p.line(time_np[fit_start_index:fit_end_index], y_curve_fit, line_color='red', line_dash='solid')  # No legend label
    
    # Add the uncertainty band for the fitting portion
    source = ColumnDataSource(data=dict(x=time_np[fit_start_index:fit_end_index], lower=y_fit - uncertainty, upper=y_fit + uncertainty))
    band = Band(base='x', lower='lower', upper='upper', source=source, level='underlay', fill_alpha=0.2, fill_color='red')
    p.add_layout(band)

    # Prepare fit text with only b value and uncertainty
    fit_info = (f"{popt[1]:.2f} h^{-1} ± {1.96 * std_err[1]:.2f} h^{-1}")
    
    # Retrieve label position for the current series
    x_offset, y_offset = label_positions[label]
    label_x = time_np[fit_end_index-1] + dt.timedelta(minutes=x_offset)  # Move label to the right
    label_y = y_curve_fit[-1] + y_offset  # Move label slightly above the fit line end
    
    # Add Label with LaTeX-like text
    p.add_layout(Label(x=label_x, y=label_y, text=fit_info, text_font_size='10pt', text_align='left', text_baseline='middle'))

    # Add Arrow pointing to the fit line
    arrow = Arrow(end=OpenHead(size=10, line_color='black'), line_color='black', x_start=label_x, y_start=label_y, x_end=time_np[fit_end_index-1], y_end=y_curve_fit[-1])
    p.add_layout(arrow)

# Show the plot
show(p)

# %%


import pandas as pd
import numpy as np
from scipy import optimize
from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource, Band, Label, Arrow, OpenHead
from bokeh.io import output_notebook

# For visualization in Jupyter Notebook
output_notebook()

# Load data from Excel file
# Make sure to replace 'data.xlsx' with the path to your Excel file
file_path = 'data.xlsx'
sheet_name = 'Sheet1'  # Change this if your sheet name is different

# Read data into DataFrame
df = pd.read_excel(file_path, sheet_name=sheet_name)

# Extract the minutes and data series
minutes_np = df['minutes_since_start'].values
time_hours_np = minutes_np / 60  # Convert minutes to hours
decay_data = {
    'Series 1': df['Series 1'].values,
    'Series 2': df['Series 2'].values
}

# Define different ranges for fitting each series in minutes
fit_ranges = {
    'Series 1': (180, 480),  # Fit from 180 to 480 minutes (3 to 8 hours)
    'Series 2': (360, 720)   # Fit from 360 to 720 minutes (6 to 12 hours)
}

# Define specific label positions for each series
label_positions = {
    'Series 1': (60, 0.15),  # x_offset in minutes, y_offset in units above the line
    'Series 2': (120, 0.2)   # x_offset in minutes, y_offset in units above the line
}

# Define custom line colors and line types for each series
line_properties = {
    'Series 1': {'color': 'blue', 'line_dash': 'solid'},
    'Series 2': {'color': 'green', 'line_dash': 'dotted'}
}

# Define the x-axis and y-axis ranges
x_range_start = minutes_np.min()
x_range_end = minutes_np.max() + 60  # Cover an additional hour for a full view
y_range_start = 0.01  # Avoid log scale issues by not starting at zero
y_range_end = 1.5     # Example upper limit for the y-axis

def exponential_decay(x, a, b):
    return a * np.exp(-b * x)

def fit_exponential_curve(x, y):
    # Fit the exponential decay model
    popt, pcov = optimize.curve_fit(exponential_decay, x, y, p0=(1, 1))
    a, b = popt
    # Calculate standard error for the parameter 'b'
    std_err = np.sqrt(np.diag(pcov))
    y_fit = exponential_decay(x, *popt)
    return popt, y_fit, std_err

# Create a Bokeh figure with logarithmic y-axis and minutes on x-axis
p = figure(
    title="Decay Data with Exponential Curve Fit",
    x_axis_label='Time (minutes)',
    y_axis_label='-Ln(C<sub>t1</sub> / C<sub>t0</sub>)',
    y_axis_type='log',
    x_axis_type='linear',
    x_range=(x_range_start, x_range_end),
    y_range=(y_range_start, y_range_end)
)

# Plot decay data
for label, data in decay_data.items():
    # Get line properties for the current series
    color = line_properties[label]['color']
    line_dash = line_properties[label]['line_dash']
    
    # Plot the original data with legend
    p.line(minutes_np, data, legend_label=label, line_width=2, color=color, line_dash=line_dash)
    
    # Define the range for fitting
    fit_start_min, fit_end_min = fit_ranges[label]
    
    # Extract fitting data
    fit_start_index = np.searchsorted(minutes_np, fit_start_min)
    fit_end_index = np.searchsorted(minutes_np, fit_end_min)
    
    # Fit exponential curve on the specified portion of the time series
    x_fit = time_hours_np[fit_start_index:fit_end_index]
    y_fit = data[fit_start_index:fit_end_index]
    popt, y_curve_fit, std_err = fit_exponential_curve(x_fit, y_fit)
    
    # Define the fit curve and uncertainty band
    curve_fit_y = exponential_decay(time_hours_np, *popt)
    
    # Calculate uncertainty for the portion of the data used in the fit
    uncertainty = 1.96 * std_err[1] * np.exp(-popt[1] * x_fit)
    
    # Add the curve fit line for the fitting portion without legend
    p.line(minutes_np[fit_start_index:fit_end_index], y_curve_fit, line_color='red', line_dash='dotted')  # No legend label
    
    # Add the uncertainty band for the fitting portion
    source = ColumnDataSource(data=dict(x=minutes_np[fit_start_index:fit_end_index], lower=y_fit - uncertainty, upper=y_fit + uncertainty))
    band = Band(base='x', lower='lower', upper='upper', source=source, level='underlay', fill_alpha=0.2, fill_color='red')
    p.add_layout(band)

    # Prepare fit text with only b value and uncertainty
    fit_info = (f"b value: {popt[1]:.2f} h$^{-1}$ ± {1.96 * std_err[1]:.2f}")
    
    # Retrieve label position for the current series
    x_offset, y_offset = label_positions[label]
    label_x = minutes_np[fit_end_index-1] + x_offset  # Move label to the right
    label_y = y_curve_fit[-1] + y_offset  # Move label slightly above the fit line end
    
    # Add Label with LaTeX-like text
    p.add_layout(Label(x=label_x, y=label_y, text=fit_info, text_font_size='10pt', text_align='left', text_baseline='middle'))

    # Add Arrow pointing to the fit line
    arrow = Arrow(end=OpenHead(size=10, line_color='black'), line_color='black', x_start=label_x, y_start=label_y, x_end=minutes_np[fit_end_index-1], y_end=y_curve_fit[-1])
    p.add_layout(arrow)

# Show the plot
show(p)

# %%
