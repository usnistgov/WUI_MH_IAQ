import pandas as pd
import os

# Define the base directory containing the folders
base_directory = r'C:\Users\nml\OneDrive - NIST\Documents\NIST\WUI_smoke\burn_data\aerotraks'

# List of folders to process
folders = ['bedroom2', 'kitchen']

# Loop through each folder
for folder in folders:
    # Define the file path for the .bac file
    file_path = os.path.join(base_directory, folder, 'all_data.xlsx.bac')
    
    # Check if the file exists
    if os.path.exists(file_path):
        # Load the data
        df = pd.read_excel(file_path)

        # Sort the data by 'Date and Time'
        df['Date and Time'] = pd.to_datetime(df['Date and Time'], errors='coerce')  # Convert to datetime
        df = df.sort_values(by='Date and Time', ascending=True)  # Sort from oldest to newest

        # Remove duplicate rows
        df = df.drop_duplicates()

        # Define the output file path for the cleaned data
        output_file_path = os.path.join(base_directory, folder, 'all_data.xlsx')

        # Save the cleaned DataFrame to an Excel file
        df.to_excel(output_file_path, index=False)

        print(f"Processed {file_path} and saved cleaned data to {output_file_path}")
    else:
        print(f"File not found: {file_path}")
