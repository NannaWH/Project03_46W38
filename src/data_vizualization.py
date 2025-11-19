################################################################################## Visualising data to determine patterns ## 
################################################################################

#Import relevant packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from windrose import WindroseAxes 

from data_cleaning import load_data

## Importing data 
if __name__ == "__main__": 
    data = load_data('Location2.csv')

# We create a function that creates a time series plot of data
# Function to create a time series plot for a specific period
def timeseries_plot(data, start_time, end_time):
    
    # Ensure 'Time' is a datetime type
    data['Time'] = pd.to_datetime(data['Time'])

    # Filter data for the specified time range
    mask = (data['Time'] >= pd.to_datetime(start_time)) & (data['Time'] <= pd.to_datetime(end_time))
    data_filtered = data.loc[mask]

    # Create the plot
    fig, ax1 = plt.subplots()

    # Plot normalized power
    line1, = ax1.plot(data_filtered['Time'], data_filtered['Power'], 
                      color='black', linestyle='-', label='Normalized Power Output', linewidth=0.8)

    # Plot wind speeds on secondary y-axis
    ax2 = ax1.twinx()
    line2, = ax2.plot(data_filtered['Time'], data_filtered['windspeed_100m'],
                      color='blue', linestyle=':', label='Wind Speed 100m (m/s)', linewidth=0.8)
    line3, = ax2.plot(data_filtered['Time'], data_filtered['windspeed_10m'],
                      color='green', linestyle='--', label='Wind Speed 10m (m/s)', linewidth=0.8)

    # Labels and title
    ax1.set_xlabel('Time', fontsize=8)
    ax1.set_ylabel('Power Output (Normalized 0-1)', fontsize=8)
    ax2.set_ylabel('Wind Speed (m/s)', fontsize=8)
    plt.title("Power and Wind Speeds over Time", fontsize=11)

    # We set the size of the tick labelse
    ax1.tick_params(axis='both', which='major', labelsize=7)
    ax2.tick_params(axis='both', which='major', labelsize=7)

    # Combine legends
    lines = [line1, line2, line3]
    labels = [line.get_label() for line in lines]
    fig.subplots_adjust(right=0.85)
    ax1.legend(lines, labels, loc='center left', bbox_to_anchor=(1.15, 0.5), borderaxespad=0, fontsize=8)

    ax1.grid(True)
    fig.autofmt_xdate()

    # Save figure
    plt.savefig("outputs/Timeseriesplot_power_windspeed.png", bbox_inches='tight', dpi=300)

#We create a function that scatter plots power output (y) and each (x) variable
def plots(data):
        
    #We scatter plot power output (y) and each (x) variable
    variables = [
        "temperature_2m",
        "relativehumidity_2m",
        "dewpoint_2m",
        "windspeed_10m",
        "windspeed_100m",
        "windgusts_10m"
    ]

    for variable in variables:
        x = data[variable]
        y = data["Power"]

        plt.figure()
        plt.scatter(x, y)
        plt.xlabel(variable)
        plt.ylabel("Power output")
        plt.title(f"Scatter Plot: Power vs {variable}")

        plt.savefig(f"outputs/scatterplots/scatter_{variable}.png")
        plt.close()

    # We create a wind rose of wind directions and wind speeds for 10 and 100 meter above surface

    meters_as = [10, 100]

    for meters in meters_as:
        # Create wind rose plot - wind speed
        fig = plt.figure(figsize=(6, 6))
        ax = WindroseAxes.from_ax(fig=fig)
        ax.bar(data[f'winddirection_{meters}m'], data[f'windspeed_{meters}m'], normed=True, opening=1, edgecolor='white')
        ax.set_legend(title="Wind speed (m/s)")
        plt.savefig(f"outputs/scatterplots/windrose_{meters}m.png")
        plt.close()

        # Create wind rose plot - power output
        fig = plt.figure(figsize=(6, 6))
        ax = WindroseAxes.from_ax(fig=fig)
        ax.bar(data[f'winddirection_{meters}m'], data[f'Power'], normed=True, opening=1, edgecolor='white')
        ax.set_legend(title="Wind power (0-1)")
        plt.savefig(f"outputs/scatterplots/windrose_power_{meters}m.png")
        plt.close()
  

# We run the data load and plot function for a selected location
data = load_data('Location2.csv')
timeseries_plot(data, start_time="2017-08-01 00:00", end_time="2017-09-01 00:00")
plots(data)
