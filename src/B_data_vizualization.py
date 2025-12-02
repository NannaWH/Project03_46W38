################################################################################## Visualising data to determine patterns ######################################
################################################################################

#Import relevant packages
import pandas as pd
import matplotlib.pyplot as plt
from windrose import WindroseAxes 

# We create a function that creates a time series plot of data
def timeseries_plot(data, start_time, end_time):
    """Creating scatter plots showing the relationship between power output and each of meterological and wind data variables
    
    Args:
        data: weather and power output data
        start_time: start time for subplot of timeseries plot
        end_time: end time for subplot of timeseries plot

    Returns: 
        A plot of the timeseries data (power output, wind speed) for a subset of the data. 
    """
    
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
    plt.savefig("outputs/timeseries/Timeseriesplot_power_windspeed.png", bbox_inches='tight', dpi=300)


#We create a function that scatter plots power output (y) and each (x) variable
def scatter_plots(data):
    """Creating scatter plots showing the relationship between power output and each of meterological and wind data variables
    
    Args:
        data: weather and power output data

    Returns: 
        Scatterplots showing the correlation between out y-variable (power output) and the explanatory x-variables.

    """
        
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

def wind_rose_plot(data):
    """Creating a windrose plot of the wind directions
    
    Args:
        data: weather and power output data.

    Returns:
        A saved windrose diagram showing the power uput for difference windspeeds and directions.
    """

    # We create a wind rose of wind directions and wind speeds for 10 and 100 meter above surface
    meters_as = [10, 100]

    for meters in meters_as:
        # Create wind rose plot - wind speed
        fig = plt.figure(figsize=(6, 6))
        ax = WindroseAxes.from_ax(fig=fig)
        ax.bar(data[f'winddirection_{meters}m'], data[f'windspeed_{meters}m'], normed=True, opening=1, edgecolor='white')
        ax.set_legend(title="Wind speed (m/s)")
        plt.savefig(f"outputs/windrose/windrose_{meters}m.png")
        plt.close()
  

