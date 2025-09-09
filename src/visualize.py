import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
import argparse

def plot_flight_data(df, aircraft_id, output_path):
    """
    Plots the time-series of key flight parameters for a specific aircraft and saves it to a file.
    """
    aircraft_df = df[df['Id'] == aircraft_id].copy()
    
    if aircraft_df.empty:
        print(f"No data found for aircraft ID: {aircraft_id}")
        return

    # Create a color mapping for maneuvers
    maneuver_labels = aircraft_df['Maneuver_Label'].dropna().unique()
    colors = list(mcolors.TABLEAU_COLORS.values())
    maneuver_colors = {label: colors[i % len(colors)] for i, label in enumerate(maneuver_labels) if label != ''}

    fig, axes = plt.subplots(5, 1, figsize=(18, 22), sharex=True)
    time_col = 'Time'

    # Plot Roll, Pitch, Yaw
    axes[0].plot(aircraft_df[time_col], aircraft_df['Roll'], label='Roll')
    axes[0].plot(aircraft_df[time_col], aircraft_df['Pitch'], label='Pitch')
    axes[0].plot(aircraft_df[time_col], aircraft_df['Yaw'], label='Yaw')
    axes[0].set_ylabel('Degrees')
    axes[0].set_title(f'Flight Data for Aircraft ID: {aircraft_id}')
    axes[0].legend()
    axes[0].grid(True)

    # Plot Altitude and TAS
    axes[1].plot(aircraft_df[time_col], aircraft_df['Altitude'], label='Altitude', color='b')
    ax2 = axes[1].twinx()
    ax2.plot(aircraft_df[time_col], aircraft_df['TAS'], label='TAS', color='r')
    axes[1].set_ylabel('Feet', color='b')
    ax2.set_ylabel('Knots', color='r')
    axes[1].legend(loc='upper left')
    ax2.legend(loc='upper right')
    axes[1].grid(True)

    # Plot G-force
    axes[2].plot(aircraft_df[time_col], aircraft_df['G'], label='G-force')
    axes[2].set_ylabel('G')
    axes[2].legend()
    axes[2].grid(True)

    # Plot FFP_Label
    axes[3].plot(aircraft_df[time_col], aircraft_df['FFP_Label'], label='FFP Label', drawstyle='steps-post')
    axes[3].set_ylabel('FFP Label')
    axes[3].legend()
    axes[3].grid(True)

    # Plot Maneuver_Label
    axes[4].plot(aircraft_df[time_col], aircraft_df['Maneuver_Label'], label='Maneuver Label', drawstyle='steps-post')
    axes[4].set_ylabel('Maneuver Label')
    axes[4].legend()
    axes[4].grid(True)

    # Color the background for maneuvers
    for ax in axes:
        for label, color in maneuver_colors.items():
            for _, group in aircraft_df[aircraft_df['Maneuver_Label'] == label].groupby((aircraft_df['Maneuver_Label'] != label).cumsum()):
                if not group.empty:
                    start_time = group[time_col].iloc[0]
                    end_time = group[time_col].iloc[-1]
                    ax.axvspan(start_time, end_time, color=color, alpha=0.3)

    plt.xlabel('Time (seconds)')
    plt.tight_layout()
    
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize time-series flight data for a specific aircraft.")
    parser.add_argument("input_csv", help="Path to the maneuver_labeled CSV file.")
    parser.add_argument("output_plot", help="Path to save the output plot (.png).")
    parser.add_argument("-id", "--aircraft_id", help="Specific aircraft ID to plot. If not provided, the aircraft with the most data points will be used.", default=None)
    args = parser.parse_args()
    
    if not os.path.exists(args.input_csv):
        print(f"Error: Input file not found at '{args.input_csv}'")
    else:
        df = pd.read_csv(args.input_csv)
        
        aircraft_id_to_plot = args.aircraft_id
        if not aircraft_id_to_plot:
            if not df.empty and 'Id' in df.columns:
                aircraft_id_to_plot = df['Id'].value_counts().index[0]
            else:
                print("No data or 'Id' column found in the CSV. Cannot determine which aircraft to plot.")
                aircraft_id_to_plot = None
        
        if aircraft_id_to_plot:
             # Convert aircraft ID to string to match pandas behavior
            plot_flight_data(df, str(aircraft_id_to_plot), args.output_plot)