import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors
import os
import argparse

def plot_3d_flight_path(df, aircraft_id, output_path):
    """
    Plots the 3D flight path of a specific aircraft and saves it to a file.
    """
    aircraft_df = df[df['Id'] == aircraft_id].copy()

    if aircraft_df.empty:
        print(f"No data found for aircraft ID: {aircraft_id}")
        return

    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111, projection='3d')

    # Create a color mapping for maneuvers
    maneuver_labels = aircraft_df['Maneuver_Label'].dropna().unique()
    colors = list(mcolors.TABLEAU_COLORS.values())
    maneuver_colors = {label: colors[i % len(colors)] for i, label in enumerate(maneuver_labels) if label != ''}
    maneuver_colors[''] = 'gray' # Default color for no maneuver

    # Plot the flight path with colors for maneuvers
    for i in range(len(aircraft_df) - 1):
        segment_df = aircraft_df.iloc[i:i+2]
        maneuver = segment_df['Maneuver_Label'].iloc[0]
        color = maneuver_colors.get(maneuver, 'gray')
        ax.plot(segment_df['Longitude'], segment_df['Latitude'], segment_df['Altitude'], color=color, alpha=0.7, linewidth=2)

    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_zlabel('Altitude (ft)')
    ax.set_title(f'3D Flight Path for Aircraft ID: {aircraft_id}')

    # Create a legend for the maneuvers
    legend_elements = [plt.Line2D([0], [0], color=color, lw=4, label=label) for label, color in maneuver_colors.items() if label and label != '']
    ax.legend(handles=legend_elements)

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
    plt.savefig(output_path)
    print(f"3D plot saved to {output_path}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize 3D flight path for a specific aircraft.")
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
            plot_3d_flight_path(df, str(aircraft_id_to_plot), args.output_plot)