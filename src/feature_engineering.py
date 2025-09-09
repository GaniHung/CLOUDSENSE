import pandas as pd
import numpy as np
import os
import argparse

# --- CONSTANTS and calculation functions remain the same ---
FEET_TO_M = 0.3048
KNOTS_TO_MS = 0.514444
G = 9.80665
EARTH_RADIUS_M = 6371000
# (All calculation functions like calculate_rates_and_time, etc., are unchanged)
def calculate_rates_and_time(df):
    df = df.sort_values(by='Time').reset_index(drop=True)
    df['TimeDelta'] = df['Time'].diff().apply(lambda x: x if x > 0 else np.nan)
    df['RollRate'] = np.degrees(df['Roll'].diff() / df['TimeDelta'])
    df['PitchRate'] = np.degrees(df['Pitch'].diff() / df['TimeDelta'])
    yaw_diff = df['Yaw'].diff()
    yaw_diff[yaw_diff > np.pi] -= 2 * np.pi
    yaw_diff[yaw_diff < -np.pi] += 2 * np.pi
    df['YawRate'] = np.degrees(yaw_diff / df['TimeDelta'])
    return df
def calculate_velocity_from_position(df):
    lon_rad, lat_rad, alt_m = np.radians(df['Longitude']), np.radians(df['Latitude']), df['Altitude'] * FEET_TO_M
    dlat, dlon, y, x = lat_rad.diff(), lon_rad.diff(), np.sin(lon_rad.diff()) * np.cos(lat_rad), np.cos(lat_rad.shift()) * np.sin(lat_rad) - np.sin(lat_rad.shift()) * np.cos(lat_rad) * np.cos(lon_rad.diff())
    a = np.sin(dlat / 2.0)**2 + np.cos(lat_rad.shift()) * np.cos(lat_rad) * np.sin(dlon / 2.0)**2
    c, bearing, horizontal_dist = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a)), np.arctan2(y, x), EARTH_RADIUS_M * (2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a)))
    horizontal_v = horizontal_dist / df['TimeDelta']
    df['U_ms'], df['V_ms'], df['W_ms'], df['VS_ms'] = horizontal_v * np.cos(bearing), horizontal_v * np.sin(bearing), alt_m.diff() / df['TimeDelta'], -(alt_m.diff() / df['TimeDelta'])
    return df
def calculate_g_force(df):
    ax, ay, az = df['U_ms'].diff() / df['TimeDelta'], df['V_ms'].diff() / df['TimeDelta'], df['W_ms'].diff() / df['TimeDelta']
    accel_inertial = pd.DataFrame({'ax': ax, 'ay': ay, 'az': az}); accel_inertial['az'] += G
    phi, theta, cos_phi, sin_phi, cos_theta, sin_theta = df['Roll'], df['Pitch'], np.cos(df['Roll']), np.sin(df['Roll']), np.cos(df['Pitch']), np.sin(df['Pitch'])
    df['G_Normal'], df['G_Axial'], df['G_Lateral'] = ((-cos_phi * sin_theta * accel_inertial['ax'] - sin_phi * accel_inertial['ay'] + cos_phi * cos_theta * accel_inertial['az']) / G), ((cos_theta * accel_inertial['ax'] + sin_theta * accel_inertial['az']) / G), ((sin_phi * sin_theta * accel_inertial['ax'] - cos_phi * accel_inertial['ay'] + sin_phi * cos_theta * accel_inertial['az']) / G)
    return df
def calculate_performance_features(df):
    df['Speed_ms'] = np.sqrt(df['U_ms']**2 + df['V_ms']**2 + df['W_ms']**2)
    df['Altitude_m'] = df['Altitude'] * FEET_TO_M
    if 'TAS' in df.columns: df['Speed_ms'] = (pd.to_numeric(df['TAS'], errors='coerce') * KNOTS_TO_MS).fillna(df['Speed_ms'])
    df['TurnRate'] = np.degrees((G * np.sqrt(np.maximum(0, df['G_Normal']**2 - 1))) / df['Speed_ms'].replace(0, np.nan))
    df['SpecificEnergy'] = df['Altitude_m'] + (df['Speed_ms']**2) / (2 * G)
    df['SpecificPower'] = df['SpecificEnergy'].diff() / df['TimeDelta']
    return df

def feature_engineering(input_dir, output_dir_base):
    # --- MODIFIED: No longer looks for 'Aircraft' subdirectory ---
    if not os.path.isdir(input_dir):
        print(f"Error: Input directory not found: '{input_dir}'.")
        return

    base_folder_name = os.path.basename(input_dir.rstrip('/\\'))
    processed_folder_name = base_folder_name.replace('_Partitioned', '_Processed')
    # --- MODIFIED: Output path is also flattened ---
    aircraft_output_dir = os.path.join(output_dir_base, processed_folder_name)
    
    os.makedirs(aircraft_output_dir, exist_ok=True)
    processed_file_count = 0
    print(f"Starting feature engineering for files in '{input_dir}'...")
    
    for filename in os.listdir(input_dir):
        if filename.endswith(".csv"):
            try:
                df = pd.read_csv(os.path.join(input_dir, filename), low_memory=False)
                if df.empty or len(df) < 3: continue
                df['Id'] = os.path.splitext(filename)[0]
                numeric_cols = ['Time', 'Longitude', 'Latitude', 'Altitude', 'Roll', 'Pitch', 'Yaw', 'TAS', 'VS']
                for col in numeric_cols:
                    if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')
                    else: df[col] = np.nan
                df.dropna(subset=['Time', 'Longitude', 'Latitude', 'Altitude', 'Roll', 'Pitch', 'Yaw'], inplace=True)
                if df.empty or len(df) < 3: continue
                
                processed_df = calculate_rates_and_time(df)
                processed_df['Roll'], processed_df['Pitch'], processed_df['Yaw'] = np.radians(processed_df['Roll']), np.radians(processed_df['Pitch']), np.radians(processed_df['Yaw'])
                processed_df = calculate_velocity_from_position(processed_df)
                processed_df = calculate_g_force(processed_df)
                processed_df = calculate_performance_features(processed_df)
                processed_df['Roll'], processed_df['Pitch'], processed_df['Yaw'] = np.degrees(processed_df['Roll']), np.degrees(processed_df['Pitch']), np.degrees(processed_df['Yaw'])
                processed_df = processed_df.iloc[1:].reset_index(drop=True)
                if processed_df.empty: continue

                columns_to_keep = ['Id', 'Time', 'Longitude', 'Latitude', 'Altitude', 'Roll', 'Pitch', 'Yaw', 'TAS', 'Speed_ms', 'VS_ms', 'G_Normal', 'G_Axial', 'G_Lateral', 'RollRate', 'PitchRate', 'YawRate', 'TurnRate', 'SpecificEnergy', 'SpecificPower']
                final_df = processed_df.reindex(columns=columns_to_keep)
                final_df.to_csv(os.path.join(aircraft_output_dir, filename), index=False, float_format='%.4f')
                processed_file_count += 1
            except Exception as e:
                print(f"Error processing file {filename}: {e}")
        
    print(f"\nFeature engineering complete. Processed {processed_file_count} aircraft files.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate features from partitioned aircraft data.")
    parser.add_argument("input_dir", help="Directory containing partitioned data (e.g., '..._Partitioned/').")
    parser.add_argument("output_dir", help="Base directory to save the new processed data folder.")
    args = parser.parse_args()
    feature_engineering(args.input_dir, args.output_dir)