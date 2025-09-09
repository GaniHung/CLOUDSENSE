import pandas as pd
import numpy as np
import argparse
import os

# All functions (get_ffp_label, ffp_recognition, maneuver_recognition, recognize_complex_maneuvers) are unchanged.
def get_ffp_label(row):
    roll_thresh, pitch_thresh, vs_thresh, g_thresh_high, roll_rate_thresh, pitch_rate_thresh, turn_rate_thresh, ps_thresh, inverted_thresh, nose_high_thresh, nose_low_thresh = 10, 10, 2.032, 1.1, 5, 5, 3, 10, 135, 45, -45
    g_normal, roll_rate, pitch_rate, roll, pitch, vs, turn_rate, specific_power = row.get('G_Normal', 1.0), row.get('RollRate', 0.0), row.get('PitchRate', 0.0), row.get('Roll', 0.0), row.get('Pitch', 0.0), row.get('VS_ms', 0.0), row.get('TurnRate', 0.0), row.get('SpecificPower', 0.0)
    if abs(roll) > inverted_thresh: return "Inverted_Flight"
    if pitch > nose_high_thresh: return "Nose_High_Climb"
    if pitch < nose_low_thresh: return "Nose_Low_Dive"
    if abs(roll) >= roll_thresh and g_normal >= g_thresh_high and abs(turn_rate) > turn_rate_thresh:
        if abs(vs) < vs_thresh: return "Level_Turn"
        elif vs > vs_thresh: return "Climbing_Turn"
        else: return "Descending_Turn"
    if abs(roll) < roll_thresh:
        if abs(vs) < vs_thresh: return "Steady_Level_Flight"
        elif specific_power > ps_thresh or vs > vs_thresh: return "Steady_Climb"
        else: return "Steady_Descent"
    if abs(roll_rate) > roll_rate_thresh: return "Roll_Motion"
    if abs(pitch_rate) > pitch_rate_thresh: return "Pitch_Motion"
    return "Undefined"
def ffp_recognition(df):
    print("Performing FFP recognition..."); df['FFP_Label'] = df.fillna(0).apply(get_ffp_label, axis=1); return df
def maneuver_recognition(df):
    print("Performing simple maneuver recognition..."); df['Maneuver_Label'] = ''
    maneuver_definitions = {'Sustained_Turn': {'core_ffp': 'Level_Turn', 'min_duration': 3.0}, 'Chandelle': {'core_ffp': 'Climbing_Turn', 'min_duration': 3.0}}
    for _, group in df.groupby('Id'):
        group['ffp_block'] = (group['FFP_Label'] != group['FFP_Label'].shift()).cumsum()
        for _, block in group.groupby('ffp_block'):
            ffp_label = block['FFP_Label'].iloc[0]
            for maneuver_name, params in maneuver_definitions.items():
                if ffp_label == params['core_ffp'] and (block['Time'].iloc[-1] - block['Time'].iloc[0]) >= params['min_duration']: df.loc[block.index, 'Maneuver_Label'] = maneuver_name
    return df
def recognize_complex_maneuvers(df):
    print("Performing complex maneuver recognition...")
    complex_maneuver_patterns = {'Split_S': {'sequence': ['Roll_Motion', 'Inverted_Flight', 'Nose_Low_Dive', 'Pitch_Motion'], 'min_total_duration': 4.0, 'max_total_duration': 20.0}, 'Immelmann': {'sequence': ['Pitch_Motion', 'Nose_High_Climb', 'Roll_Motion'], 'min_total_duration': 4.0, 'max_total_duration': 20.0}, 'Aileron_Roll': {'sequence': ['Roll_Motion', 'Inverted_Flight', 'Roll_Motion'], 'min_total_duration': 2.0, 'max_total_duration': 8.0}}
    for aircraft_id, group in df.groupby('Id'):
        ffp_blocks = group.groupby((group['FFP_Label'] != group['FFP_Label'].shift()).cumsum()); ffp_sequence_info = []
        for _, block in ffp_blocks:
            if not block.empty and block['FFP_Label'].iloc[0] != "Undefined": ffp_sequence_info.append({'label': block['FFP_Label'].iloc[0], 'start_time': block['Time'].iloc[0], 'end_time': block['Time'].iloc[-1], 'indices': block.index})
        num_blocks = len(ffp_sequence_info)
        for maneuver_name, params in complex_maneuver_patterns.items():
            pattern_len = len(params['sequence'])
            if num_blocks < pattern_len: continue
            for i in range(num_blocks - pattern_len + 1):
                if [ffp_sequence_info[j]['label'] for j in range(i, i + pattern_len)] == params['sequence']:
                    total_duration = ffp_sequence_info[i + pattern_len - 1]['end_time'] - ffp_sequence_info[i]['start_time']
                    if params.get('min_total_duration', 0) <= total_duration <= params.get('max_total_duration', float('inf')):
                        print(f"Found '{maneuver_name}' for aircraft {aircraft_id}!"); all_indices = []
                        for j in range(i, i + pattern_len): all_indices.extend(ffp_sequence_info[j]['indices'])
                        df.loc[all_indices, 'Maneuver_Label'] = maneuver_name
    return df

def main(input_dir, output_dir_base):
    if not os.path.isdir(input_dir):
        print(f"Error: Input directory not found: '{input_dir}'.")
        return
        
    base_folder_name = os.path.basename(input_dir.rstrip('/\\'))
    labeled_folder_name = base_folder_name.replace('_Processed', '_Labeled')
    output_dir = os.path.join(output_dir_base, labeled_folder_name)
    
    print(f"Output will be saved in: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    file_count = 0
    for filename in os.listdir(input_dir):
        if filename.endswith(".csv"):
            input_path = os.path.join(input_dir, filename)
            print(f"Processing {filename}...")
            
            df = pd.read_csv(input_path)
            if df.empty: continue

            ffp_df = ffp_recognition(df)
            maneuver_df = maneuver_recognition(ffp_df)
            final_df = recognize_complex_maneuvers(maneuver_df)
            
            output_path = os.path.join(output_dir, filename)
            final_df.to_csv(output_path, index=False)
            file_count += 1
            
    print(f"\nLabeling complete. Processed and saved {file_count} files.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply FFP and maneuver labels to processed flight data.")
    parser.add_argument("input_dir", help="Directory containing processed data (e.g., '..._Processed/').")
    parser.add_argument("output_dir", help="Base directory to save the new labeled data folder.")
    args = parser.parse_args()
    main(args.input_dir, args.output_dir)