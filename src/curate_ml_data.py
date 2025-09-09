import pandas as pd
import numpy as np
import os
import argparse

def curate_data(input_dir, output_dir_base, padding_seconds):
    """
    Scans labeled data, extracts clips of meaningful maneuvers with padding,
    and saves them to a new, curated directory for ML training.
    """
    if not os.path.isdir(input_dir):
        print(f"Error: Input directory not found: '{input_dir}'.")
        print("Please ensure you have run maneuver_recognition.py first.")
        return

    base_folder_name = os.path.basename(input_dir.rstrip('/\\'))
    curated_folder_name = base_folder_name.replace('_Labeled', '_Curated_For_ML')
    output_dir = os.path.join(output_dir_base, curated_folder_name)

    print(f"Curated data for ML will be saved in: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    total_clips_extracted = 0
    processed_file_count = 0

    for filename in os.listdir(input_dir):
        if not filename.endswith(".csv"):
            continue

        input_path = os.path.join(input_dir, filename)
        print(f"Processing {filename}...")

        try:
            df = pd.read_csv(input_path)
            if df.empty or 'Maneuver_Label' not in df.columns:
                continue
            
            maneuvers = df[df['Maneuver_Label'].notna() & (df['Maneuver_Label'] != '')].copy()
            
            if maneuvers.empty:
                print(f"  -> No maneuvers found in {filename}. Skipping.")
                continue


            maneuvers['block'] = (maneuvers['Maneuver_Label'] != maneuvers['Maneuver_Label'].shift()).cumsum()
            all_clips = []
            
            for _, block in maneuvers.groupby('block'):
                start_time, end_time = block['Time'].iloc[0], block['Time'].iloc[-1]
                padded_start_time, padded_end_time = start_time - padding_seconds, end_time + padding_seconds
                clip = df[(df['Time'] >= padded_start_time) & (df['Time'] <= padded_end_time)]
                all_clips.append(clip)
                total_clips_extracted += 1

            if not all_clips: continue

            curated_df = pd.concat(all_clips).drop_duplicates().sort_values(by='Time').reset_index(drop=True)
            output_path = os.path.join(output_dir, filename)
            curated_df.to_csv(output_path, index=False)
            processed_file_count += 1
            print(f"  -> Extracted {len(all_clips)} clips. Saved curated file.")

        except Exception as e:
            print(f"Error processing file {filename}: {e}")

    print("\nData curation complete.")
    print(f"Processed {processed_file_count} files and extracted a total of {total_clips_extracted} maneuver clips.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Curate labeled flight data to extract meaningful maneuver clips for ML training.")
    parser.add_argument("input_dir", help="Directory containing the labeled data folders (e.g., '..._Labeled/').")
    parser.add_argument("output_dir", help="Base directory to save the new curated data folder.")
    parser.add_argument("--padding", type=float, default=5.0, help="Seconds of padding to add before and after each maneuver clip.")
    args = parser.parse_args()
    curate_data(args.input_dir, args.output_dir, args.padding)