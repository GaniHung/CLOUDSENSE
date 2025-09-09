import csv
import os
import re
import argparse
from collections import defaultdict
import zipfile
import tempfile

def parse_acmi_content(file_stream, output_dir):
    """
    Parses ACMI content, focusing only on aircraft and saving them to a single
    flat directory. Dynamically discovers all attributes to use as headers.
    """
    BASE_KINEMATIC_HEADERS = [
        'Longitude', 'Latitude', 'Altitude', 'Roll', 'Pitch', 'Yaw', 'U', 'V', 'W'
    ]
    
    # --- MODIFIED: Only target Aircraft. Weapons will now be ignored. ---
    TARGET_OBJECT_TYPES = {
        'Aircraft': 'Air+FixedWing'
    }

    object_ids_by_type = defaultdict(set)
    object_type_map = {}
    records_by_id = defaultdict(list)
    last_known_states = {}
    found_attribute_keys = set()
    
    max_kinematic_vals = len(BASE_KINEMATIC_HEADERS)
    current_time = 0.0

    t_line_pattern = re.compile(r'^([0-9a-fA-F]+),T=(.*)$')
    time_pattern = re.compile(r'^#(\d+(\.\d+)?)$')
    type_pattern = re.compile(r'Type=([a-zA-Z0-9\+_-]+)')

    print("Parsing ACMI content (1st Pass: Discovering aircraft data points)...")
    
    for line_bytes in file_stream:
        try:
            line = line_bytes.decode('utf-8').strip()
        except UnicodeDecodeError:
            continue

        if not line: continue

        time_match = time_pattern.match(line)
        if time_match:
            current_time = float(time_match.group(1))
            continue

        if line.startswith('0,'): continue

        match = t_line_pattern.match(line)
        if match:
            object_id = match.group(1).lower()
            data_str = match.group(2)
            
            if object_id not in object_type_map:
                type_match = type_pattern.search(data_str)
                if type_match:
                    full_type = type_match.group(1)
                    for category, type_string in TARGET_OBJECT_TYPES.items():
                        if full_type == type_string:
                            object_type_map[object_id] = category
                            object_ids_by_type[category].add(object_id)
                            break
            
            if object_id not in object_type_map:
                continue

            current_state = last_known_states.get(object_id, {})
            parts = data_str.split(',', 1)
            kinematic_values_str = parts[0]
            attributes_str = parts[1] if len(parts) > 1 else ''
            
            kinematic_values = kinematic_values_str.split('|')
            max_kinematic_vals = max(max_kinematic_vals, len(kinematic_values))
            
            for i, value in enumerate(kinematic_values):
                if value and i < len(BASE_KINEMATIC_HEADERS):
                    current_state[BASE_KINEMATIC_HEADERS[i]] = value
            
            if attributes_str:
                for attr_pair in attributes_str.split(','):
                    if '=' in attr_pair:
                        key, value = attr_pair.split('=', 1)
                        current_state[key] = value
                        found_attribute_keys.add(key)
            
            last_known_states[object_id] = current_state
            
            records_by_id[object_id].append({
                'time': current_time,
                'state': current_state.copy()
            })

    print(f"Parsing complete. Found {len(object_ids_by_type.get('Aircraft', set()))} aircraft.")
    print(f"Discovered {len(found_attribute_keys)} unique attributes.")

    final_kinematic_headers = BASE_KINEMATIC_HEADERS[:]
    if max_kinematic_vals > len(BASE_KINEMATIC_HEADERS):
        for i in range(len(BASE_KINEMATIC_HEADERS), max_kinematic_vals):
            final_kinematic_headers.append(f'ExtraValue_{i}')
            
    sorted_attribute_keys = sorted(list(found_attribute_keys))
    final_header = ['Time'] + final_kinematic_headers + sorted_attribute_keys

    for category, object_ids in object_ids_by_type.items():
        if not object_ids: continue

        print(f"Generating {len(object_ids)} CSV files in '{output_dir}'...")

        for object_id in object_ids:
            records = records_by_id.get(object_id, [])
            if not records: continue

            output_csv_path = os.path.join(output_dir, f"{object_id}.csv")
            try:
                with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(final_header)
                    
                    for record in records:
                        row_data = [record['time']]
                        for key in final_kinematic_headers:
                            row_data.append(record['state'].get(key, ''))
                        for key in sorted_attribute_keys:
                            row_data.append(record['state'].get(key, ''))
                        writer.writerow(row_data)
            except Exception as e:
                print(f"Failed to write file '{output_csv_path}': {e}")


def convert_acmi_to_partitioned_csv(acmi_filepath, output_dir=None, session_name=None):
    if not os.path.exists(acmi_filepath):
        print(f"Error: Input file '{acmi_filepath}' not found.")
        return

    base_output_dir = output_dir if output_dir else os.path.dirname(acmi_filepath)
    
    if session_name:
        partition_folder_name = f"{session_name}_FlightData_Partitioned"
    else:
        base_filename = os.path.basename(acmi_filepath)
        if base_filename.lower().endswith('.zip.acmi'):
            folder_name, _ = os.path.splitext(base_filename)
            folder_name, _ = os.path.splitext(folder_name)
        else:
            folder_name, _ = os.path.splitext(base_filename)
        partition_folder_name = f"{folder_name}_FlightData_Partitioned"

    flight_data_partition_dir = os.path.join(base_output_dir, partition_folder_name)

    os.makedirs(flight_data_partition_dir, exist_ok=True)

    if acmi_filepath.lower().endswith('.zip.acmi'):
        print(f"Detected '.zip.acmi' file. Unzipping...")
        with tempfile.TemporaryDirectory() as tempdir:
            try:
                with zipfile.ZipFile(acmi_filepath, 'r') as zip_ref:
                    zip_ref.extractall(tempdir)
                
                acmi_file_in_zip = None
                for item in os.listdir(tempdir):
                    if item.lower().endswith('.acmi'):
                        acmi_file_in_zip = os.path.join(tempdir, item)
                        break
                
                if acmi_file_in_zip:
                    print(f"Found '{os.path.basename(acmi_file_in_zip)}' in archive. Processing...")
                    with open(acmi_file_in_zip, 'rb') as f:
                        parse_acmi_content(f, flight_data_partition_dir)
                else:
                    print("Error: No .acmi file found inside the zip archive.")
                    return
            except zipfile.BadZipFile:
                print(f"Error: '{acmi_filepath}' is not a valid zip file.")
                return
    else:
        print(f"Processing standard '.acmi' file...")
        with open(acmi_filepath, 'rb') as f:
            parse_acmi_content(f, flight_data_partition_dir)
            
    print("\n--- Conversion Complete ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Converts Tacview .acmi files to partitioned CSVs for aircraft, dynamically discovering all headers."
    )
    parser.add_argument(
        "input_file",
        help="Path to the input .acmi or .zip.acmi file."
    )
    parser.add_argument(
        "-o", "--output_dir", 
        help="Base directory to save the output folder. Defaults to the input file's directory.", 
        default=None
    )
    parser.add_argument(
        "-sn", "--session_name", 
        help="A specific session name to use for the output folder, overriding the default naming scheme.", 
        default=None
    )
    args = parser.parse_args()
    convert_acmi_to_partitioned_csv(args.input_file, args.output_dir, args.session_name)