import pandas as pd
import argparse
import os

def csv_to_acmi(input_path, output_path):
    """
    Converts a CSV file (in Tacview's detailed format) back to a basic .acmi format.
    """
    if not os.path.exists(input_path):
        print(f"Error: Input file not found at '{input_path}'")
        return

    df = pd.read_csv(input_path)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        # Write header
        f.write("FileType=text/acmi/tacview\n")
        f.write("FileVersion=2.2\n")
        
        # Set a reference time based on the first entry if available
        first_iso_time = df['ISO time'].iloc[0] if 'ISO time' in df.columns and not df.empty else "2023-01-01T00:00:00Z"
        f.write(f"0,ReferenceTime={first_iso_time}\n")

        # Group by the 'Relative Time' or 'Unix time' column
        time_col = 'Relative Time' if 'Relative Time' in df.columns else 'Unix time'

        for timestamp, group in df.groupby(time_col):
            f.write(f"#{timestamp}\n")
            for _, row in group.iterrows():
                # Create the kinematic data string
                kinematic_data = '|'.join(map(str, [
                    row.get('Longitude', ''),
                    row.get('Latitude', ''),
                    row.get('Altitude', ''),
                    row.get('Roll', ''),
                    row.get('Pitch', ''),
                    row.get('Yaw', '')
                ]))
                
                # Create the attribute string
                attributes = [
                    f"Type={row.get('Type', '')}",
                    f"Name={row.get('Name', '')}",
                    f"TAS={row.get('TAS', '')}",
                    f"G={row.get('G', '') if pd.notna(row.get('G')) else ''}",
                    f"VS={row.get('VS', '') if pd.notna(row.get('VS')) else ''}",
                    f"AOA={row.get('AOA', '') if pd.notna(row.get('AOA')) else ''}",
                    f"Color={row.get('Color', '')}",
                    f"Coalition={row.get('Coalition', '')}",
                    f"Country={row.get('Country', '')}",
                    f"Pilot={row.get('Pilot', '')}",
                ]
                attribute_str = ",".join(filter(None, attributes))
                
                object_id = row.get('Id', '0')
                f.write(f"{object_id},T={kinematic_data},{attribute_str}\n")

    print(f"Successfully converted {input_path} to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert a detailed flight CSV to a basic ACMI file.")
    parser.add_argument("input_csv", help="Path to the input CSV file.")
    parser.add_argument("output_acmi", help="Path to save the output .acmi file.")
    args = parser.parse_args()
    csv_to_acmi(args.input_csv, args.output_acmi)