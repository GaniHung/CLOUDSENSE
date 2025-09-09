import os
import subprocess
import argparse
import hashlib

def run_command(command, step_name):
    """Executes a command line command and prints its status."""
    print(f"\n{'='*20}\n[RUNNING] {step_name}\n{'='*20}")
    print(f"Executing: {' '.join(command)}\n")
    try:
        subprocess.run(command, check=True, text=True)
        print(f"[SUCCESS] {step_name} completed.")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] {step_name} failed.")
        print(f"Return code: {e.returncode}")
        print(f"Output:\n{e.stdout}\n{e.stderr}")
        exit(1)
    except FileNotFoundError:
        print(f"[ERROR] Command 'python' not found or script path is incorrect.")
        print("Please ensure Python is in your system's PATH.")
        exit(1)

def main():
    """
    Main function to run the data processing pipeline with step control and deterministic naming.
    """
    
    pipeline_steps = [
        {"name": "Step 1: ACMI Conversion", "short_name": "convert", "command_template": ["python", "src/acmi_converter.py", "{input_file}", "-o", "{output_dir}", "-sn", "{session_name}"]},
        {"name": "Step 2: Feature Engineering", "short_name": "feature", "command_template": ["python", "src/feature_engineering.py", "{partitioned_dir}", "{output_dir}"]},
        {"name": "Step 3: Maneuver Recognition", "short_name": "recog", "command_template": ["python", "src/maneuver_recognition.py", "{processed_dir}", "{output_dir}"]},
        {"name": "Step 4: Curate ML Data", "short_name": "curate", "command_template": ["python", "src/curate_ml_data.py", "{labeled_dir}", "{output_dir}", "--padding", "5"]},
        {"name": "Step 5: Prepare Data for ML", "short_name": "prepare", "command_template": ["python", "src/prepare_data_for_ml.py", "{curated_dir}", "{sequences_path}", "{labels_path}"]},
        {"name": "Step 6: Train LSTM Model", "short_name": "train", "command_template": ["python", "src/train_lstm.py", "{sequences_path}", "{labels_path}", "{model_path}"]}
    ]

    step_choices = [step['short_name'] for step in pipeline_steps]
    step_map = {step['short_name']: i for i, step in enumerate(pipeline_steps)}
    
    help_text_lines = ["Choose a step using its short name:"]
    for step in pipeline_steps:
        help_text_lines.append(f"  {step['short_name']:<10} - {step['name']}")
    step_help = "\n".join(help_text_lines)

    parser = argparse.ArgumentParser(
        description="Run the complete aircraft maneuver recognition pipeline. Generates a unique, consistent name for each input file.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("input_file", help="Path to the input .acmi or .zip.acmi file.")
    parser.add_argument("output_dir", help="Path to the base output directory for all processed data.")

    step_control_group = parser.add_mutually_exclusive_group()
    step_control_group.add_argument("--start-step", choices=step_choices, default=step_choices[0], help=f"Start the pipeline from this step.\n(default: {step_choices[0]})\n\n{step_help}")
    step_control_group.add_argument("--single-step", choices=step_choices, help=f"Run only a single specified step.\n\n{step_help}")
    
    args = parser.parse_args()

    # --- 1. Configuration: Generate a deterministic unique name and derive all paths ---
    hasher = hashlib.md5()
    hasher.update(args.input_file.encode('utf-8'))
    session_id = hasher.hexdigest()[:6]
    base_name = f"Run-{session_id}"
    
    print(f"Input file: {args.input_file}")
    print(f"Generated consistent session name for output: '{base_name}'")

    path_context = {
        "session_name": base_name,
        "input_file": args.input_file,
        "output_dir": args.output_dir,
        "partitioned_dir": os.path.join(args.output_dir, f"{base_name}_FlightData_Partitioned"),
        "processed_dir": os.path.join(args.output_dir, f"{base_name}_FlightData_Processed"),
        "labeled_dir": os.path.join(args.output_dir, f"{base_name}_FlightData_Labeled"),
        "curated_dir": os.path.join(args.output_dir, f"{base_name}_FlightData_Curated_For_ML"),
        "ml_output_dir": os.path.join(args.output_dir, "ml_data"),
        "sequences_path": os.path.join(args.output_dir, "ml_data", f"{base_name}_sequences.npy"),
        "labels_path": os.path.join(args.output_dir, "ml_data", f"{base_name}_labels.npy"),
        "model_path": os.path.join("models", f"{base_name}_lstm_model.h5")
    }

    os.makedirs(path_context["ml_output_dir"], exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    for step in pipeline_steps:
        step["command"] = [p.format(**path_context) for p in step["command_template"]]

    # --- 2. Execution Logic ---
    if args.single_step:
        step_index = step_map[args.single_step]
        step_to_run = pipeline_steps[step_index]
        run_command(step_to_run["command"], step_to_run["name"])
    else:
        start_index = step_map[args.start_step]
        steps_to_run = pipeline_steps[start_index:]
        for step in steps_to_run:
            run_command(step["command"], step["name"])

    print(f"\n{'='*20}\nPIPELINE EXECUTION FINISHED.\n{'='*20}")
    print(f"All outputs for this run are named with the consistent prefix: '{base_name}'")

if __name__ == "__main__":
    main()
