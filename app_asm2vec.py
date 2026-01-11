from flask import Flask, request, render_template, jsonify, send_from_directory
import subprocess
import pandas as pd
import sys
import os
import threading
import tempfile
import time
import datetime
import re

app = Flask(__name__)

# Global variable to track processing status
processing_status = {
    'message': 'Waiting to start...',
}

# Constants for file paths
TEMP_BINARY_PATH = './Binaries/temp_uploaded_file'

# Constants for function extraction subprocess
CONDA_ENV_EXTRACT_LIST_FUNCTIONS = '/home/marcos/.conda/envs/test-3.9-env/bin/python'  # Conda env for extract_list_functions
FUNCTIONS_FILE_PATH = './Dictionaries_list_of_functions/extracted_functions.pkl'

# Constants for geometric data creation subprocess
CONDA_ENV_CREATE_GEOMETRIC_DATAS = '/home/marcos/anaconda3/envs/test-3.10.0-env/bin/python'  # Conda env for create_geometric_datas and GNN model inference
CONDA_ENV_ASM2VEC_INFERENCE = '/home/marcos/anaconda3/envs/asm2vec/bin/python'  # Conda env for asm2vec model inference
ASM2VEC_SCRIPT_PATH = './Asm2Vec/asm2vec_inference.py' 
GEOMETRIC_DATA_OUTPUT_PATH = './Saved_Geometric_Datas/geometric_datas.pt' 
CFG_INFO_PATH = "./Saved_Geometric_Datas/"

# Constants for model inference subprocess
MODEL_PATH = './GNNs_Models/best_model.pth'
PARAMS_PATH = './GNNs_Models/best_params.json'
OUTPUT_FILE_PATH = './Results/output_results.csv' 

@app.route('/')
def index():
    """Render the upload form."""
    return render_template('index.html')

def save_uploaded_file(file):
    """Save the uploaded file to a temporary location."""
    binary_data = file.read()
    with open(TEMP_BINARY_PATH, 'wb') as temp_file:
        temp_file.write(binary_data)

def extract_functions():
    print(f"1) STARTING EXTRACTING LIST OF FUNCTIONS")

    """Call the external script to extract functions."""
    subprocess.run([
        CONDA_ENV_EXTRACT_LIST_FUNCTIONS, 
        './Binaries/extract_list_functions.py', 
        TEMP_BINARY_PATH, 
        FUNCTIONS_FILE_PATH
    ])

def create_geometric_data():
    print(f"2) STARTING CREATING PYTORCH GEOMETRIC DATAS LIST")
    
    """Call the create_geometric_datas script."""
    subprocess.run([
        CONDA_ENV_CREATE_GEOMETRIC_DATAS, 
        './Saved_Geometric_Datas/create_geometric_datas.py', 
        TEMP_BINARY_PATH, 
        FUNCTIONS_FILE_PATH, 
        CONDA_ENV_ASM2VEC_INFERENCE, 
        ASM2VEC_SCRIPT_PATH, 
        GEOMETRIC_DATA_OUTPUT_PATH,
        CFG_INFO_PATH
    ])

def run_model_inference():
    print(f"3) STARTING MODEL INFERENCE")

    """Call the model inference script."""
    subprocess.run([
        CONDA_ENV_CREATE_GEOMETRIC_DATAS, 
        './GNNs_Models/model_inference.py', 
        MODEL_PATH, 
        PARAMS_PATH, 
        GEOMETRIC_DATA_OUTPUT_PATH, 
        OUTPUT_FILE_PATH
    ])

def load_predictions():
    """Load predictions from the output CSV file."""
    df = pd.read_csv(OUTPUT_FILE_PATH)
    # Convert DataFrame to a list of dictionaries
    predictions = df.to_dict(orient='records')  
    
    # Add CFG image path to each prediction
    for prediction in predictions:
        function_name = prediction['Name']  # Function name
        hex_address = prediction['Address']  # Function address
        # Extract file names
        image_filename = f"CFG_{function_name}_{hex_address}.png" # Name of the CFG png image
        assembly_filename = f"AssemblyCode_{function_name}_{hex_address}.txt" # Name of the assembly txt file
        # Add columns with the retrieved files
        prediction['CFG_image'] = f"./Saved_Geometric_Datas/{image_filename}"   
        prediction['Assembly_code'] = f"./Saved_Geometric_Datas/{assembly_filename}"

    return predictions  



@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and process the binary file."""
    if 'file' not in request.files:
        return 'No file part', 400
    
    file = request.files['file']
    if file.filename == '':
        return 'No selected file', 400
    
    # Save the uploaded binary file
    save_uploaded_file(file)  
    
    # Store the input file name in the session
    app.config['input_file_name'] = file.filename  # Store the input file name
    
    # Start processing in the background
    threading.Thread(target=process_file).start()
    
    # Show loading page
    return render_template('loading.html', message=processing_status['message'])  # Initial loading page


def _sanitize_filename(name: str) -> str:
    """Return a safe filename-like string (keeps ascii letters, digits, - and _)."""
    name = name or "unknown"
    # replace spaces with underscore
    name = name.replace(" ", "_")
    # keep only safe chars
    name = re.sub(r'[^A-Za-z0-9_.-]', '', name)
    return name


def process_file():
    """Process the uploaded file in the background."""
    global processing_status

    # overall wall-clock start
    overall_start = time.time()

    # Ensure CFG_INFO_PATH exists
    try:
        os.makedirs(CFG_INFO_PATH, exist_ok=True)
    except Exception as e:
        print(f"Warning: could not create directory {CFG_INFO_PATH}: {e}")

    # Step 1: Extract the list of functions to analyse in the form of a dictionary {name: address}
    #         filtering out those with a cfg with less than 3 nodes / edges
    processing_status['message'] = "Identifying all the functions from the loaded binary executable"
    t1_start = time.time()
    extract_functions()
    t1_end = time.time()
    step1_time = t1_end - t1_start
    print(f"Step 1 completed in {step1_time:.2f} seconds")

    # Step 2: Create a list of Pytorch Geometric datas (enriching the CFG with the Asm2Vec mode)
    #         from the dictionary of the extracted functions to be analyzed
    processing_status['message'] = "Creating the enriched Control Flow Graph for each function"
    t2_start = time.time()
    create_geometric_data()
    t2_end = time.time()
    step2_time = t2_end - t2_start
    print(f"Step 2 completed in {step2_time:.2f} seconds")

    # Step 3: Run model inference on the list of Pytorch Geometric datas previously created
    #         and save the results
    processing_status['message'] = "Use the model to classify the identified functions"
    t3_start = time.time()
    run_model_inference()
    t3_end = time.time()
    step3_time = t3_end - t3_start
    print(f"Step 3 completed in {step3_time:.2f} seconds")

    # compute totals
    total_steps_time = step1_time + step2_time + step3_time
    overall_end = time.time()
    overall_elapsed = overall_end - overall_start

    # Save timing report to a text file
    try:
        input_file_name = app.config.get('input_file_name', 'unknown_input')
        safe_name = _sanitize_filename(input_file_name)
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        timings_filename = f"timings_{safe_name}_{timestamp}.txt"
        timings_path = os.path.join(CFG_INFO_PATH, timings_filename)

        lines = [
            f"Timing Report - {datetime.datetime.now().isoformat()}",
            f"Input file: {input_file_name}",
            "",
            f"Step 1 - Extract list of functions: {step1_time:.2f} seconds",
            f"Step 2 - Create Pytorch Geometric datas: {step2_time:.2f} seconds",
            f"Step 3 - Model inference: {step3_time:.2f} seconds",
            "",
            f"Total time (sum of steps 1..3): {total_steps_time:.2f} seconds",
            f"Overall wall-clock elapsed (start to end of process_file): {overall_elapsed:.2f} seconds",
            "",
            "Notes:",
            "- The 'Total time (sum of steps 1..3)' is the sum of individual step durations.",
            "- 'Overall wall-clock elapsed' may differ slightly due to small overheads (writing files, loading predictions, etc.)."
        ]

        with open(timings_path, 'w') as f:
            f.write("\n".join(lines))

        print(f"Timings saved to {timings_path}")
        # Optionally expose the timings path in processing_status so frontend can link to it
        processing_status['timings_file'] = timings_path

    except Exception as e:
        print(f"Error saving timings file: {e}")

    # Load predictions from the output file
    try:
        predictions = load_predictions()
    except Exception as e:
        print(f"Error loading predictions: {e}")
        predictions = []

    # Save predictions to a session or a temporary storage to access later
    app.config['predictions'] = predictions

    # Mark processing as complete
    processing_status['message'] = "Processing complete"


@app.route('/results')
def results():
    """Render the results page."""
    predictions = app.config.get('predictions', [])
    input_file_name = app.config.get('input_file_name', 'Unknown File')  # Get the input file name
    return render_template('result.html', predictions=predictions, input_file_name=input_file_name)  # Pass the input file name

@app.route('/status')
def status():
    """Check the status of the processing."""
    # Check if processing is complete
    if processing_status['message'] == "Processing complete":
        return jsonify({'status': 'complete', 'message': processing_status['message']})
    return jsonify({'status': 'processing', 'message': processing_status['message']})

@app.route('/Saved_Geometric_Datas/<path:filename>')
def send_image(filename):
    return send_from_directory('Saved_Geometric_Datas', filename)

@app.route('/run_code', methods=['POST'])
def run_code():
    code = request.form['code']

    # 1) Write the code out to a temporary .py file
    with tempfile.NamedTemporaryFile('w', suffix='.py', delete=False) as tmp:
        tmp.write(code)
        tmp_path = tmp.name

    try:
        # 2) Run it in a brand-new Python interpreter subprocess
        #    - capture both stdout and stderr
        #    - enforce a timeout (e.g. 15 seconds)
        proc = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=180, # timeout in seconds
        )
        # 3) Combine stdout+stderr so the user sees everything
        output = proc.stdout + proc.stderr

    except subprocess.TimeoutExpired:
        output = 'Error: your code took too long (timeout after 15s).'
    except Exception as e:
        # any other failure to launch the process
        output = f'Error launching subprocess: {e}'
    finally:
        # 4) Clean up the temp file
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    return jsonify({'output': output})

if __name__ == '__main__':
    app.run(debug=True)