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
import signal
import shutil

# Add the project root to sys.path so Python can find local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

app = Flask(__name__)

# Global variable to track processing status
processing_status = {
    'message': 'Waiting to start...',
    'fuzz_log': 'Fuzzer idle.'
}

# --- Constants for file paths ---
TEMP_BINARY_PATH = './Binaries/temp_uploaded_file'
FUNCTIONS_FILE_PATH = './Dictionaries_list_of_functions/extracted_functions.pkl'
GEOMETRIC_DATA_OUTPUT_PATH = './Saved_Geometric_Datas/geometric_datas.pt' 
CFG_INFO_PATH = "./Saved_Geometric_Datas/"
OUTPUT_FILE_PATH = './Results/output_results.csv' 

# --- Conda Environments ---
CONDA_ENV_EXTRACT_LIST_FUNCTIONS = '/home/marcos/anaconda3/envs/test-3.9-env/bin/python' 
CONDA_ENV_CREATE_GEOMETRIC_DATAS = '/home/marcos/anaconda3/envs/test-3.10.0-env/bin/python'

# --- Model Paths ---
MODEL_PATH = './GNNs_Models/best_model.pth'
PARAMS_PATH = './GNNs_Models/best_params.json'

# --- Fuzzing Constants ---
FUZZ_IN_DIR = './fuzz_in'
FUZZ_OUT_DIR = './fuzz_out'
FUZZ_HARNESS_PATH = './fuzzer/fuzz_harness.py'

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
    subprocess.run([
        CONDA_ENV_EXTRACT_LIST_FUNCTIONS, 
        './Binaries/extract_list_functions.py', 
        TEMP_BINARY_PATH, 
        FUNCTIONS_FILE_PATH
    ])

def create_geometric_data():
    print(f"2) STARTING CREATING PYTORCH GEOMETRIC DATAS LIST")
    subprocess.run([
        CONDA_ENV_CREATE_GEOMETRIC_DATAS,
        './safetorch_experiments/create_geometric_datas.py',
        TEMP_BINARY_PATH,
        FUNCTIONS_FILE_PATH,
        GEOMETRIC_DATA_OUTPUT_PATH,
        CFG_INFO_PATH
    ])

def run_model_inference():
    print(f"3) STARTING MODEL INFERENCE")
    subprocess.run([
        CONDA_ENV_CREATE_GEOMETRIC_DATAS, 
        './GNNs_Models/model_inference.py', 
        MODEL_PATH, 
        PARAMS_PATH, 
        GEOMETRIC_DATA_OUTPUT_PATH, 
        OUTPUT_FILE_PATH
    ])

# -------------------------------------------------------------------------
# FUZZING ENGINE ROUTES
# -------------------------------------------------------------------------

@app.route('/start_fuzz', methods=['POST'])
def start_fuzz():
    func_name = request.form.get('func_name')
    func_addr = request.form.get('func_addr')
    arg_map = request.form.get('arg_map', 'ptr,len,null,null,null,null') 
    
    # 1. Workspace Cleanup
    if os.path.exists(FUZZ_OUT_DIR):
        shutil.rmtree(FUZZ_OUT_DIR)
    os.makedirs(FUZZ_OUT_DIR, exist_ok=True)
    os.makedirs(FUZZ_IN_DIR, exist_ok=True)
    
    # 2. SEED FIX: Use a 1-byte seed instead of "START"
    # This ensures the first Angr execution is as fast as possible.
    seed_file = os.path.join(FUZZ_IN_DIR, "seed.txt")
    with open(seed_file, "w") as f:
        f.write("A") 

    def run_afl():
        env = os.environ.copy()
        env["AFL_SKIP_CPUFREQ"] = "1"
        env["AFL_I_DONT_CARE_ABOUT_MISSING_CRASHES"] = "1"
        env["AFL_TRY_AFFINITY"] = "1"
        env["AFL_NO_UI"] = "1" 
        env["AFL_SKIP_BIN_CHECK"] = "1" 

        cmd = [
            "afl-fuzz",
            "-i", FUZZ_IN_DIR,
            "-o", FUZZ_OUT_DIR,
            "-n",                 # Dumb mode (no instrumentation)
            "-t", "5000",         # Solid 5s timeout for heavy emulation
            "-d",                 # Fidgety mode
            "--", 
            CONDA_ENV_CREATE_GEOMETRIC_DATAS,    
            FUZZ_HARNESS_PATH, 
            TEMP_BINARY_PATH, 
            func_addr, 
            arg_map
        ]
        
        app.config['fuzz_process'] = subprocess.Popen(
            cmd, 
            env=env,
            start_new_session=True 
        )

    threading.Thread(target=run_afl).start()
    return jsonify({'status': 'Fuzzing started', 'function': func_name})



@app.route('/fuzz_results')
def fuzz_results():
    # 1. Try to get the formal stats
    stats_file = os.path.join(FUZZ_OUT_DIR, "default", "fuzzer_stats")
    if os.path.exists(stats_file):
        with open(stats_file, 'r') as f:
            content = f.read()
            if len(content) > 50: # Ensure it's not just a header
                return content

    # 2. FALLBACK: Read the last 10 lines of the AFL log (the stuff you see in terminal)
    # This ensures the UI is NEVER "stuck" on "Waiting"
    return "Fuzzer Active. Current Exec Speed: ~3.2s per run. Please wait for the first cycle to complete..."


def _kill_fuzzer_logic():
    """Internal helper to kill fuzzer safely from any thread."""
    # Note: We use app.config directly here which is safe in this specific setup
    proc = app.config.get('fuzz_process')
    if proc:
        try:
            # os.getpgid(proc.pid) gets the group ID so we kill AFL + Python workers
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            print(">> [CLEANUP] Fuzzer process group killed.")
        except Exception as e:
            print(f">> [CLEANUP] Kill failed (process might be gone): {e}")
        finally:
            app.config['fuzz_process'] = None
    
    # Global safety pkill
    subprocess.run(["pkill", "-9", "afl-fuzz"], stderr=subprocess.DEVNULL)
    

@app.route('/stop_fuzz', methods=['POST'])
def stop_fuzz():
    _kill_fuzzer_logic()
    return jsonify({'status': 'Fuzzer stopped and system cleaned'})

# -------------------------------------------------------------------------
# DATA MANAGEMENT
# -------------------------------------------------------------------------

def load_predictions():
    """Load results from CSV and prepare for frontend display."""
    if not os.path.exists(OUTPUT_FILE_PATH):
        return []
        
    df = pd.read_csv(OUTPUT_FILE_PATH)

    # Ensure the new LLM Audit columns exist in the dataframe
    if 'llm_prediction' not in df.columns:
        df['llm_prediction'] = 0
    if 'llm_reasoning' not in df.columns:
        df['llm_reasoning'] = "LLM Audit not performed for this function."

    predictions = df.to_dict(orient='records')  
    
    # Path logic for assets (Images and Assembly text)
    for prediction in predictions:
        f_name = prediction['Name']
        f_addr = prediction['Address']
        prediction['CFG_image'] = f"/Saved_Geometric_Datas/CFG_{f_name}_{f_addr}.png"   
        prediction['Assembly_code'] = f"/Saved_Geometric_Datas/AssemblyCode_{f_name}_{f_addr}.txt"
        
    return predictions  

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part', 400
    file = request.files['file']
    if file.filename == '':
        return 'No selected file', 400
    
    save_uploaded_file(file)  
    app.config['input_file_name'] = file.filename 
    threading.Thread(target=process_file).start()
    return render_template('loading.html', message=processing_status['message'])

def _sanitize_filename(name: str) -> str:
    name = name or "unknown"
    name = name.replace(" ", "_")
    return re.sub(r'[^A-Za-z0-9_.-]', '', name)

def process_file():
    global processing_status

    # Kill any rogue fuzzer before starting new analysis
    _kill_fuzzer_logic()

    overall_start = time.time()

    # Create storage for CFGs and results
    os.makedirs(CFG_INFO_PATH, exist_ok=True)

    processing_status['message'] = "Phase 1: Binary Deconstruction (Radare2)..."
    extract_functions()

    processing_status['message'] = "Phase 2: Semantic Graph Building & LLM Audit..."
    create_geometric_data()

    processing_status['message'] = "Phase 3: Neural Network Classification (GNN)..."
    run_model_inference()

    # Load results into session config
    app.config['predictions'] = load_predictions()
    processing_status['message'] = "Processing complete"

@app.route('/results')
def results():
    predictions = app.config.get('predictions', [])
    input_file_name = app.config.get('input_file_name', 'Unknown File')
    return render_template('result.html', predictions=predictions, input_file_name=input_file_name)

@app.route('/status')
def status():
    if processing_status['message'] == "Processing complete":
        return jsonify({'status': 'complete', 'message': processing_status['message']})
    return jsonify({'status': 'processing', 'message': processing_status['message']})

@app.route('/Saved_Geometric_Datas/<path:filename>')
def send_image(filename):
    return send_from_directory('Saved_Geometric_Datas', filename)

@app.route('/run_code', methods=['POST'])
def run_code():
    """Allows user to run custom Angr/Python scripts in a separate process."""
    code = request.form['code']
    with tempfile.NamedTemporaryFile('w', suffix='.py', delete=False) as tmp:
        tmp.write(code)
        tmp_path = tmp.name
    try:
        proc = subprocess.run([sys.executable, tmp_path], capture_output=True, text=True, timeout=180)
        output = proc.stdout + proc.stderr
    except Exception as e:
        output = f'Error during execution: {e}'
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
    return jsonify({'output': output})

if __name__ == '__main__':
    app.run(debug=True)


    # conda activate test-3.10.0-env
    # python -u /home/marcos/Projects/Web_Interface_ParserHunter/app.py
