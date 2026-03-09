# Web Interface for ParserHunter

This project provides a simple web interface for classifying functions as a parser or not from a binary executable file using a Graph Neural Network (GNN) model. The application is based on the work presented in the paper *ParserHunter: Identify Parsing Functions in Binary Code*, which introduces a method for identifying parsing functions in binaries. The interface allows users to upload binary files, processes them to extract function information, and presents the results in a user-friendly format.

## Features

- **Binary Analysis**: Upload ELF/PE binaries to extract function semantics via Radare2.
- **GNN Classification**: Automatically identify "Parser" vs "Non-Parser" functions using a pre-trained Graph Neural Network.
- **CFG Visualization**: Generate and view Control Flow Graphs (CFG) enriched with semantic node features.
- **LLM Parser Audit**: Use the Groq API to perform automated reasoning over assembly code and function requirements to confirm parser identity and logic.
- **Dynamic Fuzzing Engine**: Launch targeted AFL++ fuzzing sessions directly from the UI to exercise identified parser functions with custom register-to-argument mapping.
- **Symbolic Execution**: Built-in console for custom Angr scripts and symbolic analysis.

## Technologies Used

- **Flask**: A lightweight WSGI web application framework for Python, used for building the web interface.
- **Pandas**: A data manipulation and analysis library for Python, used for handling and processing data.
- **Subprocess**: For running external scripts and commands, facilitating the execution of various analysis tools.
- **Bootstrap**: For responsive web design, enhancing the user interface of the web application.
- **PyTorch & PyTorch Geometric**: Core GNN implementation for graph-based function classification.
- **Angr**: A Python framework for analyzing binaries, used for extracting control flow graphs and function information.
- **Radare2**: A set of utilities to examine binaries, used for identifying functions within the binary files.
- **AFL++**: High-performance fuzzer used to dynamically exercise identified parsing functions.
- **Asm2Vec**: A tool for generating embeddings from assembly code, used for enriching the features of the functions being analyzed.
- **SAFE**: Another tool for generating embeddings from assembly code, providing additional features for function classification.
- **Groq API**: High-speed LLM inference (Llama-3) used for assembly-level reasoning and parser identification.
- **NetworkX**: A library for the creation, manipulation, and study of complex networks, used for handling control flow graphs.
- **Matplotlib**: A plotting library for Python, used for visualizing control flow graphs.

## Usage

To run this project, follow these steps:

### 1. Set Up Conda Environments

You need to create three different conda environments for the various functionalities of the project. Please refer to the requirements.txt file for detailed instructions on setting up each environment and the required packages.

### 2. Modify app.py

Before running the application, you need to specify the paths for the conda environments in the app.py file. Locate the following constants and update them with the correct paths to your conda environments:

- **CONDA_ENV_EXTRACT_LIST_FUNCTIONS**: Path to the Python executable in the environment for extracting functions.
- **CONDA_ENV_CREATE_GEOMETRIC_DATAS**: Path to the Python executable in the environment for creating geometric data and running GNN model inference.
- **CONDA_ENV_ASM2VEC_INFERENCE**: Path to the Python executable in the environment for Asm2Vec model inference.

Example:
```python
CONDA_ENV_EXTRACT_LIST_FUNCTIONS = '/home/marcos/.conda/envs/test-3.9-env/bin/python'
CONDA_ENV_CREATE_GEOMETRIC_DATAS = '/home/marcos/anaconda3/envs/test-3.10.0-env/bin/python'
CONDA_ENV_ASM2VEC_INFERENCE = '/home/marcos/anaconda3/envs/asm2vec/bin/python'
```

### 3. Configure Environment Variables
To enable the LLM Parser Identification feature, export your Groq API key:
```bash
export GROQ_API_KEY='your_groq_key_here'
```

### 4. Run the Flask Application
Run the app.py file to start the Flask web server:

## Screenshots

Here are some screenshots of the web interface:

### 1. Function Classification Results
This screenshot shows the table of function classifications for a binary executable.

![Function Classification Table](./Screenshots/results_jsmn_2_page1_ordered_2.png)

### 2. Symbolic Execution with Angr
This screenshot shows the box where the user can input code to perform symbolic execution using Angr.

![Symbolic Execution Box](./Screenshots/symbolic_exe.png)
