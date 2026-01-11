import angr
import torch
import pickle
import random
import numpy as np
import sys 
# Utility functions
import from_CFG_to_DataGeometric 

def load_and_convert_pkl(file_path):
    """Load a pickle file and convert it to a dictionary."""
    try:
        with open(file_path, 'rb') as file:
            data = pickle.load(file)

            # Check if the loaded data is a dictionary
            if not isinstance(data, dict):
                raise ValueError("The loaded data is not a dictionary.")
            else:
                print(f"Loaded dictionary of {len(data)} functions!")    
                return data

    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    except Exception as e:
        print(f"Error: {e}")

def create_and_save_geometric_datas_list(path_binary, path_dictionary, path_python_executable, path_script_asm2vec, path_output_file, path_save_cfg_info):
    """Create and save a list of geometric data from a binary file."""
    # Set seed for reproducibility
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

    # Create an Angr project for the binary executable
    project = angr.Project(thing=path_binary, load_options={"auto_load_libs": False})

    # Get the dictionary of functions to analyze {'name': address}
    functions_to_analyze = load_and_convert_pkl(path_dictionary)

    # Create a list of DataGeometric objects from an Angr project and a dictionary of functions
    geometric_datas_list = from_CFG_to_DataGeometric.get_Geometric_Datas(
        project=project, 
        functions_addresses=functions_to_analyze, 
        path_python_executable=path_python_executable,
        path_script_asm2vec=path_script_asm2vec,
        dictionary_labeled=False,
        path_save_cfg_info=path_save_cfg_info
    ) 

    try:
        # Save the list using torch.save
        torch.save(geometric_datas_list, path_output_file)
        print(f"\nGeometric datas list saved to {path_output_file}\n")
    except Exception as e:
        print(f"\nError saving geometric datas list: {e}\n")

def main(path_binary, path_dictionary, path_python_executable, path_script_asm2vec, path_output_file, path_save_cfg_info):
    """Main execution block."""
    # Call the function to create and save geometric data
    create_and_save_geometric_datas_list(
        path_binary=path_binary, 
        path_dictionary=path_dictionary, 
        path_python_executable=path_python_executable,
        path_script_asm2vec=path_script_asm2vec,
        path_output_file=path_output_file,
        path_save_cfg_info=path_save_cfg_info
    )

if __name__ == "__main__":
    if len(sys.argv) != 7:
        print("Usage: python create_geometric_datas.py <path_binary> <path_dictionary> <path_python_executable> <path_script_asm2vec> <path_output_file> <path_save_cfg>")
        sys.exit(1)
    
    path_binary = sys.argv[1]               # path binary executable
    path_dictionary = sys.argv[2]           # path dictionary of functions to analyze dict{name: addr}
    path_python_executable = sys.argv[3]    # path conda env for asm2vec model inference
    path_script_asm2vec = sys.argv[4]       # path script asm2vec model inference
    path_output_file = sys.argv[5]          # save file output path
    path_save_cfg_info = sys.argv[6]        # save cfg info file path 
    # main execution 
    main(path_binary, path_dictionary, path_python_executable, path_script_asm2vec, path_output_file, path_save_cfg_info)