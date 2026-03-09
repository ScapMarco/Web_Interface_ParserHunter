import angr
import torch
import pickle
import random
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Utility functions
from safetorch_experiments import from_CFG_to_DataGeometric

# use conda env test-3.10.0-env

def load_and_convert_pkl(file_path):
    """Load a pickle file and convert it to a dictionary."""
    try:
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
            print(f"    Pickle file '{file_path}' loaded successfully.")
            print(f"    Length of loaded data: {len(data)}")

            if not isinstance(data, dict):
                raise ValueError("The loaded data is not a dictionary.")
            else:
                print(f"Loaded dictionary of {len(data)} functions!")    
                return data

    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    except Exception as e:
        print(f"Error: {e}")


def create_and_save_geometric_datas_list(path_binary, path_dictionary, path_output_file, path_save_cfg_info):
    """Create and save a list of geometric data from a binary file using SAFEtorch embeddings."""
    # Set seed for reproducibility
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

    # Create an Angr project for the binary executable
    project = angr.Project(
        thing=path_binary, 
        load_options={
            "auto_load_libs": False,
            "main_opts": {'base_addr': 0x0}
            }
        )

    # Get the dictionary of functions to analyze {'name': address}
    functions_to_analyze = load_and_convert_pkl(path_dictionary)

    # Create a list of DataGeometric objects (SAFEtorch handles embeddings internally)
    geometric_datas_list = from_CFG_to_DataGeometric.get_Geometric_Datas(
        project=project,
        functions_addresses=functions_to_analyze,
        dictionary_labeled=False,
        path_save_cfg_info=path_save_cfg_info
    )

    try:
        torch.save(geometric_datas_list, path_output_file)
        print(f"\nGeometric datas list saved to {path_output_file}\n")
    except Exception as e:
        print(f"\nError saving geometric datas list: {e}\n")


def main(path_binary, path_dictionary, path_output_file, path_save_cfg_info):
    """Main execution block."""
    create_and_save_geometric_datas_list(
        path_binary=path_binary,
        path_dictionary=path_dictionary,
        path_output_file=path_output_file,
        path_save_cfg_info=path_save_cfg_info
    )


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python create_geometric_datas.py <path_binary> <path_dictionary> <path_output_file> <path_save_cfg>")
        sys.exit(1)

    path_binary = sys.argv[1]          # binary executable
    path_dictionary = sys.argv[2]      # dict{name: addr} of functions
    path_output_file = sys.argv[3]     # save file output path
    path_save_cfg_info = sys.argv[4]   # save cfg info file path

    main(path_binary, path_dictionary, path_output_file, path_save_cfg_info)
