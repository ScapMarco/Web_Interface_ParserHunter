import json
import torch
import sys
from GNNs_Models_Classifiers import MyGNNClassifier


def load_model(model_path, params_path):
    # load params
    with open(params_path, "r") as f:
        params = json.load(f)

    # initialize wrapper
    model = MyGNNClassifier(model_name="GraphSAGE", verbose=True)
    model.set_params(**params)

    # initialize underlying GNN + optimizer/loss
    model._init_model_and_optim()

    # load trained weights
    checkpoint = torch.load(model_path, map_location="cpu")
    state_dict = checkpoint if isinstance(checkpoint, dict) else checkpoint.state_dict()

    # remove "model." prefix if necessary
    new_state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
    model.model.load_state_dict(new_state_dict)

    return model


def load_input_data(input_data_path):
    """Load input data from a .pt file."""
    return torch.load(input_data_path)

def save_predictions(output_file_path, predictions):
    """Save model predictions to a specified output file."""
    with open(output_file_path, 'w') as f:
        for prediction in predictions:
            f.write(f"{prediction}\n")

def main(model_path, params_path, input_data_path, output_file_path):
    """Main execution block."""
    # Load the model with parameters
    model = load_model(model_path, params_path)  
    # Load input data
    input_data = load_input_data(input_data_path)  
    # Score the the input data and save the results into output path
    df_case_studies = model.score(input_data, case_studies=True, filename=output_file_path)
    print(f"Case studies saved, number of graphs: {len(df_case_studies)}")


if __name__ == "__main__":
    # Get model and parameters paths from command line arguments
    if len(sys.argv) != 5:
        print("Usage: python model_inference.py <MODEL_PATH> <PARAMS_PATH> <INPUT_DATA_PATH> <OUTPUT_FILE_PATH>")
        sys.exit(1)

    model_path = sys.argv[1]        # model path
    params_path = sys.argv[2]       # parameters path
    input_data_path = sys.argv[3]   # input pytorch geometric datas path
    output_file_path = sys.argv[4]  # save file output path
    # main execution 
    main(model_path, params_path, input_data_path, output_file_path)