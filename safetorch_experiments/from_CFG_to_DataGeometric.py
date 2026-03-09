import torch
from torch_geometric.data import Data
import networkx as nx
import matplotlib.pyplot as plt
import angr
import os

# For LLM analysis
from dotenv import load_dotenv
from .groq_analyzer import GroqAnalyzer

# Constants
CALLDEPTH = 2                  # Analyzes the current function and its direct calls up to two levels deep.
CONTEXT_SENSITIVITY_LEVEL = 2  # Considers different calling contexts for functions for precise behavior analysis.
NORMALIZE = True               # Simplifies the CFG structure by removing unnecessary nodes and edges.
KEEP_STATE = True              # Preserves all input states during analysis for debugging and exploration.

# API Key for the LLM analyzer (Groq)
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


def save_CFG(cfg, title, name, hex_address, save_path=None):    
    # Create a NetworkX graph to represent the CFG
    cfg_graph = nx.DiGraph()
    
    # Use the actual node objects as keys to preserve all edges correctly
    # Mapping addresses for labels later
    node_labels = {}

    for node in cfg.graph.nodes():
        addr = hex(node.addr)
        cfg_graph.add_node(node)
        node_labels[node] = addr

    for src, dst, data in cfg.graph.edges(data=True):
        cfg_graph.add_edge(src, dst)

    # Use a layout that handles directed graphs well
    # 'shell_layout' or 'spring_layout' are usually best for small CFGs
    layout = nx.spring_layout(cfg_graph, k=0.5, iterations=50)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Draw nodes
    nx.draw_networkx_nodes(cfg_graph, pos=layout, node_size=1500, node_color='lightblue', ax=ax)
    
    # Draw labels (the hex addresses)
    nx.draw_networkx_labels(cfg_graph, pos=layout, labels=node_labels, font_size=8, ax=ax)
    
    # Draw edges with curvature to reveal overlapping connections
    # 'rad=0.1' creates a slight curve so parallel edges don't hide each other
    nx.draw_networkx_edges(
        cfg_graph, 
        pos=layout, 
        edgelist=list(cfg_graph.edges()),
        arrows=True, 
        arrowsize=20, 
        connectionstyle='arc3, rad=0.1', 
        edge_color='gray',
        ax=ax
    )

    ax.set_title(f"Control Flow Graph {title}: {name} @ {hex_address} (Nodes: {len(cfg_graph.nodes)}, Edges: {len(cfg_graph.edges)})")

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        file_name = f"CFG_{name}_{hex_address}.png"
        full_path = os.path.join(save_path, file_name)
        plt.savefig(full_path)
        print(f"CFG saved as {full_path}")
    else:
        plt.show()
    plt.close(fig)


def save_assembly_code(cfg, save_path, name, hex_address):
    visited_addresses = set()
    # Create a filename using the provided name and the first address
    file_name = f"AssemblyCode_{name}_{hex_address}.txt"
    
    with open(os.path.join(save_path, file_name), 'w') as file:
        for node in cfg.graph.nodes():
            # Address of the node
            addr = hex(node.addr)
            # Check if the address has been visited
            if addr not in visited_addresses:
                # Extract the assembly code from the node
                assembly_code = extract_assembly_code_from_node(node)
                # Add the address to the set of visited addresses
                visited_addresses.add(addr)
                # Write the address and assembly code to the file
                file.write(f"Address: {addr}\n")
                file.write(f"Assembly Code:\n{assembly_code}\n\n")
    
    print(f"Assembly code saved as {os.path.join(save_path, file_name)}")



# Small cache so the model is loaded only once for all nodes.
_safetorch_cache = {
    "net": None,
    "conv": None,
    "norm": None,
    "device": None,
    "dim": None
}

def safetorch_inference(assembly_code, path_python_executable=None, path_script_embedding_model=None):
    """
    Use the local SAFE model to get an embedding for the provided assembly code string.
    - assembly_code: string with one instruction per line (as produced by extract_assembly_code_from_node).
    - path_python_executable, path_script_embedding_model: kept for compatibility with your existing calls,
      but are NOT used (we load the local safetorch model).
    Returns:
      - embedding as a list of floats (length = SAFE embedding dim).
      - never returns None; for empty input returns a zero-vector of appropriate length.
    """
    import torch
    # lazy import/init of model + converters
    global _safetorch_cache

    # Default model/vocab paths used by your original pipeline
    model_path = "./safetorch/model/SAFEtorch.pt"
    vocab_path = "./safetorch/model/word2id.json"
    max_instruction = 150
    device = "cpu"

    # Initialize model & helpers once
    if _safetorch_cache["net"] is None:
        try:
            from safetorch.safetorch.safe_network import SAFE
            from safetorch.safetorch.parameters import Config
            from safetorch.utils.function_normalizer import FunctionNormalizer
            from safetorch.utils.instructions_converter import InstructionsConverter
            
        except Exception as e:
            # give a helpful error if imports fail
            raise RuntimeError(f"Failed to import SAFE or utilities: {e}")

        cfg = Config()
        net = SAFE(cfg).to(device)
        sd = torch.load(model_path, map_location=device)
        net.load_state_dict(sd)
        net.eval()
        conv = InstructionsConverter(vocab_path)
        norm = FunctionNormalizer(max_instruction=max_instruction)

        _safetorch_cache.update({
            "net": net,
            "conv": conv,
            "norm": norm,
            "device": device,
            "dim": None
        })

    net = _safetorch_cache["net"]
    conv = _safetorch_cache["conv"]
    norm = _safetorch_cache["norm"]

    # Convert assembly string into a list of instruction strings
    # e.g. "mov eax, ebx\nadd eax, 1\nret\n" -> ["mov eax, ebx", "add eax, 1", "ret"]
    instrs = [line.strip() for line in assembly_code.splitlines() if line.strip()]

    # If we have no instructions, return a zero-vector of the known embedding dimension.
    # If the dim is unknown (first call happened with empty instrs), we compute dim by embedding a single "nop".
    if not instrs:
        if _safetorch_cache["dim"] is None:
            # compute a small embedding to determine size
            dummy = ["nop"]
            ids = conv.convert_to_ids(dummy)
            padded, lengths = norm.normalize_functions([ids])
            seq = padded[0]
            length0 = int(lengths[0])
            tensor = torch.LongTensor(seq).to(device)
            with torch.no_grad():
                emb = net(tensor, [length0]).squeeze(0).cpu()
            dim = int(emb.shape[0])
            _safetorch_cache["dim"] = dim
        else:
            dim = _safetorch_cache["dim"]
        return [0.0] * dim

    # Convert instructions -> ids -> normalized padded sequence
    ids = conv.convert_to_ids(instrs)
    padded, lengths = norm.normalize_functions([ids])
    seq = padded[0]
    length0 = int(lengths[0])

    # Run SAFE encoder
    tensor = torch.LongTensor(seq).to(device)
    with torch.no_grad():
        emb = net(tensor, [length0]).squeeze(0).cpu()

    # cache embedding dimensionality
    if _safetorch_cache["dim"] is None:
        _safetorch_cache["dim"] = int(emb.shape[0])

    # return plain Python list of floats (safe to store as node attribute later)
    return emb.numpy().tolist()

def extract_assembly_code_from_node(angr_node):
    '''
    Extract the assembly code from an Angr CFG node and return a string of assembly code.
    Parameters:
        angr_node (angr.analyses.cfg.CFGNode): The CFG node.
    Returns:
        assembly_code (str): A string of assembly code.
    '''
    # If there is no block at all, bail out early
    if angr_node.block is None:
        return ""
    try:
        assembly_code = ""
        for instr in angr_node.block.capstone.insns:
            mnemonic = instr.mnemonic
            op_str = instr.op_str
            instruct = mnemonic + " " + op_str
            assembly_code += instruct + "\n"
    except KeyError:
        # No bytes mapped for this block’s address — skip it
        return ""
    return assembly_code  # bb1, bb2, bb3, bb4, ...


def get_node_embedding(node, path_python_executable, path_script_embedding_model):
    # Extract the assembly code from the angr node
    assembly_code = extract_assembly_code_from_node(node) # assembly_code = [bb1, bb2, ...]
    # Get the embedding for the assembly code using the embedding model
    embedding = safetorch_inference(assembly_code, path_python_executable, path_script_embedding_model)
    return embedding


def get_ACFG(cfg, project, path_python_executable, path_script_embedding_model, assembly_save_path, ref):
    """Function to extract an ACFG from a CFG of a function
    Args:
        :param cfg:                         CFG of a function
        :param project:                     Angr project
        :param path_python_executable:      Path to the Python executable for Asm2Vec
        :param path_script_embedding_model: Path to the embedding script for model inference
    Output:
        :return:            ACFG of the function
    """
    
    # Create an empty ACFG
    acfg = nx.DiGraph()

    # Add non-duplicated nodes
    visited_addresses = set()
    for node in cfg.graph.nodes():
        # Address of the node
        addr = hex(node.addr)
        # Check if the address has been visited
        if addr not in visited_addresses:
            # Get the node embedding
            features = get_node_embedding(node, path_python_executable, path_script_embedding_model)
            # Add the address to the set of visited addresses
            visited_addresses.add(addr)
            # Add the new node in the ACFG with additional features
            acfg.add_node(
                # node address
                node_for_adding=addr,
                # node features
                embedding=features,
            )

    # Add edges between existing nodes
    for src, dst, data in cfg.graph.edges(data=True):
        # Retrieve the nodes addresses 
        src_addr = hex(src.addr)
        dst_addr = hex(dst.addr)

        # Ensure both source address and destination address exist in the new graph
        if (src_addr in visited_addresses) and (dst_addr in visited_addresses):
            # Add the edge in the ACFG with additional features
            acfg.add_edge(
                # edge source and destination 
                u_of_edge=src_addr, 
                v_of_edge=dst_addr, 
            )
     
    print(f"Final acfg (in get_ACFG): {acfg}")
    # Save assembly code in a txt file 
    save_assembly_code(cfg, assembly_save_path, name=ref['name'], hex_address=ref['address'])

    return acfg



def get_Geometric_Data_from_CFG(cfg, project, label=None, path_python_executable=None, path_script_embedding_model=None, ref=None, path_save_cfg_info=None, llm_data=None):
    """
    Function to extract a PyTorch Geometric Data object from a CFG of a function.
    
    Args:
        :param llm_data: A dictionary containing {'llm_prediction': int, 'llm_reasoning': str}
    """

    # Get the ACFG from the CFG
    acfg = get_ACFG(
        cfg=cfg, 
        project=project, 
        path_python_executable=path_python_executable, 
        path_script_embedding_model=path_script_embedding_model, 
        assembly_save_path=path_save_cfg_info, 
        ref=ref
    )
    
    # Save the PNG visualization
    save_CFG(cfg, "(CFG)", ref['name'], ref['address'], save_path=path_save_cfg_info)
    
    # Extract node features (SAFEtorch embeddings)
    node_features = []  
    for node_id, attributes in acfg.nodes(data=True):
        values = [value for key, value in attributes.items()]
        node_features.append(values) 

    # Edge indices mapping
    address_to_index = {address: idx for idx, address in enumerate(acfg.nodes)}
    edge_indices = [] 
    for src, dst, edge in acfg.edges(data=True):
        edge_indices.append([address_to_index[src], address_to_index[dst]])

    # Convert to PyTorch tensors
    print(f"\nConvert to PyTorch tensors:")
    flat_node_features = [emb for sublist in node_features for emb in sublist]
    x = torch.tensor(flat_node_features, dtype=torch.float32)

    if len(edge_indices) > 0:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    print(f"x.shape: {x.shape} | edge_index.shape: {edge_index.shape}")

    # --- UPDATED: LLM Data Integration ---
    # We no longer have 6 scores. We have a binary verdict and a reasoning string.
    if llm_data is not None:
        # This will add 'llm_prediction' and 'llm_reasoning' to the ref dictionary
        ref.update(llm_data)
    else:
        # Default values if LLM analysis was skipped
        ref['llm_prediction'] = 0
        ref['llm_reasoning'] = "LLM analysis was not performed."

    # Create the PyTorch Geometric Data object
    if label is not None:
        y = torch.tensor([label], dtype=torch.long)
        data = Data(x=x, edge_index=edge_index, y=y, ref=ref)
    else:
        data = Data(x=x, edge_index=edge_index, ref=ref)

    data.validate(raise_on_error=True)
    print(f"Data object: {data}")
    return data

def get_Geometric_Datas(project, functions_addresses, path_python_executable=None, path_script_embedding_model=None, dictionary_labeled=True, path_save_cfg_info=None):
    """Function to extract a list of PyTorch Geometric Data objects from a list of functions
    Args:
        :param project:                 Angr project
        :param functions_addresses:     A dictionary of the form {name:(addres, label)} of the functions
    Output:
        :return:                        List of PyTorch Geometric Data objects with the CFG information and extracted features
    """    

    # Initialize the LLM analyzer 
    analyzer = GroqAnalyzer(api_key=GROQ_API_KEY)
    requirements_path = os.path.join(os.path.dirname(__file__), "..", "Requirement", "requirements.txt")
    analyzer.load_requirements(requirements_path) # read requirements file


    tot_count0 = 0
    tot_count1 = 0
    # list of ACFGs extracted from the binary
    geometric_datas_list = []
    i = 0


    names = ["sym.jsmn_parse", "sym.test_count", "sym.jsmn_parse_string", "sym.jsmn_parse_primitive", "sym.vtokeq"]


    # Loop through each function in the project 
    for name, value in functions_addresses.items(): # functions_addresses = {name:(address, label)}
        # Check for the label 
        if dictionary_labeled:
            address, label = value
        else:
            address = value

        print(f"\n-------------------------ITERATION: {i} --- Name: {name} --- Address: {hex(address)} ------------------------------------------") 
        i+=1
        # Find the starting state for the CFG
        start_state = project.factory.blank_state(addr=address, state_add_options=angr.options.ZERO_FILL_UNCONSTRAINED_REGISTERS) # project.factory.blank_state(addr=address)

        # Get the CFG for the specified function
        cfg = project.analyses.CFGEmulated(
            starts=[address], 
            initial_state=start_state, 
            context_sensitivity_level=CONTEXT_SENSITIVITY_LEVEL, 
            normalize=NORMALIZE, 
            call_depth=CALLDEPTH, 
            state_add_options=angr.options.refs, 
            keep_state=KEEP_STATE
        )  

        ########## LLM Inference 
        # Extract FULL assembly for the function
        full_assembly = ""
        print(f"    DEBUG: CFG for {name} has {len(cfg.graph.nodes())} nodes.")
        for node in cfg.graph.nodes():
            full_assembly += extract_assembly_code_from_node(node)

        # Limit to roughly 5000 characters to stay safe with API limits
        if len(full_assembly) > 5000:
            full_assembly = full_assembly[:5000] + "\n... [TRUNCATED]"

        # Get LLM Scores
        print(f"Calling Groq for {name}...")
        if not full_assembly.strip():
            print(f"    [WARNING] Assembly is EMPTY! Skipping Groq.")
            # Use the new keys: llm_prediction and llm_reasoning
            llm_result = {"llm_prediction": 0, "llm_reasoning": "Empty assembly."}
        else:
            print(f"    [INFO] Sending {len(full_assembly)} chars to Groq.")

            llm_result = {"llm_prediction": 0, "llm_reasoning": "Limited API calls for testing."}
            if name in names:  # Limit the number of API calls for testing

                # Use the new analyzer which returns a dictionary with 'llm_prediction'
                llm_result = analyzer.analyze_assembly(full_assembly)

                
        # Check for the label
        if dictionary_labeled:
            data_geom = get_Geometric_Data_from_CFG(cfg=cfg, project=project, label=label, 
                                                    path_python_executable=path_python_executable, 
                                                    path_script_embedding_model=path_script_embedding_model,
                                                    path_save_cfg_info=path_save_cfg_info,
                                                    ref={'name': name, 'address': hex(address)},
                                                    # LLM inference
                                                    llm_data=llm_result
                                                    )
            
            # Count the number of geometric datas with label 0 and 1
            if label == 0:
                tot_count0 += 1
            elif label == 1:
                tot_count1 += 1

        else:
            data_geom = get_Geometric_Data_from_CFG(cfg=cfg, project=project, label=None,
                                                    path_python_executable=path_python_executable, 
                                                    path_script_embedding_model=path_script_embedding_model,
                                                    path_save_cfg_info=path_save_cfg_info,
                                                    ref={'name': name, 'address': hex(address)},
                                                    # LLM inference
                                                    llm_data=llm_result
                                                    )
    
        # Append the Geometric Data to the list
        geometric_datas_list.append(data_geom)



        
    print(f"\n\n--------------------------------------------------------------------------------------------------")
    print(f"Total number of geometric datas (ACFG): {len(geometric_datas_list)}")
    print(f"with label 0: {tot_count0}")
    print(f"with label 1: {tot_count1}")
    return geometric_datas_list