import pickle
import angr
import r2pipe
import sys
from multiprocessing import Process, Queue

# Constants
CALLDEPTH = 2                  # Analyzes the current function and its direct calls up to two levels deep.
CONTEXT_SENSITIVITY_LEVEL = 2  # Considers different calling contexts for functions for precise behavior analysis.
NORMALIZE = True               # Simplifies the CFG structure by removing unnecessary nodes and edges.
KEEP_STATE = True              # Preserves all input states during analysis for debugging and exploration.
MIN_NODES_EDGES = 3            # Minimum number of nodes and edges to consider 
TIMEOUT_DURATION = 60          # Timeout duration in seconds


def _cfg_worker(address, project_kwargs, out_q):
    try:
        proj = angr.Project(**project_kwargs)
        cfg = proj.analyses.CFGEmulated(
            starts=[address],
            initial_state=proj.factory.blank_state(addr=address),
            context_sensitivity_level=CONTEXT_SENSITIVITY_LEVEL,
            normalize=NORMALIZE,
            call_depth=CALLDEPTH,
            state_add_options=angr.options.refs,
            keep_state=KEEP_STATE,
            max_steps=100000
        )
        nodes = len(cfg.graph.nodes())
        edges = len(cfg.graph.edges())
        out_q.put((True, (nodes, edges)))
    except Exception as e:
        out_q.put((False, str(e)))

def filter_graphs(address, project_path, load_options):
    out_q = Queue()
    # Pass only what the worker needs: path and load_options
    p = Process(
        target=_cfg_worker,
        args=(address, {"thing": project_path, "load_options": load_options}, out_q)
    )
    p.start()
    p.join(TIMEOUT_DURATION)
    if p.is_alive():
        p.terminate()
        p.join()
        print(f"    CFG build process timed out at {hex(address)}")
        return False

    success, payload = out_q.get()
    if not success:
        print(f"    CFG error at {hex(address)}: {payload}")
        return False

    nodes, edges = payload
    if nodes <= MIN_NODES_EDGES or edges <= MIN_NODES_EDGES:
        return False

    print(f"    CFG OK for {hex(address)} (nodes={nodes}, edges={edges})")
    return True

def identify_functions_with_r2(project_path, load_options):
    """Identify functions in the binary using fast Radare2 and skip slow CFG builds."""
    print("Analyzing binary with RADARE2…")
    r2 = r2pipe.open(project_path)
    r2.cmd('e bin.libs=false')
    r2.cmd('e bin.cache=true')              # or r2.cmd('e bin.relocs.apply=true')
    r2.cmd('aa')                            # full auto-analysis (xrefs, vars, calls, etc.)
    r2.cmd('s .text')
    functions = r2.cmdj('aflj')

    results = {}
    for i, func in enumerate(functions):
        name = func['name']
        addr = func['offset']
        # skip libs/imports
        if func.get('is_lib') or name.startswith('sym.imp.'):
            continue

        # call filter_graphs
        if filter_graphs(
            address=addr,
            project_path=project_path,
            load_options=load_options
        ):
            print(f"{len(results)}: {name} @ {hex(addr)}")
            results[name] = addr

    return results

def save_labeled_data(labeled_dict, filename):
    with open(filename, 'wb') as file:
        pickle.dump(labeled_dict, file)
    print(f"    Labeled data saved to {filename}")

def main(binary_path, file_path):
    """Main execution block."""
    # Get the list of functions
    load_opts = {"auto_load_libs": False}
    functions_to_analyze = identify_functions_with_r2(
        project_path=binary_path,
        load_options=load_opts
    )
    # save the functions_to_analyze dictionary data
    save_labeled_data(functions_to_analyze, filename=file_path) 

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python extract_list_functions.py <binary_path> <file_path>")
        sys.exit(1)
    
    binary_path = sys.argv[1]   # path binary executable
    file_path = sys.argv[2]     # save file output path
    # main execution
    main(binary_path, file_path)