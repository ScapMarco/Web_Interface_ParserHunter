import sys
import angr
import logging
import os

# Disable angr noise so we only see OUR debug prints
logging.getLogger('angr').setLevel(logging.ERROR)

def terminal_msg(msg):
    """
    Uses low-level os.write to stderr (file descriptor 2).
    This bypasses Python's internal buffering to ensure you see the 
    output immediately in your Flask terminal.
    """
    os.write(2, f"{msg}\n".encode())

class AdvancedHarness:
    def __init__(self, binary_path, func_addr_raw, arg_map):
        terminal_msg(f"[DEBUG] Harness Initializing...")
        terminal_msg(f"[DEBUG] Binary: {binary_path} | Target: {func_addr_raw}")

        self.binary_path = binary_path
        self.arg_map = arg_map
        
        # 1. Address Conversion
        try:
            if "0x" in func_addr_raw.lower():
                self.func_addr = int(func_addr_raw, 16)
            else:
                self.func_addr = int(func_addr_raw)
            terminal_msg(f"[DEBUG] Address converted to: {hex(self.func_addr)}")
        except Exception as e:
            terminal_msg(f"!! [ERROR] Address conversion failed: {e}")
            sys.exit(1)

        # 2. Project Setup
        try:
            # Force base_addr to 0x0 to match Radare2 output from app.py
            self.proj = angr.Project(binary_path, auto_load_libs=False, main_opts={'base_addr': 0x0})
            
            # HOOKS: Critical for speed. Bypasses slow library loading.
            # random_delay is hooked to ReturnUnconstrained (does nothing and returns)
            self.proj.hook_symbol('printf', angr.SIM_PROCEDURES['stubs']['ReturnUnconstrained']())
            self.proj.hook_symbol('random_delay', angr.SIM_PROCEDURES['stubs']['ReturnUnconstrained']())
            self.proj.hook_symbol('strlen', angr.SIM_PROCEDURES['libc']['strlen']())
            self.proj.hook_symbol('strcpy', angr.SIM_PROCEDURES['libc']['strcpy']())
            self.proj.hook_symbol('sprintf', angr.SIM_PROCEDURES['libc']['sprintf']())
            self.proj.hook_symbol('memset', angr.SIM_PROCEDURES['libc']['memset']())
            
            terminal_msg(">> [DEBUG] Project Loaded and Library Symbols Hooked.")
        except Exception as e:
            terminal_msg(f"!! [ERROR] Project Load failed: {e}")
            sys.exit(1)

    def crash_inspection(self, state):
        """Check if the fuzzer controls the instruction pointer (RIP)."""
        if state.regs.rip.symbolic:
            terminal_msg("!! [CRASH] SUCCESS! Symbolic RIP found (Control Flow Hijack).")
            return True
        return False

    def run(self):
        # 1. Read input from AFL++ (standard stdin)
        fuzz_input = sys.stdin.buffer.read()
        if not fuzz_input:
            fuzz_input = b"seed"
        fuzz_input += b"\x00" # Null terminator safety for C-string functions

        # 2. Setup State
        # UNICORN is used for high-speed concrete emulation
        state = self.proj.factory.blank_state(
            addr=self.func_addr,
            add_options={
                angr.options.UNICORN, 
                angr.options.ZERO_FILL_UNCONSTRAINED_REGISTERS
            }
        )

        # 3. Memory Mapping
        # Map the fuzzer data to a specific, high memory address
        fuzz_base_addr = 0x1000000 
        state.memory.store(fuzz_base_addr, fuzz_input)

        # 4. Map Registers based on arg_map (e.g., "ptr,len,null...")
        arg_regs = ['rdi', 'rsi', 'rdx', 'rcx', 'r8', 'r9']
        mapping = self.arg_map.split(',')

        for i, arg_type in enumerate(mapping):
            if i >= len(arg_regs): break
            reg_name = arg_regs[i]
            
            if arg_type == "ptr":
                setattr(state.regs, reg_name, fuzz_base_addr)
            elif arg_type == "len":
                setattr(state.regs, reg_name, len(fuzz_input) - 1)
            elif arg_type == "val":
                # First 8 bytes interpreted as a long
                val = int.from_bytes(fuzz_input[:8].ljust(8, b'\x00'), 'little')
                setattr(state.regs, reg_name, val)
            else:
                setattr(state.regs, reg_name, 0)

        # 5. Run simulation
        simgr = self.proj.factory.simgr(state)
        try:
            # Step limit prevents getting stuck in loops within the target
            simgr.run(n=1000)

            # Signal a crash to AFL++ if angr detects a fault
            if simgr.errored:
                sys.exit(1)

            # Check every "deadended" path to see if the fuzzer gained control
            for s in simgr.deadended:
                if self.crash_inspection(s):
                    sys.exit(1)

        except Exception as e:
            # Silently exit on standard execution errors so AFL continues
            sys.exit(1)

if __name__ == "__main__":
    # AFL++ calls: python fuzz_harness.py <binary> <addr> <map>
    if len(sys.argv) < 4:
        terminal_msg("!! [ERROR] Missing Arguments. Usage: harness.py <bin> <addr> <map>")
        sys.exit(1)
        
    harness = AdvancedHarness(sys.argv[1], sys.argv[2], sys.argv[3])
    harness.run()