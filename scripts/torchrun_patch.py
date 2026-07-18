# Custom wrapper to execute torchrun with an early C++ binding level monkey patch.
# Bypasses Windows-specific libuv socket socket compatibility errors in PyTorch's TCPStore.

import sys
import os
import torch
import torch._C._distributed_c10d as C_dist

# 1. Capture the native C++ TCPStore class and patch it immediately
orig_tcp_store = C_dist.TCPStore

def patched_tcp_store(*args, **kwargs):
    # TCPStore positional signature: (host, port, world_size, is_master, timeout, use_libuv, ...)
    args_list = list(args)
    if len(args_list) >= 6:
        args_list[5] = False
    else:
        kwargs["use_libuv"] = False
    return orig_tcp_store(*args_list, **kwargs)

C_dist.TCPStore = patched_tcp_store

# 2. Patch the Python distributed package namespaces
import torch.distributed as dist
dist.TCPStore = patched_tcp_store

# Dynamic monkey-patch across all imported modules in sys.modules
for ns in list(sys.modules.keys()):
    if ns.startswith("torch.distributed") or "rendezvous" in ns:
        try:
            mod = sys.modules[ns]
            if hasattr(mod, "TCPStore"):
                setattr(mod, "TCPStore", patched_tcp_store)
        except Exception:
            pass

# 3. Import and run PyTorch's standard distributed run entrypoint
from torch.distributed.run import main

if __name__ == "__main__":
    # Remove this wrapper file from sys.argv and invoke standard main
    sys.argv[0] = "torchrun"
    main()
