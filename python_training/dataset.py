import os
import numpy as np
import torch
from torch.utils.data import Dataset

class ShardedMmapDataset(Dataset):
    """
    Zero-copy memory-mapped PyTorch dataset reading from raw binary token shards.
    Constructs autoregressive pairs: inputs (X) and targets (Y) offset by 1 token.
    """
    def __init__(self, bin_path: str, seq_len: int):
        super().__init__()
        self.bin_path = bin_path
        self.seq_len = seq_len
        
        if not os.path.exists(bin_path):
            raise FileNotFoundError(f"Binary token file not found at: {bin_path}")
            
        # Memory-map the file as a read-only numpy array of uint32
        self.tokens = np.memmap(bin_path, dtype=np.uint32, mode="r")
        
        # Calculate number of full sequences we can fit, leaving 1 extra token at the end for targets
        self.num_sequences = (len(self.tokens) - 1) // seq_len
        
    def __len__(self) -> int:
        return self.num_sequences

    def __getitem__(self, idx: int):
        start_idx = idx * self.seq_len
        end_idx = start_idx + self.seq_len
        
        # Slice the memory-mapped tokens (zero-copy) and convert to PyTorch tensors (int64)
        x = torch.from_numpy(self.tokens[start_idx:end_idx].astype(np.int64))
        y = torch.from_numpy(self.tokens[start_idx + 1:end_idx + 1].astype(np.int64))
        
        return x, y

def prepare_dataset_from_text(
    txt_path: str, 
    tokenizer_path: str, 
    bin_path: str, 
    chunk_size: int = 1024
) -> int:
    """
    Bridge utility calling the compiled Rust extension to tokenize a raw text file
    directly into a raw binary shard.
    """
    import rust_dataloader
    print(f"Tokenizing {txt_path} into binary format using Rust dataloader...")
    total_tokens = rust_dataloader.tokenize_file_to_bin(
        input_path=txt_path,
        tokenizer_path=tokenizer_path,
        output_path=bin_path,
        chunk_size=chunk_size
    )
    print(f"Preprocessing completed! Total tokens processed: {total_tokens}")
    return total_tokens
