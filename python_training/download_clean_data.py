import os
import sys
import argparse
import numpy as np
from tokenizers import Tokenizer
from datasets import load_dataset
from tqdm import tqdm

def main():
      parser = argparse.ArgumentParser(description="Download and tokenize mC4 Turkish dataset from Common Crawl")
      parser.add_argument("--tokenizer_path", type=str, default="../shared/mock_tokenizer.json", help="Path to tokenizer file")
      parser.add_argument("--output_path", type=str, default="../shared/clean_data.bin", help="Path to output binary file")
      parser.add_argument("--target_tokens", type=int, default=3000000000, help="Target number of tokens (default: 3 billion)")
      parser.add_argument("--chunk_size", type=int, default=1000000, help="Buffer size in tokens before writing to disk")
      args = parser.parse_args()

      print(f"Loading tokenizer from: {args.tokenizer_path}")
      if not os.path.exists(args.tokenizer_path):
          print(f"Error: Tokenizer file not found at {args.tokenizer_path}")
          sys.exit(1)
      tokenizer = Tokenizer.from_file(args.tokenizer_path)

      print(f"Opening output file: {args.output_path}")
      if os.path.dirname(args.output_path):
          os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
      
      mode = "ab" if os.path.exists(args.output_path) else "wb"
      existing_tokens = 0
      if os.path.exists(args.output_path):
          existing_tokens = os.path.getsize(args.output_path) // 4
          print(f"Output file already exists with {existing_tokens:,} tokens.")
          if existing_tokens >= args.target_tokens:
              print("Target token count already reached. Exiting.")
              return
      
      print("Loading mc4 Turkish dataset in streaming mode...")
      # mc4 is a pre-cleaned, high quality Common Crawl split by Google.
      # Streaming mode is critical to avoid high memory/disk allocation.
      try:
          dataset = load_dataset("mc4", "tr", split="train", streaming=True, trust_remote_code=True)
      except Exception as e:
          print(f"Error loading dataset: {e}")
          print("Make sure you have internet access and the 'datasets' python package installed (pip install datasets).")
          sys.exit(1)

      token_buffer = []
      total_tokens = existing_tokens
      
      pbar = tqdm(initial=total_tokens, total=args.target_tokens, desc="Downloading & Tokenizing")

      with open(args.output_path, mode) as f:
          for item in dataset:
              text = item.get("text", "")
              if not text:
                  continue
                  
              # Tokenize raw text
              enc = tokenizer.encode(text)
              token_buffer.extend(enc.ids)
              
              # Buffer flush
              if len(token_buffer) >= args.chunk_size:
                  arr = np.array(token_buffer, dtype=np.uint32)
                  f.write(arr.tobytes())
                  total_tokens += len(token_buffer)
                  pbar.update(len(token_buffer))
                  token_buffer = []
                  
              if total_tokens >= args.target_tokens:
                  break

          # Flush remaining tokens
          if token_buffer:
              remaining_needed = args.target_tokens - total_tokens
              if remaining_needed > 0:
                  write_buf = token_buffer[:remaining_needed]
                  arr = np.array(write_buf, dtype=np.uint32)
                  f.write(arr.tobytes())
                  total_tokens += len(write_buf)
                  pbar.update(len(write_buf))
                  
      pbar.close()
      print(f"Completed! Total tokens saved to bin: {total_tokens:,}")

if __name__ == "__main__":
      main()
