import os
import sys

# Ensure python_training directory is in path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "python_training"))

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
import rust_dataloader

def main():
    shared_dir = os.path.join(os.path.dirname(__file__), "..", "shared")
    os.makedirs(shared_dir, exist_ok=True)
    
    tokenizer_path = os.path.join(shared_dir, "mock_tokenizer.json")
    text_path = os.path.join(shared_dir, "mock_input.txt")
    bin_path = os.path.join(shared_dir, "mock_data.bin")
    
    ham_path = os.path.join(shared_dir, "ham_veri.txt")
    if os.path.exists(ham_path):
        print(f"Directly using custom corpus from: {ham_path}...")
        text_path = ham_path
        
        # Clean up old mock_input.txt if it exists to free disk space
        old_mock_input = os.path.join(shared_dir, "mock_input.txt")
        if os.path.exists(old_mock_input):
            try:
                os.remove(old_mock_input)
            except Exception:
                pass
    else:
        print("ham_veri.txt not found, falling back to default sentences.")
        lines = [
            "In the beginning God created the heaven and the earth.",
            "And the earth was without form, and void; and darkness was upon the face of the deep.",
            "LLaMA is a great model for natural language processing.",
            "Rust is a high performance systems programming language with memory safety."
        ]
        text_path = os.path.join(shared_dir, "mock_input.txt")
        repeated_text = "\n".join(lines * 120)
        with open(text_path, "w", encoding="utf-8") as f:
            f.write(repeated_text)
        
    # 2. Train a ByteLevel BPE tokenizer
    print("Training ByteLevel BPE tokenizer...")
    from tokenizers import ByteLevelBPETokenizer
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(
        files=[text_path],
        vocab_size=5000,
        special_tokens=["<s>", "<pad>", "</s>", "<unk>"]
    )
    tokenizer.save(tokenizer_path)
    print(f"Saved tokenizer to: {tokenizer_path}")
    
    # 3. Call the compiled Rust data loader to tokenize the text file directly into binary
    print("Executing Rust binary tokenization pipeline...")
    total_tokens = rust_dataloader.tokenize_file_to_bin(
        input_path=text_path,
        tokenizer_path=tokenizer_path,
        output_path=bin_path,
        chunk_size=64
    )
    print(f"Rust pipeline completed! Written {total_tokens} tokens to binary shard: {bin_path}")
    
    # Verify file is readable and has expected size (4 bytes per token for u32)
    expected_size = total_tokens * 4
    actual_size = os.path.getsize(bin_path)
    print(f"Verification: Expected size {expected_size} bytes, Actual size {actual_size} bytes.")
    if expected_size == actual_size:
        print("Success! Rust tokenizer integration verified.")
    else:
        print("Error: Size mismatch in binary output file!")
        sys.exit(1)

if __name__ == "__main__":
    main()
