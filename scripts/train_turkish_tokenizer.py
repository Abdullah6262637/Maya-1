import os
import sys
import argparse
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder

def train_tokenizer(corpus_path, output_path, vocab_size=50000):
    if not os.path.exists(corpus_path):
        print(f"Error: Corpus file not found at {corpus_path}")
        sys.exit(1)
        
    print(f"Initializing ByteLevelBPE Model...")
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    
    # Use byte-level pre-tokenizer (same as GPT-2/Llama)
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    
    # Trainer configuration
    special_tokens = ["<s>", "<pad>", "</s>", "<unk>"]
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        initial_alphabet=ByteLevel.alphabet()
    )
    
    # Train from files
    print(f"Training tokenizer on {corpus_path} with vocab_size={vocab_size}...")
    tokenizer.train([corpus_path], trainer)
    
    # Enable decoding with ByteLevel
    tokenizer.decoder = ByteLevelDecoder()
    
    # Save the trained tokenizer
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    tokenizer.save(output_path)
    print(f"Tokenizer trained successfully and saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ByteLevelBPE Tokenizer on Turkish Corpus")
    parser.add_argument("--corpus", type=str, required=True, help="Path to plain text corpus file")
    parser.add_argument("--output", type=str, default="shared/turkish_bpe_50k.json", help="Output JSON path")
    parser.add_argument("--vocab_size", type=int, default=50000, help="Vocabulary size")
    args = parser.parse_args()
    
    train_tokenizer(args.corpus, args.output, args.vocab_size)
