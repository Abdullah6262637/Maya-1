import os
from tokenizers import Tokenizer

def add_chat_tokens(tokenizer_path):
    if not os.path.exists(tokenizer_path):
        print(f"Error: Tokenizer file not found at {tokenizer_path}")
        return
        
    print(f"Loading tokenizer from {tokenizer_path}...")
    tokenizer = Tokenizer.from_file(tokenizer_path)
    
    vocab_size_before = tokenizer.get_vocab_size()
    print(f"Vocab size before: {vocab_size_before}")
    
    # Define chat special tokens
    chat_tokens = ["<|sistem|>", "<|kullanici|>", "<|asistan|>", "<|son|>"]
    
    # Add tokens
    num_added = tokenizer.add_special_tokens(chat_tokens)
    vocab_size_after = tokenizer.get_vocab_size()
    print(f"Added {num_added} special tokens. New vocab size: {vocab_size_after}")
    
    # Print the IDs of the added tokens
    for token in chat_tokens:
        token_id = tokenizer.token_to_id(token)
        print(f"Token: {token} -> ID: {token_id}")
        
    # Save the updated tokenizer
    tokenizer.save(tokenizer_path)
    print(f"Updated tokenizer saved back to: {tokenizer_path}")

if __name__ == "__main__":
    tokenizer_path = "shared/turkish_bpe_50k.json"
    add_chat_tokens(tokenizer_path)
