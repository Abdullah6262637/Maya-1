import os
import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

def run_sft_1b():
    print("=== Starting Maya-1 1.5B Parameter Model Fine-Tuning (SFT) ===")
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "shared", "mega_sft_data.json")
    output_dir = os.path.join(base_dir, "shared", "checkpoints", "sft_1b")
    
    # 1. Load dataset
    print(f"Loading custom mega dataset from: {data_path}...")
    with open(data_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    print(f"Dataset contains {len(dataset)} items.")
    
    # Format for HuggingFace Dataset
    from datasets import Dataset
    hf_dataset = Dataset.from_list(dataset)
    
    # 2. Load pre-trained model & tokenizer
    model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    print("Loading Maya-1 1.5B Base Model...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load model in bfloat16 for high efficiency on A40
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    
    # 3. Configure LoRA (Low-Rank Adaptation)
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # 4. Training Arguments
    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=5,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=1,
        save_strategy="no", # Do not save intermediate steps to save disk space
        bf16=torch.cuda.is_available(), # Use bf16 if CUDA is available
        optim="adamw_torch",
        report_to="none",
        max_length=512
    )
    
    # 5. Initialize SFTTrainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=hf_dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
        args=training_args,
    )
    
    # 6. Train!
    print("Launching SFT training loop on A40 GPU...")
    trainer.train()
    
    # 7. Merge LoRA weights with base model and save
    print("Merging LoRA weights with base model...")
    merged_model = trainer.model.merge_and_unload()
    
    print(f"Saving final merged model to: {output_dir}...")
    merged_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print("[SUCCESS] Maya-1 1.5B Parameter SFT Fine-Tuning Completed!")

if __name__ == "__main__":
    run_sft_1b()
