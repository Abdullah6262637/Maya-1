#!/usr/bin/env bash
# Bash Integration Test Suite for Multilingual AI System
# Runs all checks sequentially and prints validation status.

set -e

GREEN='\033[0;32%'
YELLOW='\033[1;33%'
RED='\033[0;31%'
NC='\033[0%' # No Color

echo -e "${YELLOW}==========================================================${NC}"
echo -e "${YELLOW}      STARTING UNIX INTEGRATION VERIFICATION SUITE${NC}"
echo -e "${YELLOW}==========================================================${NC}"

# 1. Check Rust Dataloader Compilation and Build Shared Library
echo -e "\n${YELLOW}[STEP 1] Building Rust dataloader shared library...${NC}"
pushd rust_dataloader > /dev/null
export PYO3_PYTHON="python"
export PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1
export CARGO_TARGET_DIR=".target"
cargo build --release
cp .target/release/librust_dataloader.so ../rust_dataloader.so
popd > /dev/null
echo -e "${GREEN}[SUCCESS] Rust module compiled and copied successfully!${NC}"

# 2. Run Data Preprocessing (Rust Tokenizer)
echo -e "\n${YELLOW}[STEP 2] Launching Python + Rust preprocessing script...${NC}"
python scripts/prepare_mock_data.py
echo -e "${GREEN}[SUCCESS] Data preprocessing successfully verified!${NC}"

# 3. Check LLaMA PyTorch Training & DB Logging
echo -e "\n${YELLOW}[STEP 3] Running 10-step PyTorch training loop on CPU (BFloat16)...${NC}"
python python_training/train.py \
    --data_path shared/mock_data.bin \
    --vocab_size 5000 \
    --hidden_size 768 \
    --num_hidden_layers 12 \
    --num_attention_heads 12 \
    --num_key_value_heads 4 \
    --intermediate_size 2048 \
    --seq_len 128 \
    --batch_size 4 \
    --max_steps 10 \
    --checkpoint_interval 5
echo -e "${GREEN}[SUCCESS] PyTorch training loop executed successfully!${NC}"

# 4. Check SQLite Database Records
echo -e "\n${YELLOW}[STEP 4] Querying SQLite metrics database...${NC}"
dbCheck=$(python -c "import sqlite3; conn = sqlite3.connect('shared/metrics.db'); cursor = conn.cursor(); cursor.execute('SELECT name, count(*) FROM metrics GROUP BY name'); print(cursor.fetchall())")
echo -e "Database metrics count: ${dbCheck}"
echo -e "${GREEN}[SUCCESS] SQLite database metrics logging verified!${NC}"

echo -e "\n${GREEN}==========================================================${NC}"
echo -e "${GREEN}  SUCCESS! ALL MULTILINGUAL PIPELINE TESTS PASSED!${NC}"
echo -e "${GREEN}==========================================================${NC}"
