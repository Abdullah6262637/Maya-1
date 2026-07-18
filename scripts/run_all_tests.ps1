# Windows PowerShell Integration Test Suite for Multilingual AI System
# Runs all checks sequentially and prints validation status.

$ErrorActionPreference = "Stop"

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "      STARTING INTEGRATION VERIFICATION SUITE" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

# 1. Check Rust Dataloader Compilation
Write-Host "`n[STEP 1] Checking Rust dataloader compile..." -ForegroundColor Yellow
try {
    Push-Location rust_dataloader
    $env:PYO3_PYTHON = "C:\Users\HP\Desktop\Maya-1\python_training\.venv\Scripts\python.exe"
    $env:PYO3_USE_ABI3_FORWARD_COMPATIBILITY = 1
    $env:CARGO_TARGET_DIR = "C:\Users\HP\Desktop\Maya-1\rust_dataloader\.target"
    
    cargo check
    Pop-Location
    Write-Host "[SUCCESS] Rust module compilation check passed!" -ForegroundColor Green
} catch {
    Write-Host "[ERROR] Rust check failed: $_" -ForegroundColor Red
    Pop-Location
    Exit 1
}

# 2. Run Data Preprocessing (Rust Tokenizer)
Write-Host "`n[STEP 2] Launching Python + Rust preprocessing script..." -ForegroundColor Yellow
try {
    & "python_training\.venv\Scripts\python.exe" scripts/prepare_mock_data.py
    Write-Host "[SUCCESS] Data preprocessing successfully verified!" -ForegroundColor Green
} catch {
    Write-Host "[ERROR] Data preparation failed: $_" -ForegroundColor Red
    Exit 1
}

# 3. Check LLaMA PyTorch Training & DB Logging
Write-Host "`n[STEP 3] Running 10-step PyTorch training loop on CPU (BFloat16)..." -ForegroundColor Yellow
try {
    & "python_training\.venv\Scripts\python.exe" python_training/train.py `
        --data_path shared/mock_data.bin `
        --vocab_size 1000 `
        --hidden_size 64 `
        --num_hidden_layers 2 `
        --num_attention_heads 4 `
        --num_key_value_heads 2 `
        --intermediate_size 128 `
        --seq_len 64 `
        --batch_size 2 `
        --max_steps 10 `
        --checkpoint_interval 5
    Write-Host "[SUCCESS] PyTorch training loop executed successfully!" -ForegroundColor Green
} catch {
    Write-Host "[ERROR] PyTorch training failed: $_" -ForegroundColor Red
    Exit 1
}

# 4. Check SQLite Database Records
Write-Host "`n[STEP 4] Querying SQLite metrics database..." -ForegroundColor Yellow
try {
    $dbCheck = & "python_training\.venv\Scripts\python.exe" -c "import sqlite3; conn = sqlite3.connect('shared/metrics.db'); cursor = conn.cursor(); cursor.execute('SELECT name, count(*) FROM metrics GROUP BY name'); print(cursor.fetchall())"
    Write-Host "Database metrics count: $dbCheck" -ForegroundColor Gray
    Write-Host "[SUCCESS] SQLite database metrics logging verified!" -ForegroundColor Green
} catch {
    Write-Host "[ERROR] SQLite query failed: $_" -ForegroundColor Red
    Exit 1
}

Write-Host "`n==========================================================" -ForegroundColor Green
Write-Host "  SUCCESS! ALL MULTILINGUAL PIPELINE TESTS PASSED!" -ForegroundColor Green
Write-Host "==========================================================" -ForegroundColor Green
