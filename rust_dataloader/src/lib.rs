use pyo3::prelude::*;
use rayon::prelude::*;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use tokenizers::Tokenizer;

/// Tokenizes a batch of texts in parallel using Rayon and returns a flattened Vec<u32>
#[pyfunction]
fn tokenize_batch(texts: Vec<String>, tokenizer_path: String) -> PyResult<Vec<u32>> {
    let tokenizer = Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Failed to load tokenizer: {}", e)))?;

    let token_ids: Vec<u32> = texts
        .par_iter()
        .flat_map(|text| {
            tokenizer
                .encode(text.as_str(), true)
                .map(|enc| enc.get_ids().to_vec())
                .unwrap_or_default()
        })
        .collect();

    Ok(token_ids)
}

/// Reads a text file line-by-line, tokenizes in parallel chunks, and writes raw u32 bytes directly to a binary file
#[pyfunction]
fn tokenize_file_to_bin(
    input_path: String,
    tokenizer_path: String,
    output_path: String,
    chunk_size: usize,
) -> PyResult<usize> {
    let tokenizer = Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Failed to load tokenizer: {}", e)))?;

    let infile = File::open(&input_path)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("Failed to open input file: {}", e)))?;
    let reader = BufReader::new(infile);

    let outfile = File::create(&output_path)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("Failed to create output binary file: {}", e)))?;
    let mut writer = BufWriter::new(outfile);

    let mut total_tokens = 0;
    let mut lines_chunk = Vec::with_capacity(chunk_size);

    for line_result in reader.lines() {
        let line = line_result.map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("Read line error: {}", e)))?;
        lines_chunk.push(line);

        if lines_chunk.len() >= chunk_size {
            let tokens = process_and_write_chunk(&lines_chunk, &tokenizer, &mut writer)?;
            total_tokens += tokens;
            lines_chunk.clear();
        }
    }

    if !lines_chunk.is_empty() {
        let tokens = process_and_write_chunk(&lines_chunk, &tokenizer, &mut writer)?;
        total_tokens += tokens;
    }

    writer.flush()
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("Failed to flush output binary file: {}", e)))?;

    Ok(total_tokens)
}

fn process_and_write_chunk(
    lines: &[String],
    tokenizer: &Tokenizer,
    writer: &mut BufWriter<File>,
) -> PyResult<usize> {
    // Tokenize lines in parallel
    let chunk_tokens: Vec<u32> = lines
        .par_iter()
        .flat_map(|line| {
            tokenizer
                .encode(line.as_str(), true)
                .map(|enc| enc.get_ids().to_vec())
                .unwrap_or_default()
        })
        .collect();

    let byte_slice: &[u8] = bytemuck::cast_slice(&chunk_tokens);
    writer.write_all(byte_slice)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("Failed to write binary data: {}", e)))?;

    Ok(chunk_tokens.len())
}

/// PyO3 Module definition
#[pymodule]
fn rust_dataloader(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(tokenize_batch, m)?)?;
    m.add_function(wrap_pyfunction!(tokenize_file_to_bin, m)?)?;
    Ok(())
}
