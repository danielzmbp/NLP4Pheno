#!/usr/bin/env python
"""Generate corpus from parquet file for NER prediction - Optimized version."""

import os
import sys
import polars as pl
import pyarrow.parquet as pq
import time
from datetime import datetime
import gc


def main():
    start_time = time.time()
    
    # Get command line arguments
    input_file = sys.argv[1]
    output_dir = sys.argv[2]
    corpus_size = int(sys.argv[3])
    
    # Convert to absolute path
    output_dir = os.path.abspath(output_dir)
    
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting optimized corpus generation")
    print(f"Input file: {input_file}")
    print(f"Output directory: {output_dir}")
    print(f"Target corpus size: {corpus_size} files")
    
    # Get file size for progress estimation
    file_size = os.path.getsize(input_file) / (1024**3)  # GB
    print(f"Input file size: {file_size:.2f} GB")
    
    pfile = pq.ParquetFile(input_file)
    
    # Create output directory
    try:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Created/verified output directory: {output_dir}")
    except Exception as e:
        print(f"Error creating directory {output_dir}: {e}")
        sys.exit(1)
    
    # First pass: count total lines
    print("First pass: Counting total lines...")
    total_lines = 0
    for batch in pfile.iter_batches(columns=["text"], batch_size=3000000):
        df = pl.from_arrow(batch)
        df = df.filter(pl.col("text").is_not_null())
        total_lines += len(df)
    
    print(f"Total lines in dataset: {total_lines:,}")
    lines_per_file = total_lines // corpus_size
    extra_lines = total_lines % corpus_size
    print(f"Each file will contain approximately {lines_per_file:,} lines")
    print(f"{extra_lines} files will have one extra line")
    
    # Initialize tracking variables
    file_index = 0
    buffer_of_lines = []
    total_lines_processed = 0
    files_written = []
    current_file_target = lines_per_file + (1 if file_index < extra_lines else 0)
    
    # Optimized parameters for 68GB file
    batch_size = 3000000  # Process 3M rows at a time for better throughput
    
    print(f"Processing in batches of {batch_size:,} rows...")
    print(f"Distributing {total_lines:,} lines across {corpus_size} files")
    
    # Process batches
    for batch_num, batch in enumerate(pfile.iter_batches(columns=["text"], batch_size=batch_size)):
        batch_start = time.time()
        
        if file_index >= corpus_size:
            print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Reached target corpus size. Stopping.")
            break
        
        # Process batch efficiently
        df = pl.from_arrow(batch)
        df = df.filter(pl.col("text").is_not_null())  # More efficient than drop_nulls
        
        # Process in chunks to avoid memory issues
        texts = df["text"].to_list()
        
        for i, line in enumerate(texts):
            buffer_of_lines.append(line)
            total_lines_processed += 1
            
            # Write file when buffer reaches target size for current file
            if len(buffer_of_lines) >= current_file_target:
                if file_index < corpus_size:
                    output_file = f"{output_dir}/{file_index:04d}.txt"
                    
                    # Write with optimized I/O
                    content = "\n".join(buffer_of_lines)
                    with open(output_file, "w", buffering=131072) as f:  # 128KB buffer
                        f.write(content)
                    
                    files_written.append(output_file)
                    
                    # Progress logging every 50 files  
                    if file_index % 50 == 0:
                        elapsed = time.time() - start_time
                        progress_pct = (file_index / corpus_size) * 100
                        print(f"File {file_index:04d}/{corpus_size} | Lines: {total_lines_processed:,} ({progress_pct:.1f}%)")
                    
                    file_index += 1
                    
                    # Update target for next file
                    current_file_target = lines_per_file + (1 if file_index < extra_lines else 0)
                    
                buffer_of_lines = []  # Clear buffer
                
                if file_index >= corpus_size:
                    break
        
        # Batch progress update (reduced verbosity)
        if batch_num % 10 == 0:
            elapsed = time.time() - start_time
            progress_pct = (file_index / corpus_size) * 100 if corpus_size > 0 else 0
            print(f"Progress: {file_index}/{corpus_size} files ({progress_pct:.1f}%) | {elapsed/60:.1f} min")
            
            # Force garbage collection periodically to free memory
            if batch_num % 10 == 0:
                gc.collect()
    
    # Write remaining lines to final file
    if buffer_of_lines and file_index < corpus_size:
        output_file = f"{output_dir}/{file_index:04d}.txt"
        with open(output_file, "w", buffering=131072) as f:
            f.write("\n".join(buffer_of_lines))
        files_written.append(output_file)
        print(f"Written final file {file_index:04d}.txt with {len(buffer_of_lines)} lines")
        file_index += 1
    
    # Final summary
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] CORPUS GENERATION COMPLETE")
    print(f"{'='*60}")
    print(f"Files created: {len(files_written)}")
    print(f"Total lines processed: {total_lines_processed:,}")
    print(f"Total time: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
    print(f"Average rate: {total_lines_processed/total_time:.0f} lines/sec")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()