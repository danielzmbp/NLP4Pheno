#!/usr/bin/env python3

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import os
import sys

# Check if input file exists and has content
if not os.path.exists(snakemake.input[0]):
    print(f"Error: Input file {snakemake.input[0]} does not exist")
    sys.exit(1)

# Check if file is empty
if os.path.getsize(snakemake.input[0]) == 0:
    print(f"Warning: Input file {snakemake.input[0]} is empty - no data to convert")
    # Create empty parquet with schema
    schema = pa.schema([
        ('pmcid', pa.string()),
        ('section', pa.string()),
        ('paragraph', pa.int64()),
        ('sentence', pa.int64()),
        ('text', pa.string())
    ])
    empty_table = pa.table({
        'pmcid': [],
        'section': [],
        'paragraph': [],
        'sentence': [],
        'text': []
    }, schema=schema)
    pq.write_table(empty_table, snakemake.output[0])
    sys.exit(0)

print(f"Processing CSV file: {snakemake.input[0]}")

try:
    # Read the single CSV file directly
    # The R script now outputs a single CSV file with all data combined
    df = pd.read_csv(snakemake.input[0])
    
    # Check if dataframe is empty
    if df.empty:
        print("Warning: CSV file contains no data")
        # Create empty parquet with schema
        schema = pa.schema([
            ('pmcid', pa.string()),
            ('section', pa.string()),
            ('paragraph', pa.int64()),
            ('sentence', pa.int64()),
            ('text', pa.string())
        ])
        empty_table = pa.table({
            'pmcid': [],
            'section': [],
            'paragraph': [],
            'sentence': [],
            'text': []
        }, schema=schema)
        pq.write_table(empty_table, snakemake.output[0])
        sys.exit(0)
    
    print(f"Successfully read CSV with {len(df)} rows")
    
    # Convert to arrow table and write to parquet
    table = pa.Table.from_pandas(df)
    pq.write_table(table, snakemake.output[0], compression='snappy')
    print(f"Written parquet to {snakemake.output[0]}")
    
    # Remove the CSV file to save space (it's marked as temp in Snakefile anyway)
    if os.path.exists(snakemake.input[0]):
        os.remove(snakemake.input[0])
        print(f"Removed temporary CSV file: {snakemake.input[0]}")
    
except Exception as e:
    print(f"Error processing CSV file: {str(e)}")
    # Try to create empty parquet as fallback
    schema = pa.schema([
        ('pmcid', pa.string()),
        ('section', pa.string()),
        ('paragraph', pa.int64()),
        ('sentence', pa.int64()),
        ('text', pa.string())
    ])
    empty_table = pa.table({
        'pmcid': [],
        'section': [],
        'paragraph': [],
        'sentence': [],
        'text': []
    }, schema=schema)
    pq.write_table(empty_table, snakemake.output[0])
    print("Created empty parquet file as fallback")
    sys.exit(1)

