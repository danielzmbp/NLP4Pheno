import os
import polars as pl
import pyarrow.parquet as pq

size = 2812
pfile = pq.ParquetFile("./snakemake_PMC/pmc.parquet")

corpus_dir = f"corpus{size}"
os.makedirs(corpus_dir, exist_ok=True)

file_index = 0
buffer_of_lines = []

for batch in pfile.iter_batches(columns=["text"], batch_size=100000):
    df = pl.from_arrow(batch).select("text").drop_nulls()
    for line in df["text"].to_list():
        buffer_of_lines.append(line)
        if len(buffer_of_lines) >= 550000:
            if file_index <= size:
                with open(f"{corpus_dir}/{file_index:04}.txt", "w") as f:
                    f.write("\n".join(buffer_of_lines))
                file_index += 1
            buffer_of_lines.clear()
    if file_index > size:
        break

if buffer_of_lines and file_index <= size:
    with open(f"{corpus_dir}/{file_index:04}.txt", "w") as f:
        f.write("\n".join(buffer_of_lines))
