import more_itertools
import os
import polars as pl

size = "1100"

corpus = pl.read_parquet(
    '../snakemake_PMC/pmc.parquet')

# corpus = corpus.filter(pl.col("text").str.len_chars() <= 512)

corpus_text = corpus["text"].to_list()

corpus_text = [i for i in corpus_text if i is not None]

corpus_dir = f"corpus{size}"
os.makedirs(corpus_dir, exist_ok=True)
for i, batch in enumerate(more_itertools.batched(corpus_text, 550000)):
    if (i < (int(size) + 1)):
        with open(f"{corpus_dir}/{i:>04}.txt", "w") as f:
            for s in batch:
                try:
                    f.write(s)
                    f.write("\n")
                except:
                    pass
    else:
        pass
