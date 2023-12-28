import pyarrow.parquet as pq
import more_itertools
import os

size = "1000"

corpus = pq.read_table('/home/gomez/biobert-pytorch/my_dataset/snakemake_pmc/pmc.parquet', columns=['text'])

corpus_text = corpus.to_pandas().text.to_list()

corpus_text = [i for i in corpus_text if i is not None]

corpus_text = [i for i in corpus_text if (len(i) <= 512) and (len(i) > 40)]

corpus_dir = f"corpus{size}"
os.makedirs(corpus_dir, exist_ok=True)
for i, batch in enumerate(more_itertools.batched(corpus_text, 500000)):
    if (i < (int(size) + 1)):
        with open(f"{corpus_dir}/{i:>03}.txt", "w") as f:
            for s in batch:
                try:
                    f.write(s)
                    f.write("\n")
                except:
                    pass
    else:
        pass
