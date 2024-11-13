import pandas as pd
import os

df = pd.read_csv(snakemake.input[0], header=None)

parquets = []
for f in df.iloc[:, 0]:
    parquet = f.replace('csv', 'parquet')
    try:
        # might need to index=0 but not sure yet
        d = pd.read_csv(f, index_col=0)
        d.loc[:, "pmcid"] = f.split('/')[-1].split('.')[0]
        d.to_parquet(parquet)
        parquets.append(parquet)
        os.remove(f)
        os.remove(f.replace('csv', 'xml'))
    except:
        pass


with open(snakemake.output[0], 'w') as output_file:
    for line in parquets:
        output_file.write(f"{line}\n")
