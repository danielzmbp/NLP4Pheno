from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
from transformers import pipeline
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset
import argparse
import pandas as pd


class ListDataset(Dataset):
    def __init__(self, original_list):
        self.original_list = original_list

    def __len__(self):
        return len(self.original_list)

    def __getitem__(self, i):
        return self.original_list[i]


parser = argparse.ArgumentParser(description='Run REL on corpus')

parser.add_argument('--model', type=str, help='model name')
parser.add_argument('--device', type=int, default=0,
                    help='device to run model on')
parser.add_argument('--input', type=str, help='path to input parquet file')
parser.add_argument('--output', type=str, help='path to output file')

args = parser.parse_args()
m = args.model

path = f"REL_output/{m}/"

tokenizer = AutoTokenizer.from_pretrained(path)
model = AutoModelForSequenceClassification.from_pretrained(path)


nlp = pipeline(task='text-classification', model=model,
               tokenizer=tokenizer, device=args.device)

ner1 = m.split("-")[0]
ner2 = m.split("-")[1].split(":")[0]

ners = [ner1, ner2]
ners.remove("STRAIN")
n = ners[0]

df = pd.read_parquet(args.input)

df = df[df["formatted_text"] != ""]

dfn = df[df["ner"] == n]
sl = dfn.formatted_text.to_list()

dataset = ListDataset(sl)
result = []

for out in tqdm(nlp(dataset, batch_size=32), total=len(dataset)):
    result.append(out)

dfn.loc[:, "re_result"] = result

dfc = pd.concat([dfn, dfn.re_result.apply(pd.Series).rename(columns={"score": "rel_score"})],axis=1)

dfc["label"] = dfc.label.str.split("_",expand=True)[1].astype(int)

sents = dfc[dfc["label"] == 1]

sents.loc[:, "rel"] = m

sents.to_parquet(args.output)
