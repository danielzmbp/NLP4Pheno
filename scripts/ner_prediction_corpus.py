from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
from tqdm import tqdm
import argparse
import pandas as pd
from torch.utils.data import Dataset


class ListDataset(Dataset):
    def __init__(self, original_list):
        self.original_list = original_list

    def __len__(self):
        return len(self.original_list)

    def __getitem__(self, i):
        return self.original_list[i]


parser = argparse.ArgumentParser(description='Run NER on corpus')

parser.add_argument('--model_path', type=str, help='path to model')
parser.add_argument('--device', type=int, default=0,
                    help='device to run model on')
parser.add_argument('--corpus', type=str, help='path to corpus file')
parser.add_argument('--output', type=str, help='path to output file')

args = parser.parse_args()
path = args.model_path

tokenizer = AutoTokenizer.from_pretrained(path, model_max_length=512)
model = AutoModelForTokenClassification.from_pretrained(path)

nlp = pipeline(task='ner', model=model, tokenizer=tokenizer,
               grouped_entities=True, ignore_subwords=True, device=args.device)

corpus = args.corpus

with open(corpus, "r") as f:
    text = f.read()

texts = text.split("\n")
texts.pop(-1)

dataset = ListDataset(texts)
result = []

for out in tqdm(nlp(dataset, batch_size=64), total=len(dataset)):
    result.append(out)

df = pd.DataFrame({"text": texts, "ner": result})
df.to_parquet(args.output)
