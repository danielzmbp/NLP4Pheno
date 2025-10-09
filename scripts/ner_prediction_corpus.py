from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
from tqdm.auto import tqdm
import argparse
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import re
import torch


class ListDataset(Dataset):
    def __init__(self, original_list):
        self.original_list = original_list

    def __len__(self):
        return len(self.original_list)

    def __getitem__(self, i):
        return self.original_list[i]


parser = argparse.ArgumentParser(description='Run NER on corpus')

parser.add_argument('--model_path', type=str, help='path to model')
parser.add_argument('--model', dest='model_path')
parser.add_argument('--device', type=int, default=0,
                    help='device to run model on')
parser.add_argument('--corpus', type=str, help='path to corpus file')
parser.add_argument('--output', type=str, help='path to output file')

parser.add_argument('--no-half-precision', dest='half_precision', action='store_false',
                    help='run inference in float32 on GPU')
parser.set_defaults(half_precision=True)

args = parser.parse_args()
path = args.model_path

if path is None:
    raise ValueError('Model path must be provided via --model_path or --model.')

tokenizer = AutoTokenizer.from_pretrained(path, model_max_length=512)
model_kwargs = {}
if args.half_precision:
    if args.device < 0:
        raise ValueError('Half precision requires a GPU device (device index >= 0). Use --no-half-precision for CPU runs.')
    model_kwargs['torch_dtype'] = torch.float16

model = AutoModelForTokenClassification.from_pretrained(path, **model_kwargs)

if args.half_precision:
    model = model.to(f"cuda:{args.device}")

pipeline_kwargs = dict(
    task='ner',
    model=model,
    tokenizer=tokenizer,
    aggregation_strategy="max",
    device=args.device,
)

if args.half_precision:
    pipeline_kwargs['torch_dtype'] = torch.float16

nlp = pipeline(**pipeline_kwargs)

corpus = args.corpus

with open(corpus, "r") as f:
    text = f.read()

texts = text.split("\n")
texts.pop(-1)

# Replace hyphens between words with spaces using regex
texts = [re.sub(r'(?<=\w)-(?=\w)', ' ', sentence) for sentence in texts]

dataset = ListDataset(texts)
result = []

with torch.inference_mode():
    for out in tqdm(nlp(dataset, batch_size=64), total=len(dataset)):
        result.append(out)

df = pd.DataFrame({"text": texts, "ner": result})
df.to_parquet(args.output)
