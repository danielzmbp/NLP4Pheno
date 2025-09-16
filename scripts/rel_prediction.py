"""
Apply trained Relation Extraction models to entity pairs in corpus.

This script uses fine-tuned BERT-based RE models to classify relationships
between entity pairs identified by the NER pipeline. It processes sentences
containing masked entity pairs and predicts the semantic relationship.

The pipeline:
1. Loads NER predictions containing entity pairs
2. Filters for relevant entity types (e.g., STRAIN-COMPOUND pairs)
3. Applies trained RE model to classify relationships
4. Outputs relationship predictions with confidence scores

This completes the NLP extraction pipeline before genomic analysis.
"""

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
from transformers import pipeline
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset
import argparse


class ListDataset(Dataset):
    """
    Simple dataset wrapper for batch processing text with HuggingFace pipelines.
    
    Converts a list of texts into a PyTorch Dataset for efficient batch processing
    during relation extraction inference.
    """
    def __init__(self, original_list):
        """Initialize with list of formatted texts containing entity pairs."""
        self.original_list = original_list

    def __len__(self):
        """Return number of texts in dataset."""
        return len(self.original_list)

    def __getitem__(self, i):
        """Get text at index i."""
        return self.original_list[i]


# Parse command line arguments
parser = argparse.ArgumentParser(description='Apply Relation Extraction model to entity pairs')
parser.add_argument('--model', type=str, help='Relation type (e.g., STRAIN-COMPOUND:RESISTS)')
parser.add_argument('--device', type=int, default=0, help='GPU device ID')
parser.add_argument('--input', type=str, help='Input Parquet file with NER predictions')
parser.add_argument('--output', type=str, help='Output Parquet file for RE predictions')

args = parser.parse_args()
m = args.model  # Relation type identifier

# Load trained RE model and tokenizer
path = f"REL_output/{m}/"
tokenizer = AutoTokenizer.from_pretrained(path)
model = AutoModelForSequenceClassification.from_pretrained(path)

# Create inference pipeline with truncation enabled
nlp = pipeline(task='text-classification', model=model,
               tokenizer=tokenizer, device=args.device,
               truncation=True, max_length=512)

# Extract entity types from relation name
# e.g., "STRAIN-COMPOUND:RESISTS" -> ["STRAIN", "COMPOUND"]
ner1 = m.split("-")[0]  # First entity type
ner2 = m.split("-")[1].split(":")[0]  # Second entity type

# Identify the non-STRAIN entity type for filtering
ners = [ner1, ner2]
ners.remove("STRAIN")  # STRAIN is always one of the entities
n = ners[0]  # The other entity type

# Load NER predictions and filter for relevant entity pairs
df = pd.read_parquet(args.input)

# Remove empty formatted texts
df = df[df["formatted_text"] != ""]

# Filter for sentences containing the target entity type
# (e.g., for STRAIN-COMPOUND relations, filter for COMPOUND entities)
dfn = df[df["ner"] == n]
sl = dfn.formatted_text.to_list()

# Create dataset for batch processing
dataset = ListDataset(sl)
result = []

# Apply RE model to classify relationships
print(f"Processing {len(dataset)} entity pairs for relation {m}")
for out in tqdm(nlp(dataset, batch_size=32), total=len(dataset)):
    result.append(out)

# Add RE predictions to dataframe
dfn.loc[:, "re_result"] = result

# Expand RE results into separate columns
dfc = pd.concat([dfn, dfn.re_result.apply(pd.Series).rename(columns={"score": "rel_score"})], axis=1)

# Extract binary label (1 = positive relationship, 0 = negative)
dfc["label"] = dfc.label.str.split("_", expand=True)[1].astype(int)

# Keep only positive predictions (indicating the relationship exists)
sents = dfc[dfc["label"] == 1]

# Add relation type identifier
sents.loc[:, "rel"] = m

# Save predictions
print(f"Found {len(sents)} positive relationships of type {m}")
sents.to_parquet(args.output)
