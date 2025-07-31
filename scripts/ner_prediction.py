#!/usr/bin/env python
# coding=utf-8
"""
Apply trained NER models to scientific literature corpus.

This script uses the fine-tuned BERT-based NER models to identify entities
in large-scale text corpora. It processes text files and outputs predictions
in Parquet format for efficient storage and downstream processing.

The predictions include:
- Entity tokens identified by the model
- Entity types (STRAIN, SPECIES, COMPOUND, etc.)
- Confidence scores for each prediction

Output is used by the relation extraction pipeline to identify relationships
between detected entities.
"""

import argparse
import torch
import pandas as pd
from tqdm import tqdm
from transformers import (
	AutoModelForTokenClassification, 
	AutoTokenizer
)
def load_model(model_dir):
	"""
	Load trained NER model and tokenizer from directory.
	
	Args:
		model_dir: Path to directory containing model weights and tokenizer
		
	Returns:
		tuple: (model, tokenizer) ready for inference
	"""
	model = AutoModelForTokenClassification.from_pretrained(model_dir)
	tokenizer = AutoTokenizer.from_pretrained(model_dir)
	return model, tokenizer

def predict(model, tokenizer, sentence):
	"""
	Predict entity labels for tokens in a sentence.
	
	Performs tokenization and NER inference on input text, returning
	predicted entity labels for each token.
	
	Args:
		model: Trained NER model
		tokenizer: Corresponding tokenizer
		sentence: Input text to analyze
		
	Returns:
		Tensor: Predicted label IDs for each token
	"""
	inputs = tokenizer(sentence, return_tensors="pt")
	outputs = model(**inputs)
	predictions = torch.argmax(outputs.logits, dim=-1)
	return predictions

def predict_on_file(model, tokenizer, file_path, output_file):
	"""
	Process entire text file and save NER predictions.
	
	Reads sentences from input file, performs NER prediction on each,
	and saves results in Parquet format for efficient storage.
	
	Args:
		model: Trained NER model
		tokenizer: Corresponding tokenizer
		file_path: Path to input text file (one sentence per line)
		output_file: Path for output Parquet file
	"""
	data = []
	with open(file_path, 'r') as f:
		for line in tqdm(f, desc="Predicting"):
			sentence = line.strip()
			predictions = predict(model, tokenizer, sentence)
			data.append({"Sentence": sentence, "Predictions": predictions.tolist()})
	df = pd.DataFrame(data)
	df.to_parquet(output_file)

def main(model_dir, file_path, output_file):
	"""
	Main execution function for NER prediction pipeline.
	
	Loads model and processes input file to generate entity predictions.
	
	Args:
		model_dir: Directory containing trained NER model
		file_path: Input text file to process
		output_file: Output path for Parquet predictions
	"""
	model, tokenizer = load_model(model_dir)
	predict_on_file(model, tokenizer, file_path, output_file)

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="NER Prediction")
	parser.add_argument("--model_dir", type=str, help="Path to the model directory")
	parser.add_argument("--file_path", type=str, help="Path to the input file")
	parser.add_argument("--output_file", type=str, help="Path to the output Parquet file")
	args = parser.parse_args()
	main(args.model_dir, args.file_path, args.output_file)
