#!/usr/bin/env python
# coding=utf-8

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
	Load the model from the specified directory.
	"""
	model = AutoModelForTokenClassification.from_pretrained(model_dir)
	tokenizer = AutoTokenizer.from_pretrained(model_dir)
	return model, tokenizer

def predict(model, tokenizer, sentence):
	"""
	Predict the tokens for a given sentence.
	"""
	inputs = tokenizer(sentence, return_tensors="pt")
	outputs = model(**inputs)
	predictions = torch.argmax(outputs.logits, dim=-1)
	return predictions

def predict_on_file(model, tokenizer, file_path, output_file):
	"""
	Predict the tokens for each sentence in a given file and save the results to a Parquet file dataframe.
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
	model, tokenizer = load_model(model_dir)
	predict_on_file(model, tokenizer, file_path, output_file)

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="NER Prediction")
	parser.add_argument("--model_dir", type=str, help="Path to the model directory")
	parser.add_argument("--file_path", type=str, help="Path to the input file")
	parser.add_argument("--output_file", type=str, help="Path to the output Parquet file")
	args = parser.parse_args()
	main(args.model_dir, args.file_path, args.output_file)
