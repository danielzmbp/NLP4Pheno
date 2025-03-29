from nervaluate import Evaluator
import json
import yaml
import os

def load_predictions(label):
	file_path = f"NER_output/{label}/test_predictions.txt"
	with open(file_path, 'r') as f:
		return f.read()

def load_ground_truth(label):
	file_path = f"NER/{label}/test.txt"
	with open(file_path, 'r') as f:
		return f.read()

def load_labels_from_config(config_path):
	with open(config_path, 'r') as f:
		config = yaml.safe_load(f)
	return config.get('ner_labels', [])

def main():
	# Load configuration
	config_file = "./config.yaml"
	config = load_labels_from_config(config_file)

	# Iterate over labels to evaluate predictions for each
	for label in config:
		ground_truth = load_ground_truth(label)
		ground_truth = ground_truth.replace("-DOCSTART- -X- O\n", "").replace(" -X- _ ","\t")
		predictions = load_predictions(label)

		# Initialize the evaluator
		evaluator = Evaluator(ground_truth, predictions, tags=[""],loader="conll")

		# Evaluate
		
		results, results_by_tag, result_indices, result_indices_by_tag = evaluator.evaluate()

		# Write results to file
		output_dir = f"./NER_output/{label}/"
		os.makedirs(output_dir, exist_ok=True)

		# Adjust results before writing
		adjusted_results = {key: round(value, 4) if isinstance(value, float) else value for key, value in results.items()}
		adjusted_results_by_tag = {
			tag: {key: round(value, 4) if isinstance(value, float) else value for key, value in metrics.items()}
			for tag, metrics in results_by_tag.items()
		}

		with open(os.path.join(output_dir, "overall_results.json"), "w") as overall_file:
			json.dump(adjusted_results, overall_file, indent=4)

		with open(os.path.join(output_dir, "results_per_tag.json"), "w") as per_tag_file:
			json.dump(adjusted_results_by_tag, per_tag_file, indent=4)

if __name__ == "__main__":
	main()