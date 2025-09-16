import json
import os
import argparse
import yaml

def extract_unique_entities(input_file, output_file, label_filter):
	"""
	Extract unique entities based on a specific label and write them to a file.

	:param input_file: Path to the input JSON file.
	:param output_file: Path to the output text file.
	:param label_filter: The label to filter entities by (e.g., "EFFECT", "ISOLATE").
	"""
	try:
		with open(input_file, "r") as f:
			json_file = json.load(f)

		entity_catalog = []
		for item in json_file:
			if item.get("annotations"):
				for annotation in item["annotations"]:
					if annotation.get("result"):
						for result in annotation["result"]:
							if "value" in result and "labels" in result["value"]:
								if result["value"]["labels"][0] == label_filter:
									entity_catalog.append(result["value"]["text"])

		# Remove duplicates and sort
		unique_entities = sorted(set(entity_catalog))

		# Write to output file
		with open(output_file, "w") as f:
			f.write("\n".join(unique_entities))

		print(f"Unique entities with label '{label_filter}' have been written to {output_file}.")

	except Exception as e:
		print(f"An error occurred: {e}")

def load_config(config_path):
	"""
	Load configuration from YAML file.
	"""
	with open(config_path, 'r') as f:
		return yaml.safe_load(f)

def process_entities(input_file, output_dir, labels):
	"""
	Process entities based on provided labels and input file.
	"""
	try:
		os.makedirs(output_dir, exist_ok=True)  # Ensure the output directory exists

		for label in labels:
			output_file = f"{label.lower()}.txt"
			output_file_path = os.path.join(output_dir, output_file)
			extract_unique_entities(input_file, output_file_path, label)

	except Exception as e:
		print(f"An error occurred while processing entities: {e}")

def main():
	parser = argparse.ArgumentParser(description='Extract unique entities from Label Studio annotations')
	parser.add_argument('--input', type=str, help='Path to Label Studio JSON file')
	parser.add_argument('--output-dir', type=str, default='notebooks/entities', 
						help='Output directory for entity files (default: notebooks/entities)')
	parser.add_argument('--config', type=str, default='config.yaml',
						help='Path to config.yaml file (default: config.yaml)')
	
	args = parser.parse_args()
	
	# Load config
	config = load_config(args.config)
	
	# Use input from args or config
	input_file = args.input if args.input else config.get('input_file')
	if not input_file:
		print("Error: No input file specified. Use --input or set input_file in config.yaml")
		return
	
	# Get NER labels from config
	labels = config.get('ner_labels', [])
	if not labels:
		print("Error: No ner_labels found in config.yaml")
		return
	
	print(f"Processing entities from: {input_file}")
	print(f"Output directory: {args.output_dir}")
	print(f"Entity types: {', '.join(labels)}")
	
	process_entities(input_file, args.output_dir, labels)

if __name__ == "__main__":
	main()
