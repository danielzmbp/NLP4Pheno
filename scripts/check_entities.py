import json
import os

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

def process_entities():
	"""
	Process entities based on hardcoded labels and input file.
	"""
	try:
		# Hardcoded input file and entities
		input_file = "label/project-5-at-2025-03-26-15-24-95adf201.json"  # Hardcoded input file path
		output_dir = "./notebooks/entities"
		os.makedirs(output_dir, exist_ok=True)  # Ensure the output directory exists

		# Hardcoded entities and their output files
		entities = [
			{"label": "STRAIN", "output_file": "strain.txt"},
			{"label": "SPECIES", "output_file": "species.txt"},
			{"label": "ISOLATE", "output_file": "isolate.txt"},
			{"label": "COMPOUND", "output_file": "compound.txt"},
			{"label": "MEDIUM", "output_file": "medium.txt"},
			{"label": "ORGANISM", "output_file": "organism.txt"},
			{"label": "PHENOTYPE", "output_file": "phenotype.txt"},
			{"label": "EFFECT", "output_file": "effect.txt"},
			{"label": "DISEASE", "output_file": "disease.txt"},
		]

		for entity in entities:
			label = entity["label"]
			output_file = entity["output_file"]
			# Prepend the output directory to the output file
			output_file_path = os.path.join(output_dir, output_file)
			extract_unique_entities(input_file, output_file_path, label)

	except Exception as e:
		print(f"An error occurred while processing entities: {e}")

if __name__ == "__main__":
	# Example usage
	process_entities()
