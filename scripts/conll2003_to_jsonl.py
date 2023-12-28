import argparse
import json

def conll2003_to_jsonl(input_file, output_file):
	with open(input_file, 'r') as file:
		lines = file.readlines()

	data = []
	tokens = []
	ner_tags = []
	for idx, line in enumerate(lines[1:]):
		if line != '\n':
			token, _, _, ner = line.strip().split(' ')
			tokens.append(token)
			ner_tags.append(ner)
		else:
			data.append({
				"tokens": tokens,
				"ner_tags": ner_tags,
				"id": str(len(data))
			})
			tokens = []
			ner_tags = []

	with open(output_file, 'w') as outfile:
		for entry in data:
			json.dump(entry, outfile)
			outfile.write('\n')

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Convert CONLL2003 file to JSONL format')
	parser.add_argument('input_file', type=str, help='Path to the input CONLL2003 file')
	parser.add_argument('output_file', type=str, help='Path to the output JSONL file')
	args = parser.parse_args()

	conll2003_to_jsonl(args.input_file, args.output_file)
