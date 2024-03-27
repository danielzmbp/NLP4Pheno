#!/bin/bash
# This script fixes the gff files produced by prodigal in order to work downstream with proteinortho
ASSEMBLIES_PATH="/home/gomez/gomez/rel_assemblies_7"

# Loop through each folder in rel_assemblies_7
for folder in "$ASSEMBLIES_PATH"/*/; do
	if [ -d "$folder" ]; then
		folder_name=$(basename "$folder")
	# 	echo "Processing folder: $folder_name"
		
	# 	# Loop through all .gff files in the current folder
	# 	for file in "$folder"/*.gff; do
	# 		echo "Processing $file..."
	# 		# Define a temporary file
	# 		temp_file="${file}.tmp"

	# 		# Process the file and save the output to the temporary file
	# 		awk '{
	# 			if ($0 ~ /^[^#]/ && $3 == "CDS") {
	# 				# Extract the full ID at the beginning of the line
	# 				full_id=$1;
	# 				# Replace the number before "_" in the ID with the full ID, ensuring the part after "_" is preserved
	# 				sub(/ID=[^;_]+_/, "ID=" full_id "_", $0);
	# 			}
	# 			print $0;
	# 		}' "$file" >"$temp_file"

	# 		# Move the temporary file back to the original file, replacing it
	# 		mv "$temp_file" "$file"
	# 		echo "Modified $file"
		# done
		proteinortho -cpus=64 -p=diamond -threads_per_process=4 -clean -nograph -project=$folder_name $ASSEMBLIES_PATH/$folder_name/*/*.faa
	fi
done
