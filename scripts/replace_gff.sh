#!/bin/bash
# This script fixes the gff files produced by prodigal in order to work downstream with proteinortho
ASSEMBLIES_PATH="/home/gomez/gomez/assemblies_linkbert_7/"

# Loop through all .gff files matching the pattern */*/*.gff
for file in $ASSEMBLIES_PATH/*/*/*.gff; do
	echo "Processing $file..."
	# Define a temporary file
	temp_file="${file}.tmp"

	# Process the file and save the output to the temporary file
	awk '{
        if ($0 ~ /^[^#]/ && $3 == "CDS") {
            # Extract the full ID at the beginning of the line
            full_id=$1;
            # Replace the number before "_" in the ID with the full ID, ensuring the part after "_" is preserved
            sub(/ID=[^;_]+_/, "ID=" full_id "_", $0);
        }
        print $0;
    }' "$file" >"$temp_file"

	# Move the temporary file back to the original file, replacing it
	mv "$temp_file" "$file"
	echo "Modified $file"
done
