#!/bin/bash

# This script synchronizes subdirectories from a source directory to a target directory.
# It loops through each subdirectory in the target directory and checks if the corresponding subdirectory exists in the source directory.
# If it exists, it uses rsync to synchronize the contents of the subdirectory from the source to the target directory.

SOURCE_DIR="/home/gomez/gomez/assemblies_linkbert/"
TARGET_DIR="/home/gomez/gomez/assemblies_linkbert_7/"

# Loop through each subdirectory in the target
for dir in $(find $TARGET_DIR -type d); do
    # Calculate subdirectory path relative to target
    REL_DIR=${dir#$TARGET_DIR}

    # Sync only if the corresponding subdirectory exists in the source
    if [ -d "${SOURCE_DIR}${REL_DIR}" ]; then
        rsync -av --existing "${SOURCE_DIR}${REL_DIR}/" "${TARGET_DIR}${REL_DIR}/"
    fi
done
