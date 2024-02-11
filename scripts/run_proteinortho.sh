ASSEMBLIES_PATH="/home/gomez/gomez/assemblies_linkbert_7/"
proteinortho -cpus=64 -threads_per_process=4 -clean -synteny -singles -nograph -project=link $ASSEMBLIES_PATH/*/*/*.faa
