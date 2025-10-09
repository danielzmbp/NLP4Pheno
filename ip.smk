import pandas as pd
from Bio import Entrez
from Bio import SeqIO
import os
import subprocess


configfile: "config.yaml"


data = config["dataset"]
path = config["output_path"]
output_path = path + f"/assemblies_{data}/"

# Common resource configurations
COMMON_RESOURCES = {"slurm_partition": "cpu_il,cpu", "runtime": 300, "mem_mb": 4096}

DOWNLOAD_RESOURCES = {"slurm_partition": "cpu_il,cpu", "runtime": 60, "mem_mb": 2048}

INTERPROSCAN_RESOURCES = {
    "slurm_partition": "cpu_il,cpu",
    "runtime": 3600,
    "mem_mb": 32384,
}

# InterProScan configuration
INTERPROSCAN_PATH = (
    "/home/tu/tu_tu/tu_kmpaj01/ip/interproscan-5.75-106.0/interproscan.sh"
)
INTERPROSCAN_APPS = "Pfam"  # Can be expanded: SFLD,Hamap,PRINTS,ProSiteProfiles,SUPERFAMILY,SMART,CDD,PIRSR,ProSitePatterns,Pfam,PIRSF,NCBIfam

# Load strain/assembly mappings
assemblies = []
strains = []
strains_assemblies_file = f"{path}/preds{data}/REL_output/strains_assemblies.txt"

if os.path.exists(strains_assemblies_file):
    with open(strains_assemblies_file, "r") as f:
        for line in f:
            s, a = line.strip().split("/")
            strains.append(s)
            assemblies.append(a)
else:
    # File doesn't exist yet - this is expected when rel_pred.smk hasn't run
    print(f"Warning: {strains_assemblies_file} not found. Run rel_pred.smk first.")


def get_interproscan_headers():
    """Return standard InterProScan TSV headers"""
    return [
        "Protein_accession",
        "Sequence_MD5_digest",
        "Sequence_length",
        "Analysis",
        "Signature_accession",
        "Signature_description",
        "Start_location",
        "Stop_location",
        "Score",
        "Status",
        "Date",
        "InterPro_accession",
        "InterPro_description",
        "GO_annotations",
        "Pathways_annotations",
    ]


rule final:
    input:
        expand(
            f"{output_path}{{strain}}/{{assembly}}/annotation.parquet",
            zip,
            strain=strains,
            assembly=assemblies,
        ),
        expand(
            f"{output_path}{{strain}}/{{assembly}}/protein.faa",
            zip,
            strain=strains,
            assembly=assemblies,
        ),
        expand(
            f"{output_path}{{strain}}/{{assembly}}/genomic.cds",
            zip,
            strain=strains,
            assembly=assemblies,
        ),
        f"{path}/preds{data}/REL_output/strains_assemblies_downloaded.txt",


localrules:
    download,
    unzip,


rule download:
    output:
        output_path + "{strain}/{assembly}.download_success",
    params:
        zip_file=output_path + "{strain}/{assembly}.zip",
    resources:
        **DOWNLOAD_RESOURCES,
    shell:
        """
        set +e  # Disable strict error mode for this rule

        mkdir -p $(dirname {output[0]})

        # Try download up to 3 times with increasing delays
        for attempt in 1 2 3; do
            echo "Download attempt $attempt for {wildcards.assembly}"

            if datasets download genome accession {wildcards.assembly} --include gff3,cds,protein,genome,seq-report --filename {params.zip_file} --assembly-version 'latest' --api-key $(cat .ncbi_api_key) --fast-zip-validation --no-progressbar; then
                echo "Successfully downloaded {wildcards.assembly} on attempt $attempt"
                touch {output[0]}
                exit 0
            else
                echo "Attempt $attempt failed for {wildcards.assembly}" >&2
                if [ $attempt -lt 3 ]; then
                    sleep_time=$((attempt * 5))
                    echo "Waiting $sleep_time seconds before retry..." >&2
                    sleep $sleep_time
                    # Clean up partial download
                    rm -f {params.zip_file}
                fi
            fi
        done

        echo "All download attempts failed for {wildcards.assembly}" >&2
        mkdir -p $(dirname {params.zip_file})
        touch {params.zip_file}  # Create empty zip file for failed downloads
        echo "DOWNLOAD_FAILED" > {output[0]}
        exit 0
        """


rule unzip:
    input:
        success_marker=output_path + "{strain}/{assembly}.download_success",
    params:
        zip_file=output_path + "{strain}/{assembly}.zip",
    output:
        protein=output_path + "{strain}/{assembly}/protein.faa",
        genomic_cds=output_path + "{strain}/{assembly}/genomic.cds",
    resources:
        **COMMON_RESOURCES,
    shell:
        """
        set +e  # Disable strict error mode for this rule

        # Check if download was successful
        if [ ! -f {input.success_marker} ] || grep -q "DOWNLOAD_FAILED" {input.success_marker}; then
            echo "Creating empty placeholder files for {wildcards.assembly} - download failed"
            mkdir -p {output_path}{wildcards.strain}/{wildcards.assembly}
            touch {output.protein} {output.genomic_cds}
            exit 0
        fi

        # Check if zip file doesn't exist (files were copied from previous run)
        if [ ! -f {params.zip_file} ]; then
            echo "Zip file not found for {wildcards.assembly} - assuming files already exist"
            # Create empty files if they don't exist
            mkdir -p {output_path}{wildcards.strain}/{wildcards.assembly}
            touch {output.protein} {output.genomic_cds}
            exit 0
        fi

        # Create output directory
        mkdir -p {output_path}{wildcards.strain}/{wildcards.assembly}

        # Extract all files to output directory (allow CRC errors, continue extraction)
        unzip -o -qq {params.zip_file} -d {output_path}{wildcards.strain}/{wildcards.assembly}/ 2>&1 | grep -v "bad CRC" || true

        # Check if extraction created the expected directory structure
        if [ ! -d "{output_path}{wildcards.strain}/{wildcards.assembly}/ncbi_dataset/data" ]; then
            echo "Warning: Extraction failed for {wildcards.assembly} - corrupt zip file"
            # Mark download as failed and remove corrupt zip
            rm -f {params.zip_file}
            echo "DOWNLOAD_FAILED" > {input.success_marker}
            touch {output.protein} {output.genomic_cds}
            exit 0
        fi

        # Find the actual assembly directory (handles version suffixes like .1, .2, etc.)
        assembly_dir=$(find {output_path}{wildcards.strain}/{wildcards.assembly}/ncbi_dataset/data -maxdepth 1 -type d -name "{wildcards.assembly}*" | head -1)

        if [ -z "$assembly_dir" ]; then
            echo "Warning: Could not find assembly directory for {wildcards.assembly}"
            touch {output.protein} {output.genomic_cds}
            exit 0
        fi

        # Handle protein.faa
        if [ -f "$assembly_dir/protein.faa" ] && [ -s "$assembly_dir/protein.faa" ]; then
            cp "$assembly_dir/protein.faa" {output.protein}
        else
            echo "Warning: protein.faa missing or empty for {wildcards.assembly}"
            touch {output.protein}
        fi

        # Handle cds_from_genomic.fna -> genomic.cds
        if [ -f "$assembly_dir/cds_from_genomic.fna" ] && [ -s "$assembly_dir/cds_from_genomic.fna" ]; then
            cp "$assembly_dir/cds_from_genomic.fna" {output.genomic_cds}
        else
            echo "Warning: cds_from_genomic.fna missing or empty for {wildcards.assembly}"
            touch {output.genomic_cds}
        fi

        # Clean up extracted directory structure
        rm -rf {output_path}{wildcards.strain}/{wildcards.assembly}/ncbi_dataset
        rm -f {output_path}{wildcards.strain}/{wildcards.assembly}/README.md
        rm -f {output_path}{wildcards.strain}/{wildcards.assembly}/md5sum.txt

        exit 0
        """


rule ip:
    input:
        output_path + "{strain}/{assembly}/protein.faa",
    output:
        output_path + "{strain}/{assembly}/annotation.parquet",
    params:
        annotation_tsv=output_path + "{strain}/{assembly}/annotation.tsv",
    threads: 32
    resources:
        **INTERPROSCAN_RESOURCES,
    run:
        import os
        import pandas as pd

        # Check if protein file is empty (from failed downloads)
        if not os.path.exists(input[0]) or os.path.getsize(input[0]) == 0:
            print(
                f"Creating empty annotation file for {wildcards.assembly} - protein file was empty"
            )
            # Create empty parquet directly
            headers = get_interproscan_headers()
            df = pd.DataFrame(columns=headers)
            df.drop(columns=["Status", "Date"], inplace=True)
            df.to_parquet(output[0])
        else:
            # Run InterProScan
            shell(
                f"{{INTERPROSCAN_PATH}} -T $TMPDIR -goterms -dra --iprlookup --cpu {{threads}} -i {{input[0]}} -o {{params.annotation_tsv}} -f TSV -appl {{INTERPROSCAN_APPS}}"
            )

            # Convert to parquet immediately
            headers = get_interproscan_headers()
            na_tokens = ["-", "None"]
            read_kwargs = dict(sep="\t", names=headers)
            try:
                df = pd.read_csv(params.annotation_tsv, engine="pyarrow", **read_kwargs)
            except Exception:
                df = pd.read_csv(
                    params.annotation_tsv, na_values=na_tokens, **read_kwargs
                )
            else:
                df.replace(na_tokens, pd.NA, inplace=True)
            df.drop(columns=["Status", "Date"], inplace=True)
            df.to_parquet(output[0])


rule record_downloads:
    input:
        expand(
            output_path + "{strain}/{assembly}.download_success",
            zip,
            strain=strains,
            assembly=assemblies,
        ),
    output:
        f"{path}/preds{data}/REL_output/strains_assemblies_downloaded.txt",
    resources:
        **COMMON_RESOURCES,
    run:
        successful_assemblies = []
        failed_assemblies = []

        for strain, assembly in zip(strains, assemblies):
            success_file = f"{output_path}{strain}/{assembly}.download_success"
            if os.path.exists(success_file):
                with open(success_file, "r") as f:
                    content = f.read().strip()
                    if content != "DOWNLOAD_FAILED":
                        # Check if zip file was successfully downloaded
                        zip_file = f"{output_path}{strain}/{assembly}.zip"
                        if os.path.exists(zip_file) and os.path.getsize(zip_file) > 0:
                            successful_assemblies.append(f"{strain}/{assembly}")
                        else:
                            failed_assemblies.append(f"{strain}/{assembly}")
                    else:
                        failed_assemblies.append(f"{strain}/{assembly}")
            else:
                failed_assemblies.append(f"{strain}/{assembly}")

                # Write successful assemblies file
        with open(output[0], "w") as f:
            for assembly in successful_assemblies:
                f.write(f"{assembly}\n")

                # Log results
        print(f"Successfully downloaded: {len(successful_assemblies)} assemblies")
        print(f"Failed downloads: {len(failed_assemblies)} assemblies")
        if failed_assemblies:
            print("Failed assemblies:", failed_assemblies[:10])  # Show first 10
