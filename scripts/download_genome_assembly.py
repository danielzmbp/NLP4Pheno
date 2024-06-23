from Bio import Entrez
from Bio import SeqIO
import os

def download_genome(assembly_accession):
    """Downloads a genome from NCBI based on its assembly accession.

    Args:
        assembly_accession (str): The NCBI assembly accession (e.g., "GCA_000001405.29")

    Returns:
        str: The path to the downloaded FASTA file.
    """

    Entrez.email = "d.gomez@lmu.de"  # Replace with your actual email address

    # Search for the assembly summary using Entrez
    handle = Entrez.esearch(db="assembly", term=assembly_accession)
    record = Entrez.read(handle)
    assembly_summary_id = record["IdList"][0]

    # Fetch the download link for the FASTA file
    esummary_handle = Entrez.esummary(db="assembly", id=assembly_summary_id)
    esummary_record = Entrez.read(esummary_handle)
    fasta_download_url = esummary_record["DocumentSummarySet"]["DocumentSummary"][0]["FtpPath_RefSeq"]

    # Determine the filename based on the assembly accession
    filename = f"{assembly_accession}_genomic.fna.gz"
    output_path = os.path.join("downloaded_genomes", filename)  # Customize the output directory

    ## problem here, should just download the ftp and not use entrez efetch
    # Download the genome
    download_result = Entrez.efetch(db="nucleotide", 
                                    rettype="fasta", 
                                    retmode="text", 
                                    id=fasta_download_url)
    
    # Save the downloaded genome
    with open(output_path, "w") as f:
        f.write(download_result.read())

    return output_path

# Example usage
assembly_to_download = "GCA_000001405.29" 
downloaded_file_path = download_genome(assembly_to_download)
print(f"Genome downloaded to: {downloaded_file_path}") 

