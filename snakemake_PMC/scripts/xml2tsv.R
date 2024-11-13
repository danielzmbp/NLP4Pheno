library(data.table)
library(xml2)
library(tidypmc)
library(doParallel)
registerDoParallel(cores=20)

# Read input file using data.table
t <- fread(snakemake@input[[1]], sep = "\t", header = TRUE)

# Convert XML to CSV in parallel
csv_files <- foreach(f = t$`Article File`) %dopar% {
  tryCatch({
    xml_file <- read_xml(f)
    csv_file <- gsub("xml", "csv", f)
    write.csv(pmc_text(xml_file), file = csv_file, row.names=FALSE) 
    csv_file
  }, error = function(err) {
    # Error handling - you can log the error or substitute default content
    warning(paste("Error processing file:", f, "- Message:", err))
    NULL  # Return NULL to indicate the file was not processed correctly
  })
}

# Write output using data.table. Filter out any NULL values (failed files).
writeLines(unlist(csv_files[!is.null(csv_files)]), snakemake@output[[1]])
