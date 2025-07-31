options(editor = "vi")
library(data.table)
library(xml2)

# Install tidypmc if not already installed (with lock protection)
if (!require(tidypmc, quietly = TRUE)) {
    lock_file <- "/tmp/tidypmc_install.lock"
    
    # Use flock for atomic package installation
    if (!file.exists(lock_file)) {
        cat("Installing tidypmc...\n")
        file.create(lock_file)
        tryCatch({
            # Try installing with dependencies and verbose output
            install.packages("tidypmc", repos = "https://cran.r-project.org/", 
                           dependencies = TRUE, quiet = FALSE, verbose = TRUE)
        }, error = function(e) {
            cat("Installation failed with error:", e$message, "\n")
            # Try installing from GitHub as fallback
            if (!require(remotes, quietly = TRUE)) {
                install.packages("remotes", repos = "https://cran.r-project.org/")
            }
            library(remotes)
            install_github("ropensci/tidypmc")
        }, finally = {
            if (file.exists(lock_file)) file.remove(lock_file)
        })
    } else {
        # Wait for other process to finish installation
        cat("Waiting for tidypmc installation to complete...\n")
        while (file.exists(lock_file)) {
            Sys.sleep(5)
        }
    }
    library(tidypmc)
}

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
