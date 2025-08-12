#!/usr/bin/env Rscript

# Chunk-based XML to CSV processing
# Processes a subset of files from multiple tar archives into a single CSV

options(editor = "vi")
library(data.table)
library(xml2)

# Install tidypmc if not already installed (with lock protection and timeout)
if (!require(tidypmc, quietly = TRUE)) {
    lock_file <- "/tmp/tidypmc_install.lock"
    max_wait <- 120  # Maximum wait time in seconds
    wait_time <- 0
    
    # Check for stale lock file (older than 5 minutes)
    if (file.exists(lock_file)) {
        lock_age <- difftime(Sys.time(), file.info(lock_file)$mtime, units = "secs")
        if (lock_age > 300) {
            cat("Removing stale lock file (age:", lock_age, "seconds)\n")
            file.remove(lock_file)
        }
    }
    
    # Use flock for atomic package installation
    if (!file.exists(lock_file)) {
        cat("Installing tidypmc...\n")
        file.create(lock_file)
        tryCatch({
            # Try installing from GitHub directly (CRAN may not have it)
            if (!require(remotes, quietly = TRUE)) {
                install.packages("remotes", repos = "https://cran.r-project.org/")
            }
            library(remotes)
            remotes::install_github("ropensci/tidypmc", quiet = FALSE, upgrade = "never")
        }, error = function(e) {
            cat("Installation failed with error:", e$message, "\n")
            stop("Failed to install tidypmc")
        }, finally = {
            if (file.exists(lock_file)) file.remove(lock_file)
        })
    } else {
        # Wait for other process to finish installation with timeout
        cat("Waiting for tidypmc installation to complete...\n")
        while (file.exists(lock_file) && wait_time < max_wait) {
            Sys.sleep(5)
            wait_time <- wait_time + 5
        }
        if (wait_time >= max_wait) {
            cat("Timeout waiting for installation, removing lock file and retrying\n")
            file.remove(lock_file)
            # Recursive call to retry installation
            source(commandArgs()[length(commandArgs())])
            quit(save = "no")
        }
    }
    library(tidypmc)
}

library(doParallel)

# Setup parallel processing with random port to avoid conflicts
n_cores <- 16  # Reduced from 60 since chunks are smaller
cat(sprintf("Using %d cores for parallel processing\n", n_cores))

# Try to create cluster with random port, retry if needed
max_attempts <- 5
for (attempt in 1:max_attempts) {
  tryCatch({
    # Use random port between 11000 and 12000
    port <- as.integer(runif(1, 11000, 12000))
    cl <- makeCluster(n_cores, type = "PSOCK", port = port)
    registerDoParallel(cl)
    cat(sprintf("Successfully created cluster on port %d\n", port))
    break
  }, error = function(e) {
    if (attempt == max_attempts) {
      cat("Failed to create parallel cluster after multiple attempts. Running sequentially.\n")
      n_cores <- 1
      cl <- NULL
    } else {
      cat(sprintf("Attempt %d failed: %s. Retrying...\n", attempt, e$message))
      Sys.sleep(1)
    }
  })
}

# Export required libraries to cluster workers (only if cluster was created)
if (!is.null(cl)) {
  clusterEvalQ(cl, {
    library(xml2)
    library(tidypmc)
  })
}

# Read chunk filelist
chunk_files <- fread(snakemake@input$filelist, sep = "\t", header = TRUE)
comm_tar_files <- snakemake@input$comm_tars
noncomm_tar_files <- snakemake@input$noncomm_tars

cat(sprintf("Processing chunk with %d files\n", nrow(chunk_files)))

# Group files by their PMC dataset
files_by_pmc <- split(chunk_files, chunk_files$pmc_dataset)

# Create CSV output file path
csv_output_file <- snakemake@output[[1]]
csv_output_dir <- dirname(csv_output_file)
if (!dir.exists(csv_output_dir)) {
  dir.create(csv_output_dir, recursive = TRUE, showWarnings = FALSE)
}

# Initialize combined data frame
all_data <- list()
temp_extract_dir <- file.path("temp", "batch_extract", sprintf("chunk_%s_%d", snakemake@wildcards$xmlchunk, Sys.getpid()))

for (pmc_id in names(files_by_pmc)) {
  pmc_files <- files_by_pmc[[pmc_id]]
  
  # Find the corresponding tar files - need to match PMC012 to PMC012xxxxxx pattern
  # The pmc_id is like "PMC012", but tar files are named like "PMC012xxxxxx"
  pmc_number <- gsub("PMC", "", pmc_id)  # Extract just the number part
  tar_pattern <- sprintf("PMC%sxxxxxx", pmc_number)
  
  # Get commercial tar file
  comm_tar <- comm_tar_files[grep(tar_pattern, comm_tar_files)]
  if (length(comm_tar) == 0) {
    cat(sprintf("WARNING: No commercial tar file found for %s (pattern: %s)\n", pmc_id, tar_pattern))
    next
  }
  
  # Get non-commercial tar file  
  noncomm_tar <- noncomm_tar_files[grep(tar_pattern, noncomm_tar_files)]
  
  cat(sprintf("Processing %d files from %s\n", nrow(pmc_files), pmc_id))
  if (length(comm_tar) > 0 && file.exists(comm_tar) && file.size(comm_tar) > 0) {
    cat(sprintf("  Using commercial tar: %s\n", basename(comm_tar)))
  }
  if (length(noncomm_tar) > 0 && file.exists(noncomm_tar) && file.size(noncomm_tar) > 0) {
    cat(sprintf("  Using non-commercial tar: %s\n", basename(noncomm_tar)))
  }
  
  # Process in smaller batches
  BATCH_SIZE <- 500
  n_batches <- ceiling(nrow(pmc_files) / BATCH_SIZE)
  
  for (batch_idx in 1:n_batches) {
    start_idx <- (batch_idx - 1) * BATCH_SIZE + 1
    end_idx <- min(batch_idx * BATCH_SIZE, nrow(pmc_files))
    
    batch_files <- pmc_files$`Article File`[start_idx:end_idx]
    
    # Create temp directory for this batch
    dir.create(temp_extract_dir, showWarnings = FALSE, recursive = TRUE)
    
    # Extract files from tar archives
    # Write file list to temp file to avoid command line length limits
    temp_file_list <- tempfile(pattern = "extract_list_", tmpdir = "/tmp")
    writeLines(batch_files, temp_file_list)
    
    # Track which files were successfully extracted
    extracted_files <- character(0)
    
    # Try commercial tar first (if it exists and has content)
    if (length(comm_tar) > 0 && file.exists(comm_tar) && file.size(comm_tar) > 0) {
      # Extract and get list of successfully extracted files
      extract_cmd <- sprintf("tar -xzf %s -C %s -T %s 2>/dev/null && tar -tzf %s -T %s 2>/dev/null", 
                           comm_tar, temp_extract_dir, temp_file_list,
                           comm_tar, temp_file_list)
      extracted_from_comm <- system(extract_cmd, intern = TRUE, ignore.stderr = TRUE)
      if (length(extracted_from_comm) > 0) {
        extracted_files <- c(extracted_files, extracted_from_comm)
        cat(sprintf("    Extracted %d files from commercial tar\n", length(extracted_from_comm)))
      }
    }
    
    # For non-commercial, only try files not already extracted
    if (length(noncomm_tar) > 0 && file.exists(noncomm_tar) && file.size(noncomm_tar) > 0) {
      # Create list of files not yet extracted
      remaining_files <- setdiff(batch_files, basename(extracted_files))
      
      if (length(remaining_files) > 0) {
        temp_remaining_list <- tempfile(pattern = "remaining_list_", tmpdir = "/tmp")
        writeLines(remaining_files, temp_remaining_list)
        
        extract_cmd <- sprintf("tar -xzf %s -C %s -T %s 2>/dev/null && tar -tzf %s -T %s 2>/dev/null", 
                             noncomm_tar, temp_extract_dir, temp_remaining_list,
                             noncomm_tar, temp_remaining_list)
        extracted_from_noncomm <- system(extract_cmd, intern = TRUE, ignore.stderr = TRUE)
        
        if (length(extracted_from_noncomm) > 0) {
          extracted_files <- c(extracted_files, extracted_from_noncomm)
          cat(sprintf("    Extracted %d additional files from non-commercial tar\n", length(extracted_from_noncomm)))
        }
        
        unlink(temp_remaining_list)
      }
    }
    
    # Clean up temp file list
    unlink(temp_file_list)
    
    if (length(extracted_files) == 0) {
      cat(sprintf("    WARNING: No files extracted for batch %d\n", batch_idx))
      next
    }
    
    # Process this batch in parallel and return data frames
    batch_data <- foreach(f = batch_files, .errorhandling = 'pass', .packages = c('xml2', 'tidypmc')) %dopar% {
      tryCatch({
        xml_path <- file.path(temp_extract_dir, f)
        
        # Check if file exists
        if (!file.exists(xml_path)) {
          xml_path_alt <- file.path(temp_extract_dir, basename(f))
          if (!file.exists(xml_path_alt)) {
            return(NULL)
          }
          xml_path <- xml_path_alt
        }
        
        # Extract PMCID from filename
        pmcid <- gsub("\\.xml$", "", basename(f))
        
        # Convert XML to data frame
        xml_file <- tryCatch({
          read_xml(xml_path)
        }, error = function(e) {
          return(NULL)
        })
        
        if (is.null(xml_file)) {
          return(NULL)
        }
        
        pmc_data <- tryCatch({
          pmc_text(xml_file)
        }, error = function(e) {
          return(NULL)
        })
        
        # Check if pmc_text returned valid data
        if (!is.null(pmc_data) && nrow(pmc_data) > 0) {
          # Add pmcid column
          pmc_data$pmcid <- pmcid
          return(pmc_data)
        } else {
          return(NULL)
        }
      }, error = function(err) {
        return(NULL)
      })
    }
    
    # Filter out NULLs and errors before adding to results
    valid_batch_data <- batch_data[!sapply(batch_data, is.null)]
    valid_batch_data <- valid_batch_data[!sapply(valid_batch_data, function(x) inherits(x, "error"))]
    
    # Add to combined results
    if (length(valid_batch_data) > 0) {
      all_data <- c(all_data, valid_batch_data)
    }
    
    # Clean up this batch's extracted files
    unlink(temp_extract_dir, recursive = TRUE)
    
    # Force garbage collection
    gc()
    
    if (batch_idx %% 10 == 0) {
      cat(sprintf("  Completed %d/%d batches for %s\n", batch_idx, n_batches, pmc_id))
    }
  }
}

# Combine all data frames and write to single CSV
cat(sprintf("\n=== CHUNK SUMMARY ===\n"))
cat(sprintf("Total files attempted: %d\n", nrow(chunk_files)))
cat(sprintf("Successfully converted: %d\n", length(all_data)))
cat(sprintf("Failed conversions: %d\n", nrow(chunk_files) - length(all_data)))

if (length(all_data) > 0) {
  # Combine all data frames into one
  combined_df <- rbindlist(all_data, fill = TRUE)
  
  # Write to single CSV file
  fwrite(combined_df, file = csv_output_file, row.names = FALSE)
  cat(sprintf("Combined data written to: %s\n", csv_output_file))
  cat(sprintf("Total rows in combined CSV: %d\n", nrow(combined_df)))
} else {
  warning("No files were successfully converted!")
  # Write empty CSV with expected columns
  empty_df <- data.frame(pmcid = character(),
                         section = character(),
                         paragraph = integer(),
                         sentence = integer(),
                         text = character())
  fwrite(empty_df, file = csv_output_file, row.names = FALSE)
}

# Clean up parallel cluster
if (!is.null(cl)) {
  stopCluster(cl)
}