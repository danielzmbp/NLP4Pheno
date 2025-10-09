#!/usr/bin/env Rscript

# Chunk-based XML to CSV processing
# Processes a subset of files from multiple tar archives into a single CSV

options(editor = "vi")
library(data.table)
library(xml2)

# Ensure tidypmc is available from the preinstalled user library
conda_prefix <- Sys.getenv("CONDA_PREFIX", unset = "")
user_lib <- file.path(conda_prefix, "user-library")
if (nzchar(user_lib) && dir.exists(user_lib)) {
  .libPaths(c(user_lib, .libPaths()))
}

suppressPackageStartupMessages(library(tidypmc))

library(doParallel)

# Setup parallel processing with random port to avoid conflicts
n_cores <- snakemake@threads
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

# Prepare extraction workspace and output tracking
chunk_extract_root <- file.path("temp", "batch_extract", sprintf("chunk_%s_%d", snakemake@wildcards$xmlchunk, Sys.getpid()))
dir.create(chunk_extract_root, recursive = TRUE, showWarnings = FALSE)
on.exit({
  if (dir.exists(chunk_extract_root)) {
    unlink(chunk_extract_root, recursive = TRUE)
  }
}, add = TRUE)

if (file.exists(csv_output_file)) {
  file.remove(csv_output_file)
}

written_rows <- 0L
successful_pmcids <- character()

extract_selected <- function(tar_path, list_path, dest_dir) {
  if (length(tar_path) == 0) {
    return(invisible(NULL))
  }
  tar_path <- tar_path[1]
  if (!file.exists(tar_path) || file.size(tar_path) == 0) {
    return(invisible(NULL))
  }
  res <- system2("tar", args = c("-xzf", tar_path, "-C", dest_dir, "-T", list_path), stdout = NULL, stderr = NULL)
  if (!identical(res, 0L)) {
    warning(sprintf("Extraction exited with status %s for %s", res, basename(tar_path)))
  }
  invisible(NULL)
}

for (pmc_id in names(files_by_pmc)) {
  pmc_files <- files_by_pmc[[pmc_id]]

  pmc_number <- gsub("PMC", "", pmc_id)
  tar_pattern <- sprintf("PMC%sxxxxxx", pmc_number)

  comm_tar <- comm_tar_files[grep(tar_pattern, comm_tar_files)]
  noncomm_tar <- noncomm_tar_files[grep(tar_pattern, noncomm_tar_files)]

  cat(sprintf("Processing %d files from %s\n", nrow(pmc_files), pmc_id))
  if (length(comm_tar) > 0 && file.exists(comm_tar[1]) && file.size(comm_tar[1]) > 0) {
    cat(sprintf("  Using commercial tar: %s\n", basename(comm_tar[1])))
  }
  if (length(noncomm_tar) > 0 && file.exists(noncomm_tar[1]) && file.size(noncomm_tar[1]) > 0) {
    cat(sprintf("  Using non-commercial tar: %s\n", basename(noncomm_tar[1])))
  }

  pmc_extract_dir <- file.path(chunk_extract_root, pmc_id)
  dir.create(pmc_extract_dir, recursive = TRUE, showWarnings = FALSE)

  pmc_file_list <- unique(pmc_files$`Article File`)
  if (length(pmc_file_list) == 0) {
    cat(sprintf("  WARNING: No article files listed for %s\n", pmc_id))
    unlink(pmc_extract_dir, recursive = TRUE)
    next
  }

  list_path <- tempfile(pattern = "pmc_files_", tmpdir = "/tmp")
  writeLines(pmc_file_list, list_path)
  extract_selected(comm_tar, list_path, pmc_extract_dir)
  unlink(list_path)

  path_lookup <- file.path(pmc_extract_dir, pmc_file_list)
  remaining_files <- pmc_file_list[!file.exists(path_lookup)]

  if (length(remaining_files) > 0 && length(noncomm_tar) > 0 && file.exists(noncomm_tar[1]) && file.size(noncomm_tar[1]) > 0) {
    remaining_path <- tempfile(pattern = "pmc_remaining_", tmpdir = "/tmp")
    writeLines(remaining_files, remaining_path)
    extract_selected(noncomm_tar, remaining_path, pmc_extract_dir)
    unlink(remaining_path)
  }

  available_mask <- file.exists(file.path(pmc_extract_dir, pmc_file_list))
  available_files <- pmc_file_list[available_mask]

  if (length(available_files) == 0) {
    cat(sprintf("  WARNING: No files could be extracted for %s\n", pmc_id))
    unlink(pmc_extract_dir, recursive = TRUE)
    next
  }

  pmc_files <- pmc_files[pmc_files$`Article File` %in% available_files, ]
  if (nrow(pmc_files) == 0) {
    cat(sprintf("  WARNING: No available files remain for %s after filtering\n", pmc_id))
    unlink(pmc_extract_dir, recursive = TRUE)
    next
  }

  BATCH_SIZE <- 500
  n_batches <- ceiling(nrow(pmc_files) / BATCH_SIZE)

  for (batch_idx in seq_len(n_batches)) {
    start_idx <- (batch_idx - 1) * BATCH_SIZE + 1
    end_idx <- min(batch_idx * BATCH_SIZE, nrow(pmc_files))

    batch_files <- pmc_files$`Article File`[start_idx:end_idx]
    batch_paths <- file.path(pmc_extract_dir, batch_files)
    exists_mask <- file.exists(batch_paths)

    if (!all(exists_mask)) {
      missing <- batch_files[!exists_mask]
      if (length(missing) > 0) {
        cat(sprintf("    WARNING: Missing %d files in batch %d for %s\n", length(missing), batch_idx, pmc_id))
      }
      batch_files <- batch_files[exists_mask]
      batch_paths <- batch_paths[exists_mask]
    }

    if (length(batch_files) == 0) {
      next
    }

    batch_results <- foreach(i = seq_along(batch_paths), .errorhandling = 'pass', .packages = c('xml2', 'tidypmc', 'data.table')) %dopar% {
      xml_path <- batch_paths[i]
      article_name <- batch_files[i]

      if (!file.exists(xml_path)) {
        return(NULL)
      }

      pmcid_value <- gsub('\\.xml$', '', basename(article_name))

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

      if (is.null(pmc_data) || nrow(pmc_data) == 0) {
        return(NULL)
      }

      pmc_dt <- as.data.table(pmc_data)
      pmc_dt[, pmcid := pmcid_value]
      pmc_dt
    }

    batch_results <- Filter(function(x) !is.null(x) && is.data.table(x) && nrow(x) > 0, batch_results)

    if (length(batch_results) > 0) {
      batch_data <- rbindlist(batch_results, use.names = TRUE, fill = TRUE)
      fwrite(batch_data, file = csv_output_file, append = written_rows > 0, col.names = written_rows == 0)
      written_rows <- written_rows + nrow(batch_data)
      successful_pmcids <- union(successful_pmcids, unique(batch_data$pmcid))
    }
    if (batch_idx %% 10 == 0) {
      cat(sprintf("  Completed %d/%d batches for %s\n", batch_idx, n_batches, pmc_id))
    }
  }

  unlink(pmc_extract_dir, recursive = TRUE)
}
# Combine batch results (already streamed) and emit summary
cat(sprintf("\n=== CHUNK SUMMARY ===\n"))
cat(sprintf("Total files attempted: %d\n", nrow(chunk_files)))
cat(sprintf("Successfully converted: %d\n", length(successful_pmcids)))
cat(sprintf("Failed conversions: %d\n", nrow(chunk_files) - length(successful_pmcids)))

if (written_rows > 0 && file.exists(csv_output_file)) {
  cat(sprintf("Combined data written to: %s\n", csv_output_file))
  cat(sprintf("Total rows in combined CSV: %d\n", written_rows))
} else {
  warning("No files were successfully converted!")
  empty_df <- data.frame(
    pmcid = character(),
    section = character(),
    paragraph = integer(),
    sentence = integer(),
    text = character()
  )
  fwrite(empty_df, file = csv_output_file, row.names = FALSE)
}

# Clean up parallel cluster
if (!is.null(cl)) {
  stopCluster(cl)
}

