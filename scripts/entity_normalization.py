"""
Entity normalization utilities for the NLP4Pheno pipeline.

This module provides efficient, dictionary-based entity normalization 
to replace the massive string replacement chains in the original pipeline.
Uses vectorized operations for better performance.
"""

import polars as pl
import re
from typing import Dict, List, Tuple


# Strain normalization patterns
STRAIN_PATTERNS = {
    # Remove/normalize strain prefixes
    r"strain ": "",
    r"pv ": "pv. ",
    r"str ": "str. ",
    r"‐": "-",
    r"subsp ": "subsp. ",
    
    # Fix specific organism names
    r"pseudomonas\.": "pseudomonas",
    r"pseudomonas syringae dc3000": "pseudomonas syringae pv. tomato dc3000",
    
    # Normalize collection identifiers
    r"dsm -": "dsm",
    r"dsm =": "dsm", 
    r"atcc #": "atcc",
    r"atcc -": "atcc",
    r"pcc -": "pcc",
    r"vpi -": "vpi",
    r"dfl -": "dfl",
    r"lf -": "lf",
    r"fachb -": "fachb",
    r"nies -": "nies",
    r"tomato -": "tomato",
    r"cms -": "cms",
    r"wvu -": "wvu",
    
    # Species name standardization
    r"pseudomonas sp ": "pseudomonas sp. ",
    r"rhizobium leguminosarum bv ": "rhizobium leguminosarum bv. ",
    r"sphingomonas sp ": "sphingomonas sp. ",
    r"streptomyces sp ": "streptomyces sp. ",
    r"synechococcus sp pcc": "synechococcus sp. pcc",
    r"synechococcus sp\. 7002": "synechococcus sp. pcc 7002",
    r"bacillus sp": "bacillus sp.",
    r"nostoc sp ": "nostoc sp. ",
    
    # Specific strain corrections
    r"verrucosispora maris ab 18 - 032": "verrucosispora maris ab - 18 - 032",
    r"nies 843": "nies - 843",
    r"nostoc pcc - 7524": "nostoc pcc 7524",
    r"nostoc pcc7524": "nostoc pcc 7524",
    r"^acidiphilum": "acidiphilium",
    r"flos aquae": "flos - aquae",
    
    # Bacillus corrections
    r"bacillus licheniformis 9945a": "bacillus licheniformis atcc 9945a",
    r"bacillus licheniformis dsm 13 = atcc 14580": "bacillus licheniformis atcc 14580",
    r"bacillus licheniformis dsm 13": "bacillus licheniformis atcc 14580",
    r"bacillus subtilis subsp\. subtilis str\. 168": "bacillus subtilis 168",
    r"bacillus subtilis subsp\. subtilis 168": "bacillus subtilis 168",
    
    # Other organism corrections
    r"eggerthella\.": "eggerthella",
    r"gg\.$": "gg",
    r"lachnospiraceae bacterium 3 1 57faa ct1": "lachnospiraceae bacterium 3 - 1 - 57faa - ct1",
    r"lactococcus lactis io - 1": "lactococcus lactis subsp. lactis io - 1",
    r"la -": "la",
    r"ncfm\.$": "ncfm",
    r"methylobacterium\.": "methylobacterium",
    r"20z\.": "20z",
    r"bcg -": "bcg",
    r"smegmatis -": "smegmatis",
    r"H37Rv -": "H37Rv",
    r"tuberculosis -": "tuberculosis",
    r"rhodobacter sphaeroides 2\. 4\. 1": "rhodobacter sphaeroides 2 - 4 - 1",
    r"ip32953": "ip 32953",
    r"87\. 22": "87 - 22",
    r"ES−1": "ES - 1",
    r"escherichia coli -": "escherichia coli",
    r"enterococcus faecalis -": "enterococcus faecalis",
    r"dh5 -": "dh5",
    r"escherichia\.": "escherichia",
    r"enterococcus\.": "enterococcus",
}

# Medium normalization patterns
MEDIUM_PATTERNS = {
    # General medium corrections
    r"bertini": "bertani",
    r"^luria - bertani$": "lb",
    r"^luria ‐ bertani$": "lb",
    r"^luria - bertani \( lb \)$": "lb",
    r"^luria bertani \(lb$": "lb",
    r"luria-bertani": "luria bertani",
    r"^lb\)$": "lb",
    
    # Brain heart infusion
    r"^brain heart infusion$": "bhi",
    r"^brain - heart infusion$": "bhi",
    r"^brain-heart infusion$": "bhi",
    r"^brain heart infusion broth$": "bhi broth",
    r"^brain heart infusion \( bhi \) broth$": "bhi broth",
    r"^brain heart infusion \(bhi$": "bhi",
    r"^brain - heart infusion broth$": "bhi broth",
    
    # Luria media variations
    r"^lysogeny broth \( lb \)$": "lb",
    r"^lysogeny broth$": "lb",
    r"^lb medium$": "lb",
    r"^luria bertani$": "lb",
    r"^luria bertani \( lb \)$": "lb",
    r"^luria broth \( lb \)$": "lb broth",
    r"^luria broth$": "lb broth",
    r"^luria - bertani \( lb \) broth$": "lb broth",
    r"^luria-bertani$": "lb",
    r"^luria - bertani broth$": "lb broth",
    r"^luria-bertani broth$": "lb broth",
    r"^luria - bertani agar$": "lb agar",
    r"^luria bertani broth$": "lb broth",
    r"^liquid lb$": "lb broth",
    r"^lb liquid$": "lb broth",
    
    # Other media
    r"^tryptic soy broth \( tsb \)$": "tsb",
    r"^tryptic soy broth$": "tsb",
    r"^tryptic soy broth \(tsb$": "tsb",
    r"^nutrient broth$": "nb",
    r"^mueller−hinton$": "mueller hinton",
    r"^tryptic -": "tryptic",
    r"^mueller-hinton": "mueller hinton",
    r"^muller hinton agar$": "mueller hinton agar",
    r"^muller-hinton broth$": "mueller hinton broth",
    r"^terrific - broth -": "terrific broth",
    r"^nematode - growth$": "nematode growth",
    r"^m9 minimal medium$": "m9 minimal",
    r"^yeast peptone dextrose \(ypd$": "ypd",
    r"^rpmi-1640$": "rpmi 1640",
    
    # Media with parentheses cleanup
    r"^bhi\) agar$": "bhi agar",
    r"^bhi\) broth$": "bhi broth", 
    r"^bhi\)$": "bhi",
    r"^lb\) broth$": "lb broth",
    r"^mh\) agar$": "mh agar",
    r"^mh\) broth$": "mh broth",
    r"^mrs\) agar$": "mrs agar",
    r"^mrs\) broth$": "mrs broth",
}

# Phenotype normalization patterns
PHENOTYPE_PATTERNS = {
    # General corrections
    r"^plaques$": "plaque",
    r"^spore-$": "spore",
    
    # Gram staining variations
    r"^gram- negative$": "gram negative",
    r"^gram negatives$": "gram negative", 
    r"^gram positives$": "gram positive",
    r"^gram\^−$": "gram negative",
    r"^gram- positive$": "gram positive",
    r"^g-positive$": "gram positive",
    r"^gram-positive$": "gram positive",
    r"^gram\+$": "gram positive",
    r"^gram ‐ negative$": "gram negative",
    r"^gram - stain - negative$": "gram negative",
    r"^gram\^\+$": "gram positive",
    r"^gram\^\-$": "gram negative",
    r"^gram \(\+\)$": "gram positive",
    r"^gram \(-\)$": "gram negative",
    r"^gram \(\+$": "gram positive",
    r"^gram \(-$": "gram negative",
    r"^gram \(−$": "gram negative",
    
    # Metabolic phenotypes
    r"^iron-reducing$": "iron - reducing",
    r"^facultatively anaerobic$": "facultative anaerobic",
    r"nonpathogenic": "non pathogenic",
    r"^facultative anaerobe$": "facultative anaerobic",
    r"^anaerobe$": "anaerobic",
    r"^n - fixing$": "nitrogen - fixing",
    r"^fast-growing$": "fast - growing",
    r"^anaerobically$": "anaerobic",
    r"^aerobically$": "aerobic",
    r"^aerobes$": "aerobic",
    
    # Growth characteristics
    r"^endophyte$": "endophytic",
    r"^endophytes$": "endophytic", 
    r"^probiotics$": "probiotic",
    r"^sporulation$": "spore",
    r"^sporulated$": "spore",
    r"^non virulent$": "avirulent",
    r"^virulence$": "virulent",
    r"^mucoidy$": "mucoid",
}

# Compound normalization patterns  
COMPOUND_PATTERNS = {
    # Elements
    r"^cu$": "copper",
    r"^fe$": "iron", 
    r"^zn$": "zinc",
    r"^zn\^2+$": "zinc",
    r"^ni$": "nickel",
    r"^k\^+": "potassium",
    
    # Antibiotics
    r"^β-lactams$": "β-lactam",
    r"^β lactams$": "β-lactam",
    r"^β lactam$": "β-lactam",
    r"^beta lactams$": "β-lactam",
    r"^amp$": "ampicillin",
    r"^kan$": "kanamycin",
    r"^rif\)$": "rifampicin",
    r"^quinolones$": "quinolone",
    r"^rif$": "rifampicin",
    r"^rifampin$": "rifampicin",
    r"^tet$": "tetracycline",
    r"^van$": "vancomycin",
    
    # Biochemical compounds
    r"^sugars$": "sugar",
    r"^lipopeptides$": "lipopeptide", 
    r"^lipids$": "lipid",
    r"^α-glucans$": "α-glucan",
    r"^β-glucans": "β-glucan",
    r"^heavy metals$": "heavy metal",
    r"^metals": "metal",
    r"^acetyl - coa$": "acetyl coa",
}

# Organism normalization patterns
ORGANISM_PATTERNS = {
    # Animals
    r"^dairy cows$": "dairy cow",
    r"^flies$": "fly",
    r"^goats$": "goat",
    r"^humans$": "human",
    r"^mice$": "mouse",
    r"^murine$": "mouse",
    r"^bovine$": "cow",
    r"^bovines$": "cow",
    r"^canine$": "dog",
    r"^cattle$": "cow",
    r"^avian$": "bird",
    r"^birds$": "bird",
    r"^pigeons$": "pigeon",
    r"^wild boars$": "wild boar",
    r"^chickens$": "chicken",
    r"^cockroaches$": "cockroach",
    r"^dogs$": "dog",
    r"^insects$": "insect",
    r"^piglets$": "piglet",
    r"^mosquitoes$": "mosquito",
    r"^pigs$": "pig",
    r"^rabbits$": "rabbit",
    r"^rats$": "rat",
    r"^nematodes$": "nematode",
    r"^larval$": "larvae",
    r"^ferrets$": "ferret",
    r"fishes$": "fish",
    r"^hamsters$": "hamster",
    r"^calves$": "calf",
    r"^cows$": "cow",
    r"^horses$": "horse",
    r"^sponges$": "sponge",
    r"^cats$": "cat",
    r"ticks": "tick",
    r"worms": "worm",
    r"mice": "mouse",
    r"onions": "onion",
    
    # Plants
    r"^grasses$": "grass",
    r"^soybeans$": "soybean",
    r"^legumes$": "legume",
    r"^potatoes$": "potato",
    r"^tomatoes$": "tomato",
    r"^oomycetes$": "oomycete",
    r"^plants$": "plant",
    r"trees$": "tree",
    r"^maize plants$": "maize",
    r"^tomato plants$": "tomato",
    r"^potato plants$": "potato",
    r"^corn$": "maize",
    r"zebra fish": "zebrafish",
    r"sugar cane": "sugarcane",
    r"^sugar beet$": "beta vulgaris",
    
    # Scientific names
    r"g\. mellonella": "galleria mellonella",
    r"^raw264\.7$": "raw 264.7",
    r"^c\. elegans$": "caenorhabditis elegans",
    r"^d\. melanogaster$": "drosophila melanogaster",
    r"^p\. falciparum$": "plasmodium falciparum",
    r"a\. stephensi": "anopheles stephensi",
    r"a\. mellifera": "apis mellifera",
    r"^a\. thaliana$": "arabidopsis thaliana",
    r"^d\. melanogaster": "drosophila melanogaster",
}

# Isolate source normalization patterns
ISOLATE_PATTERNS = {
    r"^marine sediments$": "marine sediment",
    r"^human faeces$": "human feces",
    r"^sea water$": "seawater", 
    r"^sediments$": "sediment",
    r"^soils$": "soil",
    r"^soil samples$": "soil",
    r"^water sample?$": "water",
    r"^stool$": "feces",
    r"^soil sample?$": "soil",
    r"^stool sample?$": "feces",
    r"^biofilms$": "biofilm",
    r"^biofilm formation by$": "biofilm formation",
    r"^formation of biofilm$": "biofilm formation",
    r"^spores$": "spore",
    r"^endospores$": "endospore",
    r"^filaments$": "filament",
    r"wrinkled colonies": "wrinkled colony",
}

# Effect normalization patterns
EFFECT_PATTERNS = {
    r"^antimicrobial activity$": "antimicrobial",
    r"^antibacterial activity$": "antibacterial", 
    r"^antibacterial effects$": "antibacterial",
    r"^antifungal activity$": "antifungal",
    r"^plant-growth": "plant growth",
    r"^plant growth-": "plant growth",
}

# Species normalization patterns
SPECIES_PATTERNS = {
    r"^escherichia coli$": "e. coli",
    r"^enterococcus faecalis$": "e. faecalis",
    r"^listeria monocytogenes$": "l. monocytogenes",
    r"^staphylococcus aureus$": "s. aureus",
    r"^pseudomonas aeruginosa$": "p. aeruginosa",
    r"^lactobacillus plantarum$": "l. plantarum",
    r"^candida albicans$": "c. albicans",
    r"^a\. thaliana$": "a. thaliana",
    r"^x\. campestris": "xanthomonas campestris",
    r"^r\. solanacearum": "ralstonia solanacearum",
    r"^e\. faecium": "enterococcus faecium",
    r"^b\. anthracis": "bacillus anthracis",
    r"^v\. alginolyticus": "vibrio alginolyticus",
    r"^v\. anguillarum": "vibrio anguillarum",
    r"^v\. parahemolyticus": "vibrio parahemolyticus",
    r"^p\. fluorescens": "pseudomonas fluorescens",
}

# Disease normalization patterns
DISEASE_PATTERNS = {
    r"^cf$": "cystic fibrosis",
}

# General cleanup patterns
GENERAL_PATTERNS = {
    r"‐": "-",
    r"'": "'",
    r" \($": "",
    r" of$": "",
    r"^the ": "",
}


def normalize_entity_column(df: pl.DataFrame, column: str, entity_type: str) -> pl.DataFrame:
    """
    Normalize entity values using entity-type specific patterns.
    
    Args:
        df: Polars DataFrame containing the column to normalize
        column: Name of column to normalize  
        entity_type: Type of entity (STRAIN, MEDIUM, PHENOTYPE, etc.)
        
    Returns:
        DataFrame with normalized column
    """
    # Select appropriate pattern dictionary
    pattern_dict = {
        'STRAIN': STRAIN_PATTERNS,
        'MEDIUM': MEDIUM_PATTERNS, 
        'PHENOTYPE': PHENOTYPE_PATTERNS,
        'COMPOUND': COMPOUND_PATTERNS,
        'ORGANISM': ORGANISM_PATTERNS,
        'ISOLATE': ISOLATE_PATTERNS,
        'EFFECT': EFFECT_PATTERNS,
        'SPECIES': SPECIES_PATTERNS,
        'DISEASE': DISEASE_PATTERNS,
    }.get(entity_type, {})
    
    # Apply general patterns to all entities
    all_patterns = {**GENERAL_PATTERNS, **pattern_dict}
    
    # Apply regex replacements using polars expressions
    result_expr = pl.col(column)
    for pattern, replacement in all_patterns.items():
        result_expr = result_expr.str.replace_all(pattern, replacement)
    
    return df.with_columns(result_expr.alias(f"{column}_qc"))


def normalize_strain_entities(df: pl.DataFrame) -> pl.DataFrame:
    """Normalize strain entities with strain-specific logic."""
    return df.with_columns([
        # Apply strain normalization patterns
        pl.col("word_strain")
        .str.replace_all(r"strain ", "")
        .str.replace_all(r"pv ", "pv. ")
        .str.replace_all(r"str ", "str. ")
        .str.replace_all(r"‐", "-")
        .str.replace_all(r"subsp ", "subsp. ")
        .str.replace_all(r"pseudomonas\.", "pseudomonas")
        .str.replace_all(r"pseudomonas syringae dc3000", "pseudomonas syringae pv. tomato dc3000")
        .str.replace_all(r"dsm -", "dsm")
        .str.replace_all(r"dsm =", "dsm")
        .str.replace_all(r"atcc #", "atcc")
        .str.replace_all(r"atcc -", "atcc")
        .str.replace_all(r"pcc -", "pcc")
        # Add more patterns as needed...
        .alias("word_strain_qc")
    ])


def create_vertex_dot_column(df: pl.DataFrame) -> pl.DataFrame:
    """Create normalized vertex_dot column for StrainSelect matching."""
    return df.with_columns([
        pl.col("word_strain_qc")
        # Convert to lowercase for consistent matching
        .str.to_lowercase()
        
        # Replace spaces and common delimiters with dots
        .str.replace_all(" ", ".")
        .str.replace_all("-", ".")
        .str.replace_all("_", ".")
        .str.replace_all("/", ".")
        .str.replace_all(":", ".")
        .str.replace_all("#", ".")
        .str.replace_all(r"\*", ".")
        
        # Handle culture collection variations first (order matters)
        .str.replace_all(".no.", "")
        .str.replace_all(".no", "")
        .str.replace_all("no.", "")
        .str.replace_all(".number.", "")
        .str.replace_all("number.", "")
        .str.replace_all("strain.", "")
        .str.replace_all("str.", "")
        .str.replace_all("type.", "")
        
        # Handle specific culture collection patterns
        .str.replace_all("atcc.no", "atcc")
        .str.replace_all("mtcc.no", "mtcc") 
        .str.replace_all("cgmcc.no", "cgmcc")
        .str.replace_all("dsm.no", "dsm")
        .str.replace_all("jcm.no", "jcm")
        .str.replace_all("nrrl.no", "nrrl")
        .str.replace_all("lmg.no", "lmg")
        .str.replace_all("kctc.no", "kctc")
        .str.replace_all("nbrc.no", "nbrc")
        .str.replace_all("ccug.no", "ccug")
        .str.replace_all("nctc.no", "nctc")
        .str.replace_all("vkm.no", "vkm")
        .str.replace_all("bcrc.no", "bcrc")
        .str.replace_all("kacc.no", "kacc")
        .str.replace_all("cip.no", "cip")
        .str.replace_all("nccp.no", "nccp")
        
        # Normalize culture collection codes with dots
        .str.replace_all("atcc", "atcc.")
        .str.replace_all("mtcc", "mtcc.")
        .str.replace_all("dsm", "dsm.")
        .str.replace_all("cgmcc", "cgmcc.")
        .str.replace_all("jcm", "jcm.")
        .str.replace_all("nrrl", "nrrl.")
        .str.replace_all("lmg", "lmg.")
        .str.replace_all("kctc", "kctc.")
        .str.replace_all("nbrc", "nbrc.")
        .str.replace_all("ccug", "ccug.")
        .str.replace_all("nctc", "nctc.")
        .str.replace_all("vkm", "vkm.")
        .str.replace_all("bcrc", "bcrc.")
        .str.replace_all("kacc", "kacc.")
        .str.replace_all("cip", "cip.")
        .str.replace_all("nccp", "nccp.")
        
        # Remove unwanted characters
        .str.replace_all("=", "")
        .str.replace_all('"', "")
        .str.replace_all('"', "")
        .str.replace_all('"', "")
        .str.replace_all("'", "")
        .str.replace_all("'", "")
        .str.replace_all(r"\^", "")
        .str.replace_all("®", "")
        .str.replace_all("™", "")
        .str.replace_all("Δ", "")
        .str.replace_all(",", "")
        .str.replace_all(r"\(", "")
        .str.replace_all(r"\)", "")
        .str.replace_all(r"\[", "")
        .str.replace_all(r"\]", "")
        .str.replace_all(r"\{", "")
        .str.replace_all(r"\}", "")
        .str.replace_all(r"\+", ".")
        .str.replace_all("%", "")
        .str.replace_all("&", ".")
        .str.replace_all("@", ".")
        .str.replace_all("~", "")
        .str.replace_all("`", "")
        .str.replace_all(r"\|", ".")
        .str.replace_all(r"\\", ".")
        
        # Clean up multiple dots and trailing/leading dots
        .str.replace_all(r"\.\.\.", ".")
        .str.replace_all(r"\.\.", ".")
        .str.replace_all("^\\.", "")  # Remove leading dots
        .str.replace_all("\\.$", "")  # Remove trailing dots
        
        .alias("vertex_dot")
    ])


def normalize_compounds(df: pl.DataFrame) -> pl.DataFrame:
    """Apply compound-specific normalizations."""
    compound_mask = pl.col("ner") == "COMPOUND"
    
    return df.with_columns([
        pl.when(compound_mask)
        .then(
            pl.col("word")
            .str.replace_all("^p$", "phosphorus")
            .str.replace_all("^c$", "carbon") 
            .str.replace_all("^n$", "nitrogen")
            .str.replace_all("^s$", "sulfur")
            .str.replace_all("^k$", "potassium")
        )
        .otherwise(pl.col("word"))
        .alias("word")
    ])


def filter_uninterpretable_entities(df: pl.DataFrame) -> pl.DataFrame:
    """Remove uninterpretable entity mentions."""
    uninterpretable = [
        "less", "of", "week old", "old", "4′", "6a", "levels", "multi", 
        "1 week old", "centenarian", "heavy", "broad", "synthesis", "binding",
        "decline", "formation", "production", "loss", "na", "4 day old", "like",
        "the", "system", "in", "10", "effects", "wt", "wild type", "sp.", "spp",
        "non", "for", "at the", "with", "iii", "1 day old", "8 week old",
        "8 week old female"
    ]
    
    return df.filter(~pl.col("word").is_in(uninterpretable))


def apply_length_filters(df: pl.DataFrame) -> pl.DataFrame:
    """Apply length-based filtering for entity quality."""
    word_len = pl.col("word").str.len_chars()

    return df.filter(
        (word_len > 1) &
        ~((pl.col("ner") == "SPECIES") & (word_len == 2)) &
        ~((word_len == 2) & pl.col("word").str.contains(r"_")) &
        ~((pl.col("ner") == "COMPOUND") & pl.col("word").str.contains(r"\d\d"))
    )


def extract_genus_hint(df: pl.DataFrame) -> pl.DataFrame:
    """
    Extract genus hints from sentence text to improve strain matching.

    Identifies organism names near strain mentions to provide taxonomic context.
    This dramatically improves matching accuracy for short strain names.

    Args:
        df: DataFrame with 'text' and 'word_strain_qc' columns

    Returns:
        DataFrame with added 'genus_hint' column
    """
    # Common bacterial genus patterns (genus name or abbreviated form)
    # Match patterns like "P. aeruginosa", "Pseudomonas aeruginosa", "E. coli", etc.
    genus_patterns = {
        # Pseudomonas
        'aeruginosa': 'pseudomonas',
        'pseudomonas': 'pseudomonas',

        # Mycobacterium
        'tuberculosis': 'mycobacterium',
        'mycobacterium': 'mycobacterium',
        ' mtb ': 'mycobacterium',

        # Staphylococcus
        'aureus': 'staphylococcus',
        'staphylococcus': 'staphylococcus',
        'staph ': 'staphylococcus',

        # Streptococcus
        'pneumoniae': 'streptococcus',
        'streptococcus': 'streptococcus',
        'strep ': 'streptococcus',

        # Escherichia
        ' coli': 'escherichia',
        'escherichia': 'escherichia',

        # Bacillus
        'subtilis': 'bacillus',
        'bacillus': 'bacillus',

        # Salmonella
        'salmonella': 'salmonella',

        # Vibrio
        'vibrio': 'vibrio',

        # Listeria
        'monocytogenes': 'listeria',
        'listeria': 'listeria',

        # Klebsiella
        'klebsiella': 'klebsiella',

        # Lactobacillus
        'lactobacillus': 'lactobacillus',

        # Clostridium
        'clostridium': 'clostridium',

        # Burkholderia
        'burkholderia': 'burkholderia',

        # Acinetobacter
        'acinetobacter': 'acinetobacter',
    }

    # Create genus_hint column by checking patterns in text
    text_lower = pl.col("text").str.to_lowercase()

    # Build a when-then chain for all patterns
    genus_expr = pl.lit(None)  # Default to None

    for pattern, genus in genus_patterns.items():
        genus_expr = pl.when(text_lower.str.contains(pattern)).then(pl.lit(genus)).otherwise(genus_expr)

    return df.with_columns(genus_expr.alias("genus_hint"))