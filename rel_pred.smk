import pandas as pd
import itertools
from rapidfuzz import process
from rapidfuzz import fuzz
from scipy import sparse
from tqdm.auto import tqdm
import dask.array as da
from scipy.sparse import csr_matrix
import numpy as np
from collections import defaultdict
from glob import glob

configfile: "config.yaml"

cutoff = config["cutoff_prediction"]
output_path = config["output_path"] # /home/tu/tu_tu/tu_bbpgo01/link

preds = f"{output_path}/preds" + str(config["dataset"])
labels = config["rel_labels"]
cuda = config["cuda_devices"]


rule all:
    input:
        f"{preds}/REL_output/strains_assemblies.txt",
        f"{preds}/network.tsv",
        f"{preds}/REL_output/preds.pqt",


rule format_sentences:
    input:
        f"{preds}/NER_output/preds.parquet",
    output:
        f"{preds}/NER_output/ner_preds.parquet",
    resources:
        slurm_partition="single",
        runtime=100,
        mem_mb=30000
    run:
        df = pd.read_parquet(input[0])
        df.insert(0, "formatted_text", "")
        for i in df.iterrows():
            text = i[1]["text"]
            ss = i[1]["start_strain"]
            es = i[1]["end_strain"]

            ner = i[1]["ner"]
            sn = int(i[1]["start"])
            en = int(i[1]["end"])
            # do the latest substitution first
            if ss > sn:
                if es > en:
                    stext = text[:ss] + "@STRAIN$" + text[es:]
                    ntext = stext[:sn] + "@" + ner + "$" + stext[en:]
                    df.loc[i[0], "formatted_text"] = ntext
                else:
                    continue
            else:
                if en > es:
                    ntext = text[:sn] + "@" + ner + "$" + text[en:]
                    stext = ntext[:ss] + "@STRAIN$" + ntext[es:]
                    df.loc[i[0], "formatted_text"] = stext
                else:
                    continue
        df.to_parquet(output[0])


rule make_device_file:
    output:
        f"{preds}/REL_output/device_models.txt",
    resources:
        slurm_partition="single",
        runtime=30,
        mem_mb=5000,
    run:
        dev = [str(x) for x in cuda]
        models = [x + " " + y for x, y in zip(itertools.cycle(dev), labels)]
        with open(output[0], "w") as f:
            for i in models:
                f.write(f"{i}\n")


rule run_all_models:
    input:
        f"{preds}/NER_output/ner_preds.parquet",
        f"{preds}/REL_output/device_models.txt",
    output:
        preds + "/REL_output/{l}.parquet",
    conda:
        "torch"
    resources:
        slurm_partition="gpu_4",
        slurm_extra="--gres=gpu:1",
        runtime=500
    shell:
        """
        while read -r d m; do
            if [ "$m" = "{wildcards.l}" ]; then
                python scripts/rel_prediction.py --model $m --device $d --output {preds}/REL_output/$m.parquet --input {input[0]} 
            fi
            done < {input[1]}
        """


rule merge_preds:
    input:
        expand(preds + "/REL_output/{l}.parquet", l=labels),
    output:
        f"{preds}/REL_output/preds.pqt",
    resources:
        slurm_partition="single",
        runtime=1000,
        mem_mb=20000
    run:
        l = []
        for i in input:
            l.append(pd.read_parquet(i))
        df = pd.concat(l)

        d = pd.concat(
            [
                df.drop(columns="re_result"),
                df.re_result.apply(pd.Series).add_suffix("_rel"),
            ],
            axis=1,
        )

        d.loc[:, "word_strain_qc"] = (
            d.word_strain.str.replace("strain ", "", regex=True)
            .str.replace("pv ", "pv. ", regex=True)
            .str.replace("str ", "str. ", regex=True)
            .str.replace("‐", "-", regex=True)
            .str.replace("subsp ", "subsp. ", regex=True)
            .str.replace("pseudomonas.", "pseudomonas")
            .str.replace(
                "pseudomonas syringae dc3000", "pseudomonas syringae pv. tomato dc3000"
            )
            .str.replace("dsm -", "dsm", regex=True)
            .str.replace("dsm =", "dsm", regex=True)
            .str.replace("atcc #", "atcc", regex=True)
            .str.replace("atcc -", "atcc", regex=True)
            .str.replace("pcc -", "pcc", regex=True)
            .str.replace("vpi -", "vpi", regex=True)
            .str.replace("dfl -", "dfl", regex=True)
            .str.replace("lf -", "lf", regex=True)
            .str.replace("fachb -", "fachb", regex=True)
            .str.replace("nies -", "nies", regex=True)
            .str.replace("tomato -", "tomato", regex=True)
            .str.replace("cms -", "cms", regex=True)
            .str.replace("wvu -", "wvu", regex=True)
            .str.replace("pseudomonas sp ", "pseudomonas sp. ", regex=True)
            .str.replace(
                "rhizobium leguminosarum bv ",
                    "rhizobium leguminosarum bv. ",
                    regex=True,
                )
                .str.replace("sphingomonas sp ", "sphingomonas sp. ", regex=True)
                .str.replace("streptomyces sp ", "streptomyces sp. ", regex=True)
                .str.replace("synechococcus sp pcc", "synechococcus sp. pcc", regex=True)
                .str.replace("synechococcus sp. 7002", "synechococcus sp. pcc 7002")
                .str.replace("bacillus sp", "bacillus sp.", regex=True)
                .str.replace(
                    "verrucosispora maris ab 18 - 032",
                    "verrucosispora maris ab - 18 - 032",
                    regex=True,
                )
                .str.replace("nies 843", "nies - 843", regex=True)
                .str.replace("nostoc pcc - 7524", "nostoc pcc 7524", regex=True)
                .str.replace("nostoc pcc7524", "nostoc pcc 7524", regex=True)
                .str.replace("nostoc sp ", "nostoc sp. ", regex=True)
                .str.replace("^acidiphilum", "acidiphilium", regex=True)
                .str.replace("flos aquae", "flos - aquae", regex=True)
                .str.replace(
                    "bacillus licheniformis 9945a",
                    "bacillus licheniformis atcc 9945a",
                    regex=True,
                )
                .str.replace(
                    "bacillus licheniformis dsm 13 = atcc 14580",
                    "bacillus licheniformis atcc 14580",
                    regex=True,
                )
                .str.replace(
                    "bacillus licheniformis dsm 13",
                    "bacillus licheniformis atcc 14580",
                    regex=True,
                )
                .str.replace(
                "bacillus subtilis subsp. subtilis str. 168", "bacillus subtilis 168"
            )
            .str.replace(
                "bacillus subtilis subsp. subtilis 168", "bacillus subtilis 168"
            )
            .str.replace("eggerthella.", "eggerthella")
            .str.replace("gg\.$", "gg", regex=True)
            .str.replace(
                "lachnospiraceae bacterium 3 1 57faa ct1",
                    "lachnospiraceae bacterium 3 - 1 - 57faa - ct1",
                )
                .str.replace(
                "lactococcus lactis io - 1", "lactococcus lactis subsp. lactis io - 1"
            )
            .str.replace("la -", "la", regex=True)
            .str.replace("ncfm\.$", "ncfm", regex=True)
            .str.replace("methylobacterium.", "methylobacterium")
            .str.replace("20z.", "20z")
            .str.replace("bcg -", "bcg")
            .str.replace("smegmatis -", "smegmatis")
            .str.replace("H37Rv -", "H37Rv")
            .str.replace("tuberculosis -", "tuberculosis")
            .str.replace(
                "rhodobacter sphaeroides 2. 4. 1", "rhodobacter sphaeroides 2 - 4 - 1"
            )
            .str.replace("ip32953", "ip 32953")
            .str.replace("87. 22", "87 - 22")
            .str.replace("ES−1", "ES - 1")
            .str.replace("escherichia coli -", "escherichia coli")
            .str.replace("enterococcus faecalis -", "enterococcus faecalis")
            .str.replace("dh5 -", "dh5")
            .str.replace("escherichia.", "escherichia")
            .str.replace("enterococcus.", "enterococcus")
        )

        d.loc[:, "word_qc"] = (
            # general
            d.word.str.replace("‐", "-", regex=True)
            .str.replace("’", "'", regex=True)
            # medium
            .str.replace("bertini", "bertani", regex=True)
            .str.replace("^luria - bertani$", "lb", regex=True)
            .str.replace("^luria ‐ bertani$", "lb", regex=True)
            .str.replace("^luria - bertani \( lb \)$", "lb", regex=True)
            .str.replace("^brain heart infusion$", "bhi", regex=True)
            .str.replace("^brain - heart infusion$", "bhi", regex=True)
            .str.replace("^brain-heart infusion$", "bhi", regex=True)
            .str.replace("^brain heart infusion broth$", "bhi broth", regex=True)
            .str.replace(
                "^brain heart infusion \( bhi \) broth$", "bhi broth", regex=True
            )
            .str.replace("^brain - heart infusion broth$", "bhi broth", regex=True)
            .str.replace("^lysogeny broth \( lb \)$", "lb", regex=True)
            .str.replace("^lysogeny broth$", "lb", regex=True)
            .str.replace("^lb medium$", "lb", regex=True)
            .str.replace("^luria bertani$", "lb", regex=True)
            .str.replace("^luria bertani \( lb \)$", "lb", regex=True)
            .str.replace("^luria broth \( lb \)$", "lb broth", regex=True)
            .str.replace("^luria broth$", "lb broth", regex=True)
            .str.replace("^luria - bertani \( lb \) broth$", "lb broth", regex=True)
            .str.replace("^luria-bertani$", "lb", regex=True)
            .str.replace("^luria - bertani broth$", "lb broth", regex=True)
            .str.replace("^luria-bertani broth$", "lb broth", regex=True)
            .str.replace("^luria - bertani agar$", "lb agar", regex=True)
            .str.replace("^luria bertani broth$", "lb broth", regex=True)
            .str.replace("^tryptic soy broth ( tsb )$", "tsb", regex=True)
            .str.replace("^tryptic soy broth$", "tsb", regex=True)
            .str.replace("^liquid lb$", "lb broth", regex=True)
            .str.replace("^lb liquid$", "lb broth", regex=True)
            .str.replace("^muller hinton agar$", "mueller hinton agar", regex=True)
            .str.replace("^tryptic soy broth$", "tsb", regex=True)
            .str.replace("^tryptic soy broth \( tsb \)$", "tsb", regex=True)
            .str.replace("^nutrient broth$", "nb", regex=True)
            .str.replace("^mueller−hinton$", "mueller hinton", regex=True)
            .str.replace("^tryptic -", "tryptic", regex=True)
            .str.replace("^mueller-hinton", "mueller hinton", regex=True)
            .str.replace("^terrific - broth -", "terrific broth", regex=True)
            .str.replace("^nematode - growth$", "nematode growth", regex=True)
            .str.replace("^bhi\) agar$", "bhi agar", regex=True)
            .str.replace("^bhi\) broth$", "bhi broth", regex=True)
            .str.replace("^lb\) broth$", "lb broth", regex=True)
            .str.replace("^mh\) agar$", "mh agar", regex=True)
            .str.replace("^mh\) broth$", "mh broth", regex=True)
            .str.replace("^mrs\) agar$", "mrs agar", regex=True)
            .str.replace("^mrs\) broth$", "mrs broth", regex=True)
            .str.replace("^rpmi-1640$", "rpmi 1640", regex=True)
            .str.replace("^muller-hinton broth$", "mueller hinton broth", regex=True)
            .str.replace("^m9 minimal medium$", "m9 minimal", regex=True)
            # metabolite
            .str.replace("^acetyl - coa$", "acetyl coa", regex=True)
            # phenotype
            .str.replace("^gram- negative$", "gram negative", regex=True)
            .str.replace("^gram- positive$", "gram positive", regex=True)
            .str.replace("^gram ‐ negative$", "gram negative", regex=True)
            .str.replace("^gram - stain - negative$", "gram negative", regex=True)
            .str.replace("^gram\^\+$", "gram positive", regex=True)
            .str.replace("^gram\^\-$", "gram negative", regex=True)
            .str.replace("^gram (+)$", "gram positive", regex=True)
            .str.replace("^gram (-)$", "gram negative", regex=True)
            .str.replace("^gram (+$", "gram positive", regex=True)
            .str.replace("^gram (-$", "gram negative", regex=True)
            .str.replace("^iron-reducing$", "iron - reducing", regex=True)
            .str.replace(
                "^facultatively anaerobic$", "facultative anaerobic", regex=True
            )
            .str.replace("nonpathogenic", "non pathogenic", regex=True)
            .str.replace("^facultative anaerobe$", "facultative anaerobic", regex=True)
            .str.replace("^anaerobe$", "anaerobic", regex=True)
            .str.replace("^n - fixing$", "nitrogen - fixing", regex=True)
            .str.replace("^fast-growing$", "fast - growing", regex=True)
            .str.replace("^anaerobically$", "anaerobic", regex=True)
            # isolate
            .str.replace("^marine sediments$", "marine sediment", regex=True)
            .str.replace("^human faeces$", "human feces", regex=True)
            .str.replace("^sea water$", "seawater", regex=True)
            .str.replace("^sediments$", "sediment", regex=True)
            .str.replace("^soils$", "soil", regex=True)
            .str.replace("^soil samples$", "soil", regex=True)
            .str.replace("^water sample?$", "water", regex=True)
            .str.replace("^stool$", "feces", regex=True)
            .str.replace("^soil sample?$", "soil", regex=True)
            .str.replace("^stool sample?$", "feces", regex=True)
            .str.replace("^biofilms$", "biofilm", regex=True)
            .str.replace("^spores$", "spore", regex=True)
            .str.replace("^endospores$", "endospore", regex=True)
            .str.replace("^filaments$", "filament", regex=True)
            .str.replace("wrinkled colonies", "wrinkled colony", regex=True)
            # compound
            .str.replace("^heavy metals$", "heavy metal", regex=True)
            .str.replace("^cu$", "copper", regex=True)
            .str.replace("^metals", "metal", regex=True)
            .str.replace("^zn$", "zinc", regex=True)
            .str.replace("^ni$", "nickel", regex=True)
            .str.replace("^β-lactams$", "β-lactam", regex=True)
            .str.replace("^rif$","rifampicin",regex=True)
            .str.replace("^rifampin$","rifampicin",regex=True)
            .str.replace("^sugars$", "sugar", regex=True)
            .str.replace("^lipopeptides$", "lipopeptide", regex=True)
            .str.replace("^lipids$", "lipid", regex=True)
            .str.replace("^α-glucans$", "α-glucan", regex=True)
            .str.replace("^β-glucans", "β-glucan", regex=True)
            # organism
            .str.replace("^humans$", "human", regex=True)
            .str.replace("^mice$", "mouse", regex=True)
            .str.replace("^birds$", "bird", regex=True)
            .str.replace("^soybeans$", "soybean", regex=True)
            .str.replace("^wild boars$", "wild boar", regex=True)
            .str.replace("^chickens$", "chicken", regex=True)
            .str.replace("^cockroaches$", "cockroach", regex=True)
            .str.replace("^dogs$", "dog", regex=True)
            .str.replace("^insects$", "insect", regex=True)
            .str.replace("^legumes$", "legume", regex=True)
            .str.replace("^piglets$", "piglet", regex=True)
            .str.replace("^mosquitoes$", "mosquito", regex=True)
            .str.replace("^potatoes$", "potato", regex=True)
            .str.replace("^tomatoes$", "tomato", regex=True)
            .str.replace("^pigs$", "pig", regex=True)
            .str.replace("^rabbits$", "rabbit", regex=True)
            .str.replace("^plants$", "plant", regex=True)
            .str.replace("g. mellonella", "galleria mellonella", regex=True)
            .str.replace("^raw264.7$", "raw 264.7", regex=True)
            .str.replace("^c. elegans$", "caenorhabditis elegans", regex=True)
            .str.replace("^rats$", "rat", regex=True)
            .str.replace("trees$","tree",regex=True)
            .str.replace("^nematodes$", "nematode", regex=True)
            .str.replace("^larval$", "larvae", regex=True)
            .str.replace("^ferrets$", "ferret", regex=True)
            .str.replace("^maize plants$", "maize", regex=True)
            .str.replace("^tomato plants$", "tomato", regex=True)
            .str.replace("^potato plants$", "potato", regex=True)
            .str.replace("^d. melanogaster$", "drosophila melanogaster", regex=True)
            .str.replace("^p. falciparum$", "plasmodium falciparum", regex=True)
            .str.replace("^zebra fish$", "zebrafish", regex=True)
            .str.replace("a. stephensi", "anopheles stephensi", regex=True)
            .str.replace("^hamsters$", "hamster", regex=True)
            .str.replace("^calves$", "calf", regex=True)
            .str.replace("^cows$", "cow", regex=True)
            .str.replace("^horses$", "horse", regex=True)
            .str.replace("^sponges$", "sponge", regex=True)
            .str.replace("^cats$", "cat", regex=True)
            .str.replace("^corn$", "maize", regex=True)
            .str.replace("a. mellifera", "apis mellifera", regex=True)
            .str.replace("ticks","tick",regex=True)
            .str.replace("worms","worm",regex=True)
            .str.replace("mice", "mouse", regex=True)
            .str.replace("zebra fish", "zebrafish", regex=True)
            .str.replace("sugar cane", "sugarcane", regex=True)
            .str.replace("^a. thaliana$", "arabidopsis thaliana", regex=True)
            .str.replace("^sugar beet$", "beta vulgaris", regex=True)
            .str.replace("^d. melanogaster", "drosophila melanogaster", regex=True)
            .str.replace("onions", "onion", regex=True)
            # effect
            .str.replace("^antimicrobial activity$", "antimicrobial", regex=True)
            .str.replace("^antibacterial activity$", "antibacterial", regex=True)
            .str.replace("^antibacterial effects$", "antibacterial", regex=True)
            .str.replace("^antifungal activity$", "antifungal", regex=True)
            .str.replace("^plant-growth", "plant growth", regex=True)
            .str.replace("^plant growth-", "plant growth", regex=True)
            #species
            .str.replace("^escherichia coli$", "e. coli", regex=True)
            .str.replace("^enterococcus faecalis$", "e. faecalis", regex=True)
            .str.replace("^listeria monocytogenes$", "l. monocytogenes", regex=True)
            .str.replace("^staphylococcus aureus$", "s. aureus", regex=True)
            .str.replace("^pseudomonas aeruginosa$", "p. aeruginosa", regex=True)
            .str.replace("^lactobacillus plantarum$", "l. plantarum", regex=True)
            .str.replace("^candida albicans$", "c. albicans", regex=True)
            .str.replace("^a. thaliana$", "a. thaliana", regex=True)
            #general
            .str.replace(" of$", "", regex=True)
            .str.replace("^the ", "", regex=True)
            #disease
            .str.replace("^cf$", "cystic fibrosis", regex=True)
        )
        d[d["score_rel"] > cutoff].to_parquet(output[0])


rule download_strainselect:
    output:
        f"{preds}/strainselect/StrainSelect21_edges.tab.txt",
        f"{preds}/strainselect/StrainSelect21_vertices.tab.txt",
    resources:
        slurm_partition="single",
        runtime=30,
    shell:
        "wget https://gg-sg-web.s3-us-west-2.amazonaws.com/downloads/strainselect_database/StrainSelect21/StrainSelect21_edges.tab.txt -O {output[0]}; wget https://gg-sg-web.s3-us-west-2.amazonaws.com/downloads/strainselect_database/StrainSelect21/StrainSelect21_vertices.tab.txt -O {output[1]}"

rule split_batches_strainselect:
    input:
        preds_file=f"{preds}/REL_output/preds.pqt",
    output:
        batch_files=expand(f"{preds}/REL_output/batched_input/{{batch_id}}.pqt", batch_id=range(0, 500)),
    resources:
        slurm_partition="single",
        runtime=100,
        mem_mb=10000,
        tasks=2
    run:
        df = pd.read_parquet(input[0])
        df["vertex_dot"] = df.word_strain_qc.str.replace(" ",".").str.replace(".number.","").str.replace("atcc.no","atcc").str.replace("mtcc.no","mtcc").str.replace("cgmcc.no","cgmcc").str.replace("dsm.no","dsm").str.replace("-",".").str.replace("atcc","atcc.").str.replace("mtcc","mtcc.").str.replace("dsm","dsm.").str.replace("cgmcc","cgmcc.").str.replace("=","").str.replace("“","").str.replace("”","").str.replace('"',"").str.replace("#",".").str.replace("*",".").str.replace("^","").str.replace(":",".").str.replace("®","").str.replace("™","").str.replace("’","").str.replace("‘","").str.replace("(","").str.replace(")","").str.replace(",","").str.replace("/",".").str.replace("_",".").str.replace("Δ","").str.replace("cip","cip.").str.replace("nccp","nccp.").str.replace("...",".").str.replace("..",".")
        df = df[~df['vertex_dot'].str.match("^[a-z]\.[a-z]+$")]
        num_batches = 500
        batched_df = np.array_split(df, num_batches)
        os.makedirs(f"{preds}/REL_output/batched_input/", exist_ok=True)
        for i, batch_df in enumerate(batched_df):
            batch_df.to_parquet(f"{preds}/REL_output/batched_input/{i}.pqt")

rule match_batch_strainselect:
    input:
        batch_file=f"{preds}/REL_output/batched_input/{{batch_id}}.pqt",
        vertices=f"{preds}/strainselect/StrainSelect21_vertices.tab.txt",
    output:
        batch_output=f"{preds}/batched_output_results/{{batch_id}}.parquet",
    resources:
        slurm_partition="single",
        runtime=200,
        mem_mb=100000,
        tasks=3
    run:
        workers = 6

        df = pd.read_parquet(input[0])
        vertices = pd.read_csv(input[1], sep="\t",usecols = [0,1,2])
        vertices["vertex_dot"] = vertices.vertex.str.replace("_", ".").str.lower()

        strains = df.vertex_dot.str.replace("\.$","",regex=True).str.replace("^\.","",regex=True).unique()
        strains = [strain for strain in strains if len(strain.replace(".","")) > 2]

        vertices_noass = vertices[(vertices.vertex_type.str.endswith("_assembly") == False)&(vertices.vertex_type != "gold_org")&(vertices.vertex_type != "patric_genome")&(vertices.vertex_type != "kegg_genome")&(vertices.vertex.str.contains("GCF_") == False)&(vertices.vertex.str.contains("GCA_") == False)].vertex_dot.to_list() # Remove assembly accessions

        def all_combinations(text):
            for length in range(len(text) + 1):
                for combo in itertools.combinations(text.split('.'), length):
                    yield '.'.join(combo)

        def filter_strains(df):
            def check_strainselectid(group):
                return len(group['StrainSelectID'].unique()) == 1

            filtered = df.groupby('strain').filter(check_strainselectid)

            return filtered.groupby('strain').head(1)

        def g(df):
            return df[df.groupby('strain')['score_partial'].transform("max") == df['score_partial']]

        def gf(df):
            return df[df.groupby('strain')['score_full'].transform("max") == df['score_full']]

        def to_sparse(chunk):
            return csr_matrix(chunk)

        all_matches_partial = process.cdist(strains, vertices_noass, scorer=fuzz.partial_ratio, workers=workers, score_cutoff=95)

        dask_array = da.from_array(all_matches_partial, chunks=(all_matches_partial.shape[0], all_matches_partial.shape[1]//(workers - 1)))
        del all_matches_partial
        sparse_matrix_dask = dask_array.map_blocks(to_sparse, dtype=csr_matrix)
        all_matches_partial_sparse = sparse_matrix_dask.compute()
        del sparse_matrix_dask

        nonzero_row_indices, nonzero_col_indices = all_matches_partial_sparse.nonzero()
        nonzero_values = all_matches_partial_sparse.data

        df_all_matches_partial = pd.DataFrame({
            'strain': nonzero_row_indices,
            'strainselect': nonzero_col_indices,
            'score_partial': nonzero_values
        })

        df_all_matches_partial['strain'] = df_all_matches_partial['strain'].map(lambda x: strains[x])
        df_all_matches_partial['strainselect'] = df_all_matches_partial['strainselect'].map(lambda x: vertices_noass[x])

        high_abundant = df_all_matches_partial.groupby("strain").size().sort_values(ascending=False)
        high_abundant = high_abundant[high_abundant < 2000].index.to_list()
        filtered_df_all_matches_partial = df_all_matches_partial[df_all_matches_partial['strain'].isin(high_abundant)].copy()

        del high_abundant, df_all_matches_partial
        filtered_df_all_matches_partial.loc[:,'score_parts'] = filtered_df_all_matches_partial.apply(
            lambda row: process.extractOne(row['strain'], list(all_combinations(row['strainselect']))[1:],score_cutoff=60)[1], axis=1
        )

        filtered_df_all_matches_partial = filtered_df_all_matches_partial.merge(vertices, left_on="strainselect", right_on="vertex_dot")

        df_matches_partial = filter_strains(g(filtered_df_all_matches_partial))
        del filtered_df_all_matches_partial
        df_matches_partial = df_matches_partial[df_matches_partial['score_partial'] > 70]
        df_matches_partial = df_matches_partial[~((df_matches_partial['score_parts'] <= 90) & (df_matches_partial['vertex_type'] == "biocyc_pgdb"))]

        # Full matches
        strains_left = list(set(strains) - set(df_matches_partial['strain'].unique()))

        all_matches_full = process.cdist(strains_left, vertices_noass, workers=workers, score_cutoff=90)

        dask_array = da.from_array(all_matches_full, chunks=(all_matches_full.shape[0], all_matches_full.shape[1]//(workers -1 )))
        del all_matches_full
        sparse_matrix_dask = dask_array.map_blocks(to_sparse, dtype=csr_matrix)
        all_matches_full_sparse = sparse_matrix_dask.compute()
        del sparse_matrix_dask


        nonzero_row_indices, nonzero_col_indices = all_matches_full_sparse.nonzero()
        nonzero_values = all_matches_full_sparse.data

        df_all_matches_full = pd.DataFrame({
            'strain': nonzero_row_indices,
            'strainselect': nonzero_col_indices,
            'score_full': nonzero_values
        })

        df_all_matches_full['strain'] = df_all_matches_full['strain'].map(lambda x: strains_left[x])
        df_all_matches_full['strainselect'] = df_all_matches_full['strainselect'].map(lambda x: vertices_noass[x])

        filtered_df_all_matches_full = df_all_matches_full.merge(vertices, left_on="strainselect", right_on="vertex_dot")

        df_matches_full = filter_strains(gf(filtered_df_all_matches_full))

        matches = pd.concat([df_matches_full, df_matches_partial], ignore_index=True)

        final = df.merge(matches,left_on="vertex_dot", right_on="strain", how="left")
        final = final.drop(columns = ["vertex_dot_x", "vertex_dot_y","strainselect","strain"])
        final.rename(columns={"score":"ner_score","vertex":"strainselect_vertex"})
        final.to_parquet(output[0])

rule merge_batch_outputs_strainselect:
    input:
        batch_outputs=expand(f"{preds}/batched_output_results/{{batch_id}}.parquet", batch_id=range(0, 500)),
    output:
        merged_output=f"{preds}/REL_output/preds_strainselect.pqt",
    resources:
        slurm_partition="single",
        runtime=100,
        mem_mb=20000
    run:
        l = []
        for i in input:
            l.append(pd.read_parquet(i))
        df = pd.concat(l)
        df.to_parquet(output[0])

rule group_entities:
    input:
        f"{preds}/REL_output/preds_strainselect.pqt",
    output:
        f"{preds}/REL_output/preds_strainselect_grouped.pqt",
    resources:
        slurm_partition="single",
        runtime=100,
        mem_mb=90000,
        ntasks=20
    run:
        df = pd.read_parquet(input[0])
        df = df.drop(columns=["label_rel","label"])
        df.rename(columns={"score":"ner_score",})

        l = []
        for ner in df["ner"].unique():
            df_filter = df[df["ner"] == ner]

            words = df_filter[df_filter["ner"]==ner].word_qc.value_counts()
            query_words = words[words > 1].index
            all_words = words.index
            cutoff = 95
            
            result = process.cdist(query_words, all_words, scorer=fuzz.token_sort_ratio, score_cutoff=cutoff, workers=-1)
            indices = np.argwhere(result >= cutoff)
            word_indices = list(zip(all_words[indices[:,0]], all_words[indices[:,1]]))
            matchesdf = pd.DataFrame(word_indices)

            scores = result[indices[:,0], indices[:,1]]
            matchesdf['score'] = scores
            unique_matches = matchesdf[matchesdf[0] != matchesdf[1]]

            word_counts = df.word_qc.value_counts()
            unique_matches.loc[:, 'total_count_0'] = unique_matches[0].map(word_counts)
            unique_matches.loc[:, 'total_count_1'] = unique_matches[1].map(word_counts)

            unique_matches.loc[:, 'consensus_word'] = unique_matches.apply(lambda x: x[0] if x['total_count_0'] > x['total_count_1'] else x[1], axis=1)

            # Create a dictionary to group words based on common connections
            grouped_words = defaultdict(list)
            for _, row in unique_matches.iterrows():
                grouped_words[row['consensus_word']].append(row)

            # Create a dictionary to map each consensus word to all connected words
            consensus_to_words = defaultdict(set)

            # Iterate through each group to check their abundances and select the consensus word
            for group_key, group_values in grouped_words.items():
                # Calculate the total count for each word in the group
                total_counts = {word: sum(unique_matches[unique_matches[0] == word]['total_count_0']) + sum(unique_matches[unique_matches[1] == word]['total_count_1']) for word in [row[0] for row in group_values] + [row[1] for row in group_values]}
                # Select the word with the highest total count as the consensus word
                consensus_word = max(total_counts, key=total_counts.get)
                
                # Add all words in the group to the set of the consensus word
                for row in group_values:
                    consensus_to_words[consensus_word].update([row[0], row[1]])
            
            df_filter['word_qc_group'] = df_filter['word_qc'].apply(lambda x: next((k for k, v in consensus_to_words.items() if x in v), x))
            l.append(df_filter)

        finaldf = pd.concat(l)
        finaldf.to_parquet(output[0])


rule write_download_file:
    input:
        f"{preds}/REL_output/preds_strainselect_grouped.pqt",
        f"{preds}/strainselect/StrainSelect21_vertices.tab.txt",
    output:
        f"{preds}/REL_output/strains_assemblies.txt"
    resources:
        slurm_partition="single",
        runtime=30,
        mem_mb=10000,
    run:
        df = pd.read_parquet(input[0])

        df = df.dropna(subset="StrainSelectID")
        vertices = pd.read_csv(input[1],sep="\t",usecols = [0,1,2])
        v_rs = vertices[vertices["vertex_type"]=="rs_assembly"]
        merged = df.merge(v_rs, on ="StrainSelectID")
        merged.loc[:,"assemblies"] = merged["StrainSelectID"] + "/" + merged["vertex_y"]
        assemblies = merged.drop_duplicates("assemblies").assemblies.to_list()

        with open(output[0],"w") as f:
            for a in assemblies:
                f.write(a+"\n")

rule create_network:
    input:
        f"{preds}/REL_output/preds_strainselect_grouped.pqt",
    output:
        f"{preds}/network.tsv",
        f"{preds}/strains.txt",
        # f"{preds}/network_assemblies.tsv",
        # f"{preds}/strains_assemblies.txt",
    resources:
        slurm_partition="single",
        runtime=30,
        mem_mb=10000,
    params:
        data=str(config["dataset"]),
    run:
        df = pd.read_parquet(input[0])
        # filter out wrongly assigned strains
        df = df[~df["word_strain_qc"].str.contains("adapted|covid")]

        network = df[df.StrainSelectID.isna() == False].loc[:,["StrainSelectID","word_qc_group","rel"]].drop_duplicates(["StrainSelectID","word_qc_group","rel"])
        network.loc[:,"source"] = np.where(network['rel'].str.startswith("STRAIN"), network.StrainSelectID	, network.word_qc_group)
        network.loc[:,"target"] = np.where(network['rel'].str.startswith("STRAIN")==False, network.StrainSelectID, network.word_qc_group)

        network = network.loc[:,["source","target","rel"]]

        network = pd.concat([network,
            network.rel.str.split(":",expand=True)[0].str.split("-",expand=True).rename(
            columns={0:"source_ner",1:"target_ner"})]
                ,axis=1
                )

        network["rel"] = network.rel.str.split(":",expand=True)[1]

        network.to_csv(output[0],index=False,sep="\t")

        with open(output[1], "w") as f:
            for s in sorted(set(df[df.StrainSelectID.isna() == False].StrainSelectID.to_list())):
                f.write(f"{s}\n")

        # # filtered network
        
        # folders = glob(f"{output_path}/assemblies_{params.data}/*")

        # strains = [f.split("/")[-1] for f in folders]
        # filtered_df= df[df.StrainSelectID.isin(strains)]

        # network = filtered_df.loc[:,["StrainSelectID","word_qc_group","rel"]].drop_duplicates(["StrainSelectID","word_qc_group","rel"])
        # network.loc[:,"source"] = np.where(network['rel'].str.startswith("STRAIN"), network.StrainSelectID	, network.word_qc_group)
        # network.loc[:,"target"] = np.where(network['rel'].str.startswith("STRAIN")==False, network.StrainSelectID, network.word_qc_group)

        # network = network.loc[:,["source","target","rel"]]

        # network = pd.concat([network,
        # network.rel.str.split(":",expand=True)[0].str.split("-",expand=True).rename(
        # columns={0:"source_ner",1:"target_ner"})]
        #     ,axis=1
        #     )

        # network["rel"] = network.rel.str.split(":",expand=True)[1]
        # network.to_csv(output[2],index=False,sep="\t")

        # with open(output[3], "w") as f:
        #     for s in sorted(set(filtered_df.StrainSelectID.to_list())):
        #         f.write(f"{s}\n")
