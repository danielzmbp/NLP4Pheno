import pandas as pd
import itertools


configfile: "config.yaml"


cutoff = config["cutoff_prediction"]
preds = "/home/tu/tu_tu/tu_bbpgo01/link/preds" + config["dataset"]
labels = config["rel_labels"]
cuda = config["cuda_devices"]


rule all:
    input:
        f"{preds}/REL_output/preds.pqt",


rule format_sentences:
    input:
        f"{preds}/NER_output/preds.parquet",
    output:
        f"{preds}/NER_output/ner_preds.parquet",
    resources:
        slurm_partition="single",
        runtime=100,
        mem_mb=20000
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
        runtime=500,
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
        runtime=300,
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
            .str.replace("^brain heart infusion broth$", "bhi broth", regex=True)
            .str.replace(
                "^brain heart infusion \( bhi \) broth$", "bhi broth", regex=True
            )
            .str.replace("^brain - heart infusion broth$", "bhi broth", regex=True)
            .str.replace("^lysogeny broth \( lb \)$", "lb", regex=True)
            .str.replace("^lysogeny broth$", "lb", regex=True)
            .str.replace("^lb medium$", "lb", regex=True)
            .str.replace("^lb - agar$", "lb agar", regex=True)
            .str.replace("^lb - broth$", "lb broth", regex=True)
            .str.replace("^luria bertani$", "lb", regex=True)
            .str.replace("^luria bertani \( lb \)$", "lb", regex=True)
            .str.replace("^luria broth \( lb \)$", "lb broth", regex=True)
            .str.replace("^luria broth$", "lb broth", regex=True)
            .str.replace("^luria - bertani \( lb \) broth$", "lb broth", regex=True)
            .str.replace("^luria - bertani broth$", "lb broth", regex=True)
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
            .str.replace("^mueller − hinton$", "mueller hinton", regex=True)
            .str.replace("^tryptic -", "tryptic", regex=True)
            .str.replace("^mueller - hinton$", "mueller hinton", regex=True)
            .str.replace("^mueller-hinton", "mueller hinton", regex=True)
            .str.replace("^terrific - broth -", "terrific broth", regex=True)
            .str.replace("^nematode - growth$", "nematode growth", regex=True)
            # metabolite
            .str.replace("^acetyl - coa$", "acetyl coa", regex=True)
            # phenotype
            .str.replace("^gram- negative$", "gram - negative", regex=True)
            .str.replace("^gram- positive$", "gram - negative", regex=True)
            .str.replace("^gram positive$", "gram - positive", regex=True)
            .str.replace("^gram negative$", "gram - negative", regex=True)
            .str.replace("^gram-positive$", "gram - positive", regex=True)
            .str.replace("^gram-negative$", "gram - negative", regex=True)
            .str.replace("^gram ‐ negative$", "gram - negative", regex=True)
            .str.replace("^gram - stain - negative$", "gram - negative", regex=True)
            .str.replace("^hydrogen oxidizing$", "hydrogen - oxidizing", regex=True)
            .str.replace("^acetate oxidizing$", "acetate - oxidizing", regex=True)
            .str.replace("^manganese oxidizing", "manganese - oxidizing", regex=True)
            .str.replace("^iron reducing$", "iron - reducing", regex=True)
            .str.replace("^sulfur reducing$", "sulfur - reducing", regex=True)
            .str.replace("^metal reducing$", "metal - reducing", regex=True)
            .str.replace("^nitrogen fixing$", "nitrogen - fixing", regex=True)
            .str.replace("^iron-reducing$", "iron - reducing", regex=True)
            .str.replace("^nitrate reducing$", "nitrate - reducing", regex=True)
            .str.replace("^methanotrophs$", "methanotroph", regex=True)
            .str.replace(
                "^facultatively anaerobic$", "facultative anaerobic", regex=True
            )
            .str.replace("^facultative anaerobe$", "facultative anaerobic", regex=True)
            .str.replace("^anaerobe$", "anaerobic", regex=True)
            .str.replace("^n - fixing$", "nitrogen - fixing", regex=True)
            .str.replace("^fast-growing$", "fast - growing", regex=True)
            .str.replace("^salt tolerant$", "salt - tolerant", regex=True)
            # isolate
            .str.replace("^marine sediments$", "marine sediment", regex=True)
            .str.replace("^human faeces$", "human feces", regex=True)
            .str.replace("^surface waters$", "surface water", regex=True)
            .str.replace("^sea water$", "seawater", regex=True)
            .str.replace("^sediments$", "sediment", regex=True)
            .str.replace("^soils$", "soil", regex=True)
            .str.replace("^water sample?$", "water", regex=True)
            .str.replace("^stool$", "feces", regex=True)
            .str.replace("^soil sample?$", "soil", regex=True)
            .str.replace(
                "^moss - dominated soil crusts$",
                "moss - dominated soil crust",
                regex=True,
            )
            .str.replace("chromium contaminated", "chromium - contaminated", regex=True)
            .str.replace("deep - subsurface", "deep subsurface", regex=True)
            # morphology
            .str.replace("^biofilms$", "biofilm", regex=True)
            .str.replace("^spores$", "spore", regex=True)
            .str.replace("^endospores$", "endospore", regex=True)
            .str.replace("^heterocysts$", "heterocyst", regex=True)
            .str.replace("^filaments$", "filament", regex=True)
            .str.replace("^rod-shaped$", "rod - shaped", regex=True)
            .str.replace("^rod shaped$", "rod - shaped", regex=True)
            .str.replace("wrinkled colonies", "wrinkled colony", regex=True)
            # compound
            .str.replace("^heavy metals$", "heavy metal", regex=True)
            .str.replace("^hydrocarbons$", "hydrocarbon", regex=True)
            .str.replace("^cu$", "copper", regex=True)
            .str.replace("^metals", "metal", regex=True)
            .str.replace("^zn$", "zinc", regex=True)
            .str.replace("^ni$", "nickel", regex=True)
            # organism
            .str.replace("^humans$", "human", regex=True)
            .str.replace("^mice$", "mouse", regex=True)
            .str.replace("^birds$", "bird", regex=True)
            .str.replace("^soybeans$", "soybean", regex=True)
            .str.replace("^wild boar$", "wild boar", regex=True)
            .str.replace("^chickens$", "chicken", regex=True)
            .str.replace("^cockroaches$", "cockroach", regex=True)
            .str.replace("^dogs$", "dog", regex=True)
            .str.replace("^insects$", "insect", regex=True)
            .str.replace("^legumes$", "legume", regex=True)
            .str.replace("^marine sponges$", "marine sponge", regex=True)
            .str.replace("^mosquitoes$", "mosquito", regex=True)
            .str.replace("^potatoes$", "potato", regex=True)
            # effect
            .str.replace("^antimicrobial activity$", "antimicrobial", regex=True)
            .str.replace("^antibacterial activity$", "antibacterial", regex=True)
            .str.replace("^antibacterial effects$", "antibacterial", regex=True)
            .str.replace("^antifungal activity$", "antifungal", regex=True)
            .str.replace("^plant - growth", "plant growth", regex=True)
            .str.replace("^plant-growth", "plant growth", regex=True)
            .str.replace("plant beneficial", "plant - beneficial", regex=True)
        )
        d[d["score_rel"] > cutoff].to_parquet(output[0])
