import json
from glob import glob

print("### NER")
for ner_json in glob("NER_output/*/trainer_state.json"):
    with open(ner_json,"r") as f:
        json_file = json.load(f)
    best_metric = json_file["best_metric"]
    for epoch in json_file["log_history"]:
        if epoch.get("eval_loss") == best_metric:
            print(f"{ner_json.split('/')[-2]} -> EPOCH={epoch['epoch']}: F1={epoch['eval_f1']}")

print("### RE")
for rel_json in glob("REL_output/*/trainer_state.json"):
    with open(rel_json,"r") as f:
        json_file = json.load(f)
    best_metric = json_file["best_metric"]
    for epoch in json_file["log_history"]:
        if epoch.get("eval_F1") == best_metric:
            print(f"{rel_json.split('/')[-2]} -> EPOCH={epoch['epoch']}: F1={best_metric}")
    