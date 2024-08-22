import pandas as pd
from tqdm.autonotebook import tqdm
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pickle

data = snakemake.params.data
device = snakemake.params.device[0]
path = snakemake.params.path

ip_names = pd.read_csv(
    "https://ftp.ebi.ac.uk/pub/databases/interpro/current_release/entry.list",
    sep="\t",
    header=0,
)
ip_names.set_index("ENTRY_AC", inplace=True)

ip_names["ENTRY_NAME"] = (
    ip_names["ENTRY_NAME"]
    .str.replace("[", "_")
    .str.replace("]", "_")
    .str.replace("<", "_")
)


def calculate_acc(dtest, bst, enc, y_test_binary):
    # calculate accuracy
    y_pred_binary = bst.predict(dtest)
    predictions = [round(value) for value in y_pred_binary]
    # evaluate predictions
    acc = accuracy_score(enc.transform(y_test_binary), predictions)
    return acc


def process_rel(filepath, device, ip_names):
    d_rel = []
    # Read the pickle file
    with open(filepath, "rb") as f:
        dat = pickle.load(f)
    vc = pd.DataFrame(dat[1]).value_counts()
    X = dat[0]
    y = dat[1]
    ind = dat[2]
    vc = vc[vc >= 10] #cutoff here!

    ind_names = [ip_names[ip_names.index == i]["ENTRY_NAME"].values[0] for i in ind]

    for i in vc.index:
        y_binary = ["target" if label == i[0] else "other" for label in y]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y_binary, test_size=0.2, stratify=y, random_state=42
        )

        enc = LabelEncoder().fit(y_train)

        dtrain = xgb.DMatrix(
            X_train, label=enc.transform(y_train), feature_names=ind_names
        )
        dtest = xgb.DMatrix(
            X_test, label=enc.transform(y_test), feature_names=ind_names
        )

        param = {
            "max_depth": 6,
            "eta": 0.3,
            "objective": "binary:logistic",
            "device": f"cuda:{device}",
            "eval_metric": ["logloss"],
            "colsample_bylevel": 1,
            "booster": "gbtree",
        }

        evallist = [(dtrain, "train"), (dtest, "eval")]
        bst = xgb.train(
            param,
            dtrain,
            5000,
            evals=evallist,
            early_stopping_rounds=10,
            verbose_eval=False,
        )

        accuracy = calculate_acc(dtest, bst, enc, y_test)
        d_rel.append([i, accuracy, bst])
        print(f"{filepath.split("/")[-1]} {i} {accuracy} {vc[i]}")
    return d_rel

print(snakemake.input)
result = process_rel(str(snakemake.input), device, ip_names)

with open(snakemake.output[0], "wb") as f:
    pickle.dump(result, f)
