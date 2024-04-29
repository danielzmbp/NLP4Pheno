from glob import glob

labels = glob("label/*.json")
labels.sort()

# Read the json file
with open(labels[-1], "r") as f:
    data = f.read()


data = data.replace("MORPHOLOGY","PHENOTYPE").replace("METABOLITE", "COMPOUND").replace("FORMS","PRESENTS")
with open("label/annotations.json","w") as o:
    o.write(data)
