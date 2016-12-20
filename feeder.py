import sklearn
import csv
import json

def normalize_data():
    with open("./data/preprocessed_metadata_doi.csv") as csv_meta:
        pos = { nam.strip() : i for i, nam in  enumerate( csv_meta.readline().split(";") ) }
        all_data = []
        all_targets = []
        current_series, current_data, current_target = None, None, None
        for row in csv_meta:
            row = row.split(";")
            if current_series is None or current_series != row[pos["Series Instance UID"]]:
                if current_series is not None:
                    # Add the previous.
                    all_data.append(current_data)
                    all_targets.append(current_target)
                current_series = row[pos["Series Instance UID"]]
                current_data = { k : row[pos[k]] for k in pos.keys() if k != "sstat" }
                current_data["png_path"] = [current_data["png_path"].strip()]
                current_target = row[pos["sstat"]]
            else:
                current_data["png_path"].append(row[pos["png_path"]].strip())


    all_data.append(current_data)
    all_targets.append(current_target)

    # return all_data, all_targets
    # all_data, all_targets = normalize_data()
    dataset = { 'data' : all_data, 'target' : all_targets}


    f = open("data.json", 'w')
    json.dump(dataset, f)

def generator():
    with open('data.json') as data_file:
        dataset = json.load(data_file)
        all_data, all_targets = dataset['data'], dataset['target']
        for data, target in zip(all_data, all_targets):
            yield data, target

normalize_data()


for d, t in generator():
    print d["png_path"]
    print t
