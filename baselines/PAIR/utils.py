import json


def read_data(data_path: str):
    if data_path.endswith(".json"):
        with open(data_path, "r") as file:
            data = json.load(file)
    else:
        with open(data_path, "r", encoding="utf-8") as file:
            data = file.read().split("\n")
            # filter ''
            data = list(filter(lambda x: x != '', data))

    return data