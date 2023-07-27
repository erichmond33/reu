# A function that loops through every json file in this folder and prints
# the number of calculator_outputs that have a score
# higher that .1 and the file name

import json
import os

for file in os.listdir("./"):
    if file.endswith(".json"):
        with open(file) as f:
            data = json.load(f)
            count_dict = {"0.1" : 0, "0.15" : 0, "0.2" : 0, "0.25" : 0, "0.3" : 0, "0.35" : 0, "0.4" : 0, "0.45" : 0, "0.5" : 0, "0.55" : 0, "0.6" : 0}
            for item in data:
                for output in item["calculator_outputs"]:
                    if output[0] > .1:
                        count_dict["0.1"] += 1
                    if output[0] > .15:
                        count_dict["0.15"] += 1
                    if output[0] > .2:
                        count_dict["0.2"] += 1
                    if output[0] > .25:
                        count_dict["0.25"] += 1
                    if output[0] > .3:
                        count_dict["0.3"] += 1
                    if output[0] > .35:
                        count_dict["0.35"] += 1
                    if output[0] > .4:
                        count_dict["0.4"] += 1
                    if output[0] > .45:
                        count_dict["0.45"] += 1
                    if output[0] > .5:
                        count_dict["0.5"] += 1
                    if output[0] > .55:
                        count_dict["0.55"] += 1
                    if output[0] > .6:
                        count_dict["0.6"] += 1
            print(count_dict)
            print(f"Total: " + str(sum(count_dict.values())))
            print(f.name, end="\n\n")
