'''
This file takes calc_data_1.json as input and renames the file_
'''

import json

if __name__ == '__main__':
    jsons = list()

    with open(f"combined_data.json") as f:
        jsons = json.load(f)

    start_num = 1257
    for item in jsons:
        for i in item:
            print(i)

    # with open("calc.json", 'w') as f:
    #     json.dump(jsons, f, indent=2)