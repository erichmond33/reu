'''
This file takes calc_data_1.json as input and renames the file_
'''

import json

if __name__ == '__main__':
    jsons = list()

    with open(f"calc_data_0.json") as f:
        jsons = json.load(f)

    start_num = 1257
    for item in jsons:
        for i in item['calculator_outputs']:
            try:
                i[3] = str(round(float(i[3]), 2))
            except:
                continue
        start_num += 1

    # start_num = 1257
    # for item in jsons:
    #     item['file_index'] = start_num
    #     start_num += 1

    with open("calc.json", 'w') as f:
        json.dump(jsons, f, indent=2)