import json
import os


xml_path = 'C:/fruit/HF01_00LF/HF01_00LF_000001.json'

with open(xml_path,'r',encoding='UTF8') as json_file:
    print(json_file)
    json_data = json.load(json_file)
    print(json_data["Annotations"]["OBJECT_CLASS_CODE"])
# jsons = json.load(xml_path)

# print(jsons)