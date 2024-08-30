from openai import OpenAI
import pandas as pd
import json
import csv

with open('UWCourseCatalog_05-23-2024.json', 'r') as file:
    data = json.load(file)
    data2 = []
    for list in data:
        if list[1]:
            for dict in list[1]:
                data2.append(dict)
    # print(data2)

with open('UWCourseCatalog_05-23-2024.csv', 'w', newline='') as file:
    f = csv.writer(file)
    f.writerow(["course-code", "course-title", "credits", "description", "mortarboard", "Requisites:"])
    for dict in data2:
        requisites = dict["Requisites:"] if "Requisites:" in dict else "N/A"
        f.writerow([dict.get("course-code", "N/A"), dict.get("course-title", "N/A"), dict.get("credits", "N/A"), dict.get("description", "N/A"), dict.get("mortarboard", "N/A"), requisites])

