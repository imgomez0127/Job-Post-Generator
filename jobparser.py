import json
import os
def importAllJsons():
    jsonLst = []
    path = "crawler/data/"
    files = os.listdir(path) 
    for fname in files:
        with open(path+fname,"r") as f:
            jsonLst.append(json.load(f))
    return jsonLst
if __name__ == "__main__":
    print(importAllJsons())
