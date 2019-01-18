import xlrd
from collections import OrderedDict
import json
import codecs


def GetJsonFile(jsfile):
    f = open(jsfile, encoding='utf-8')
    jsonData = json.load(f)
    # print(jsonData);
    return jsonData


def savejsonfile(jsondata,jsonfile):
    j = json.dumps(jsondata,indent=4, ensure_ascii=False)
    with codecs.open(jsonfile, "wb", "utf-8") as f:
        f.write(j)
        f.close()


if __name__ == '__main__':
    jsfile = "./data_1000.json"
    savefile = "./data_1001.json"
    jsondata = GetJsonFile(jsfile)
    savejsonfile(jsondata, savefile)
    print ("json to excel OK")