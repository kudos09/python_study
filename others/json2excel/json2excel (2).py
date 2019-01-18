# -*- coding: utf-8 -*-
# 这段代码主要的功能是把utf-8格式的json文件转换成excel表格

import openpyxl
import json

def readFromJson(jsfile):
    with open(jsfile, 'r', encoding='utf8') as fr:
        jsonData = json.load(fr)
    return jsonData

def writeToExcel(jsfile,excfile):
    json = readFromJson(jsfile)
    print(json)
    excel = openpyxl.Workbook()
    sheet1 = excel.create_sheet('param')
    print(sheet1)
    ll = list(json[0].keys())
    for i in range(0,len(ll)):
        sheet1.write(0,i,ll[i])
    for j in range(0,len(json)):
        m = 0
        ls = list(json[j].values())
        for k in ls:
            sheet1.write(j+1,m,k)
            m += 1
    excel.save(excfile)

if __name__ == '__main__':
    jsfile = "test.json"
    excfile = "ttt.xlsx"
    writeToExcel(jsfile, excfile)