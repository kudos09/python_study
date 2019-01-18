# -*- coding: utf-8 -*-
# 这段代码主要的功能是把utf-8格式的json文件转换成excel表格

import json, xlwt


def readJsonfile(jsonfiles):
    with open(jsonfiles, encoding='utf-8') as f:  # 将json文件转化为字典
        jsobj = json.load(f)
        #print(jsobj)
    return jsobj

def jsonToexcel(jsonfiles):
    jsonfile = readJsonfile(jsonfiles)
    print (jsonfile)
    workbook = xlwt.Workbook()
    sheet1 = workbook.add_sheet('param')
    ll = list(jsonfile[0].keys())
    for i in range(0,len(ll)):
        sheet1.write(0,i,ll[i])
    for j in range(0,len(jsonfile)):
        m = 0
        ls = list(jsonfile[j].values())
        for k in ls:
            sheet1.write(j+1,m,k)
            m += 1
    workbook.save('test.xls')

jsonToexcel('score.json')