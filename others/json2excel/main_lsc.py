#!/usr/bin/python
# -*- coding: UTF-8 -*-

import sys
import os
import random
import json, xlwt


def GetJsonFile(jsfile):
    f = open(jsfile, encoding='utf-8')
    jsonData = json.load(f)
    return jsonData

def UpdateDataM(jsonData):
    #print("ccc", jsonData)
    for i in range(len(jsonData['datashading'])):
        print("aaa:", jsonData['datashading'][i])
        #del jsonData['datashading'][i][4]
        #print("bbb:", jsonData['datashading'][i][5])
        #del jsonData['datashading'][i][5]
        #print("ccc:", jsonData['datashading'][i][6])
        #del jsonData['datashading'][i][6]
        #del jsonData['datashading'][i][8]
        #del jsonData['datashading'][i][9]

    #print("ss:", i,   jsonData)
    return jsonData
        #jsonData[i]['Score'] = random.randrange(10,100,5)


def UpdateData(jsonData):
    print("ccc", jsonData)
    for i in range(len(jsonData)):
        lista = jsonData[i]['TestTarget']
        listfile = lista.split('.')
        print("ss:", i,  listfile[1], lista)
        jsonData[i]['Score'] = random.randrange(10,100,5)
        if jsonData[i]['Score'] < 60.00:
            jsonData[i]['Flag'] = -1
        else:
            jsonData[i]['Flag'] = 1

        if listfile[1] == "D65":
            jsonData[i]['cur_ct'] = 6500 + random.randrange(-800,3000,50)
        elif listfile[1] == "D50":
            jsonData[i]['cur_ct'] = 5000 + random.randrange(-500,700,50)
        elif listfile[1] == "CWF":
            jsonData[i]['cur_ct'] = 4000 + random.randrange(-200,600,50)
        elif listfile[1] == "TL84":
            jsonData[i]['cur_ct'] = 3800 + random.randrange(-400,200,50)
        elif listfile[1] == "A":
            jsonData[i]['cur_ct'] = 2800 + random.randrange(-400,600,50)
        elif listfile[1] == "H":
            jsonData[i]['cur_ct'] = 2000 + random.randrange(-500,500,50)
        else:
            print("error3333333333333333333333333333")
            #return
        print("TestTarget:", jsonData[i])
    return jsonData

def rewrite_json_file(json_data):
    with open('jsonfile.json', 'w') as f:
        json.dump(json_data, f, indent=4, ensure_ascii=False)
        print("list json_data : ", json_data)
    f.close()
    return json_data

def JsonToExcel(jsonfile,excfile):
    with open(jsonfile, encoding='utf-8') as f:  # 将json文件转化为字典
        jsonData = json.load(f)
    book = xlwt.Workbook()  # 创建excel文件
    sheet = book.add_sheet('data', cell_overwrite_ok=True)  # 创建表
    title = ['TestTarget', 'Flag', 'cur_ct', 'Score']
    for titlelen in range(len(title)):
        sheet.write(0, titlelen, title[titlelen])  # 存入第一行标题
    datanum = 1
    for datalinenum in range(len(jsonData)):
        sheet.col(datalinenum).width = 256 * 30
        for datalens in range(len(jsonData[datalinenum])):
            #dataa = one_dict[]
            #sheet.write(datanum, datalens, jsonData[datalinenum][datalens])  # 匹配对应行写入表
            print("datanum:", datanum, "datalens: ",datalens, "datalinenum:", datalinenum, jsonData[datalinenum])
            datanum += 1
    book.save(excfile)  # 保存到xls


def CheckResult(testresult):
    KeyValue = testresult
    print(KeyValue)
    Fail_list = []
    for k in range(len(KeyValue)):
        print(KeyValue[k])
        if KeyValue[k]['Score'] < 60.00:
            print("Fail")
            Fail_list.append(KeyValue[k])
        elif KeyValue[k]['Score'] > 100.00:
            print("error")
            Fail_list.append(KeyValue[k])
        else:
            print("pass")

    print("Fail_list = ", Fail_list)
    return Fail_list

def plan1(cur_ct):
    print("Run plan1")

def plan2(cur_ct):
    print("Run plan2")

def plan3(cur_ct):
    print("Run plan3")

def plan4(cur_ct):
    print("Run plan4")

def plan5(cur_ct):
    print("Run plan5")

def plan6(cur_ct):
    print("Run plan6")

def plan7(cur_ct):
    print("Run plan7")

def plan8(cur_ct):
    print("Run plan8")


def AdaptLsc(Check):
    print("enter adapt lsc")
    checkresult = Check
    #print("aaa", checkresult)
    for num in range(len(checkresult)):
        cur_ct = checkresult[num]["cur_ct"]
        print("num:", num, cur_ct)
        ct_list = {"D65_max": 6500, "D65_min": 6300, "CWF_max": 4200, "CWF_min": 4000, "TL84_max": 800, "TL84_min": 3700,
                   "A_max": 3000, "A_min": 2800}
        if cur_ct > ct_list['D65_max']:
            plan1(cur_ct)
        elif ct_list['CWF_max'] < cur_ct <= ct_list['D65_min']:
            plan2(cur_ct)
        elif ct_list['CWF_min'] < cur_ct <= ct_list['CWF_max']:
            plan3(cur_ct)
        elif ct_list['TL84_max'] < cur_ct <= ct_list['CWF_min']:
            plan4(cur_ct)
        elif ct_list['TL84_min'] < cur_ct <= ct_list['TL84_max']:
            plan5(cur_ct)
        elif ct_list['A_max'] < cur_ct <= ct_list['TL84_min']:
            plan6(cur_ct)
        elif cur_ct <= ct_list['A_max']:
            plan7(cur_ct)
        else:
            print("error, run plan8 check file")
            plan8(cur_ct)

    print("end adapt lsc ")


def main():
    jsfile = "./jsonfile1.json"
    excfile = "./excelfile.xls"
    print("enter main adapt lsc")
    testresult = GetJsonFile(jsfile)
    Updata1 = UpdateDataM(testresult)
    #Updata = UpdateData(testresult)
   # rewrite_json_file(Updata1)
    #JsonToExcel(jsfile, excfile)
    #Check = CheckResult(testresult)
    #AdaptLsc(Check)


if __name__ == "__main__":
    main()
