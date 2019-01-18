# -*- coding: utf-8 -*-
# 这段代码主要的功能是把utf-8格式的json文件转换成excel表格
import json, xlwt
import sys

def writeToExcel(jsonfile,excfile):
    with open(jsonfile, encoding='utf-8') as f:
        jsonData = json.load(f)
    book = xlwt.Workbook()  # 创建excel文件

    sheet1 = book.add_sheet('List', cell_overwrite_ok=True)  # 创建list表
    for testlabel in range(len(jsonData['labell'])):
        sheet1.write(testlabel, 0, testlabel)  # 存入list sheet label
        sheet1.write(testlabel, 1, jsonData['labell'][testlabel])  # 存入list sheet label
        sheet = book.add_sheet(jsonData['labell'][testlabel], cell_overwrite_ok=True)  # 创建labell表
        for titlelen in range(len(jsonData['titall'])):
            sheet.col(titlelen).width = 256 * 30
            sheet.write(0, titlelen, jsonData['titall'][titlelen])  # 存入第一行标题
            datanum = 1
            for datalinenum in range(len(jsonData['datashading'])):
                if jsonData['labell'][testlabel] == jsonData['datashading'][datalinenum][0]:   # 匹配对应文件
                    for datalens in range(len(jsonData['datashading'][datalinenum])):
                        sheet.write(datanum, datalens, jsonData['datashading'][datalinenum][datalens])  # 匹配对应行写入表
                        print("datanum:", datanum, "testlabel: ", testlabel, "titlelen: ", titlelen,"datalens: ",datalens,"datalinenum:", datalinenum, jsonData['datashading'][datalinenum])
                    datanum +=1
    book.save(excfile)
    #print(    "Create ", excfile, " OK")


if __name__ == '__main__':
    jsfile = "./jsonfile.json"
    excfile = "./excelfile.xls"
    writeToExcel(jsfile, excfile)
print ("json to excel OK")