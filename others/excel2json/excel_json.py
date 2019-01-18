# -*- coding: utf-8 -*-
# 这段代码主要的功能是把excel表格转换成utf-8格式的json文件

import os
import sys
import codecs
import xlrd
import shutil

def FloatToString(aFloat):
    if type(aFloat) != float:
        return ""
    strTemp = str(aFloat)
    strList = strTemp.split(".")
    if len(strList) == 1:
        return strTemp
    else:
        if strList[1] == "0":
            return strList[0]
        else:
            return strTemp

def table2jsn(table, jsonfilename):
    nrows = table.nrows
    ncols = table.ncols
    f = codecs.open(jsonfilename, "wb", "utf-8")
    f.write(u"{\n\t\"list\":[\n")
    for r in range(nrows - 1):
        f.write(u"\t\t{ ")
        for c in range(ncols):
            strCellValue = u""
            CellObj = table.cell_value(r + 1, c)
            if type(CellObj) == str:
                strCellValue = CellObj
            elif type(CellObj) == float:
                strCellValue = FloatToString(CellObj)
            else:
                strCellValue = str(CellObj)
            strTmp = u"\"" + table.cell_value(0, c) + u"\":" + strCellValue
            if c < ncols - 1:
                strTmp += u", "
            f.write(strTmp)
        f.write(u" }")
        if r < nrows - 2:
            f.write(u",")
        f.write(u"\n")
    f.write(u"\t]\n}\n")
    f.close()
    print(    "Create ", jsonfilename, " OK")
    return

def move2file(path1,path2):
    filelist = os.listdir(path1)
    for files in filelist:
        filename1 = os.path.splitext(files)[1]  # 读取文件后缀名
        filename0 = os.path.splitext(files)[0]  #读取文件名
        m = filename1 == '.json'
        #print(m)
        if m :
            full_path = os.path.join(path1, files)
            despath = path2 + filename0+'.json'
            shutil.move(full_path, despath)
            #print("move", filename0,".json to source\ success!")
        else :
            continue

path1 = "./"
path2 = "../source/"
path3 = "../source/调试模块信息收集_v0.2.4.xlsx"
data = xlrd.open_workbook(path3)
table = data.sheet_by_name(u"tablelist")
rs = table.nrows
for r in range(rs - 1):
    #print( table.cell_value(r + 1, 0), "==>", table.cell_value(r + 1, 2))
    desttable = data.sheet_by_name(table.cell_value(r + 1, 0))
    destfilename = table.cell_value(r + 1, 2)
    table2jsn(desttable, destfilename)
    move2file(path1,path2)
print ("All OK")
