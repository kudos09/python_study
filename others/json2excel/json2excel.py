# -*- coding: utf-8 -*-
# 这段代码主要的功能是把utf-8格式的json文件转换成excel表格
import json, xlwt

def writeToExcel(jsonfile,excfile):
    with open(jsonfile, encoding='utf-8') as f:  # 将json文件转化为字典
        jsonData = json.load(f)

    book = xlwt.Workbook()  # 创建excel文件
    sheet1 = book.add_sheet('history')  # 创建一个表
    sheet2 = book.add_sheet('history')  # 创建一个表
    sheet3 = book.add_sheet('history')  # 创建一个表
    sheet4 = book.add_sheet('history')  # 创建一个表
    sheet5 = book.add_sheet('history')  # 创建一个表
    sheet6 = book.add_sheet('history')  # 创建一个表


    title1 = ['NO', 'date', 'Owner', 'Comment', 'mark']
    title2 = ['NO', 'date', 'Owner', 'Comment', 'mark']
    title3 = ['NO', 'date', 'Owner', 'Comment', 'mark']
    title4 = ['NO', 'date', 'Owner', 'Comment', 'mark']
    title5 = ['NO', 'date', 'Owner', 'Comment', 'mark']
    title6 = ['NO', 'date', 'Owner', 'Comment', 'mark']

    for col in range(len(title1)):  # 存入第一行标题
        sheet1.write(0, col, title1[col])
        sheet2.write(0, col, title2[col])
        sheet3.write(0, col, title3[col])
        sheet4.write(0, col, title4[col])
        sheet5.write(0, col, title5[col])
        sheet6.write(0, col, title6[col])

    row = 1  # 定义行
    for k in jsonData:
        data = jsonData[k]  # data保存姓名和分数的list
        data.insert(0, k)  # 第一列加入序号
        for index in range(len(data)):  # 依次写入每一行
            sheet1.write(row, index, data[index])
        row += 1
    book.save(excfile)

if __name__ == '__main__':
    jsfile = "../source/jsonfile.json"
    excfile = "../source/excelfile.xls"
    writeToExcel(jsfile, excfile)
print ("json to excel OK")