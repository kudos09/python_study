import xlrd
from collections import OrderedDict
import json
import codecs

def writeToExcel(jsonfile, excfile):
    worksheet = xlrd.open_workbook(excfile)
    sheet_names = worksheet.sheet_names()
    print("www", sheet_names)
    convert_list = []
    for sheet_namee in sheet_names:
        sheet = worksheet.sheet_by_name(sheet_namee)
        for sheet_name in range(len(sheet_names)-1):
            title = []#sheet.row_values(sheet_name)
            print("title:", title, "sheet_namee:", sheet_namee, "sheet_name", sheet_name)
            for rownum in range(1, sheet.nrows):
                rows = sheet.row_values(rownum)
                #print("rows:", rows)
                single = OrderedDict()
                for colnum in range(0, len(rows)):
                    single[title[colnum]] = rows[colnum]
            print("sheet_name:", sheet_name, "rownum:", rownum, "colnum:", colnum, "sheet_namee:", sheet_namee, "single:", single)  # 获取第四行内容
            convert_list.append(single)
            #cols = sheet.col_values(sheet_name)
    print("AAA")
    j = json.dumps(convert_list,indent=4, ensure_ascii=False)
    with codecs.open(jsonfile, "wb", "utf-8") as f:
        f.write(j)
        f.close()
    print("sss")


if __name__ == '__main__':
    jsfile = "./jsonfile222.json"
    excfile = "./excelfile.xls"
    writeToExcel(jsfile, excfile)
    print ("json to excel OK")