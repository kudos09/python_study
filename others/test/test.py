import json, xlwt


def read_score(jsonfile):
    with open(jsonfile, encoding='utf-8') as f:  # 将json文件转化为字典
        score_all = json.load(f)

    book = xlwt.Workbook()  # 创建excel文件
    sheet = book.add_sheet('sheet1')  # 创建一个表
    title = ['序号', '姓名', '语文', '数学', '英语', '总分', '平均分']
    for col in range(len(title)):  # 存入第一行标题
        sheet.write(0, col, title[col])
    row = 1  # 定义行
    for k in score_all:
        data = score_all[k]  # data保存姓名和分数的list
        #data.append(sum(data[1:4]))  # 倒数第二列加入总分
        #data.append(sum(data[1:4]) / 3.0)  # 最后一列加入平均分
        #data.insert(0, k)  # 第一列加入序号
        for index in range(len(data)):  # 依次写入每一行
            sheet.write(row, index, data[index])
        row += 1
    book.save('test.xls')


read_score('test.json')