# -*- coding: utf-8 -*-
# 这段代码主要的功能是把utf-8格式的json文件转换成excel表格
import xlwt

result = [
    ['姓名', '性别', '年龄'],
    ['张三11111111111111111', '男', 186],
    ['李四', '男', 18],
    ['小花', '女', 16],
    ['梅梅', '女', 14],
]

total = [
    ['男（人）', '女（人）', '共（人）'],
    [2, 2, 4],
]


# 获取字符串长度，一个中文的长度为2
def len_byte(value):
    length = len(value)
    utf8_length = len(value.encode('utf-8'))
    length = (utf8_length - length) / 2 + length
    return int(length)


# 设置字体
font = xlwt.Font()
font.bold = True

# 设置边框
borders = xlwt.Borders()
borders.left = xlwt.Borders.THIN
borders.right = xlwt.Borders.THIN
borders.top = xlwt.Borders.THIN
borders.bottom = xlwt.Borders.THIN

# 设置居中
alignment = xlwt.Alignment()
alignment.horz = xlwt.Alignment.HORZ_CENTER  # 水平方向
alignment.vert = xlwt.Alignment.VERT_TOP  # 垂直方向

# 设置背景颜色
pattern = xlwt.Pattern()
pattern.pattern = xlwt.Pattern.SOLID_PATTERN
pattern.pattern_fore_colour = 3  # 背景颜色

# 定义不同的excel style
style1 = xlwt.XFStyle()
style1.font = font
style1.borders = borders
style1.alignment = alignment
style2 = xlwt.XFStyle()
style2.borders = borders
style2.alignment = alignment
style3 = xlwt.XFStyle()
style3.borders = borders
style3.alignment = alignment
style4 = xlwt.XFStyle()
style4.borders = borders
style4.font = font
style4.pattern = pattern
style4.alignment = alignment
workbook = xlwt.Workbook(encoding='utf-8')
worksheet = workbook.add_sheet('My Worksheet')

# 确定栏位宽度
col_width = []
for i in range(len(result)):
    for j in range(len(result[i])):
        if i == 0:
            col_width.append(len_byte(result[i][j]))
        else:
            if col_width[j] < len_byte(str(result[i][j])):
                col_width[j] = len_byte(result[i][j])

# 设置栏位宽度，栏位宽度小于10时候采用默认宽度
for i in range(len(col_width)):
    if col_width[i] > 10:
        worksheet.col(i).width = 256 * (col_width[i] + 1)

# 设置栏位高度
# tall_style = xlwt.easyxf('font:height 720;') #设置字体高度
# row0 = worksheet.row(0)
# row0.set_style(tall_style)


# excel内容写入
for i in range(len(result)):
    for j in range(len(result[i])):
        if i == 0:
            worksheet.write(i, j, label=result[i][j], style=style1)
        else:
            worksheet.write(i, j, label=result[i][j], style=style2)

# excel统计结果写入
for m in range(len(total)):
    for n in range(len(total[m])):
        if m == 0:
            worksheet.write(m + i + 3, n + 2, label=total[m][n], style=style4)
        else:
            worksheet.write(m + i + 3, n + 2, label=total[m][n], style=style2)

# 合并单元格
worksheet.write_merge(m + i + 4, m + i + 6, 2, 4, label='统计:       签名：', style=style3)
workbook.save('Excel_Workbook.xls')
