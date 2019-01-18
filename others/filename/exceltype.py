style = xlwt.XFStyle()#格式信息
font = xlwt.Font()#字体基本设置
font.name = u'微软雅黑'
font.color = 'black'
font.height= 220 #字体大小，220就是11号字体，大概就是11*20得来的吧
style.font = font
alignment = xlwt.Alignment() # 设置字体在单元格的位置
alignment.horz = xlwt.Alignment.HORZ_CENTER  # May be: HORZ_GENERAL, HORZ_LEFT, HORZ_CENTER, HORZ_RIGHT, HORZ_FILLED, HORZ_JUSTIFIED, HORZ_CENTER_ACROSS_SEL, HORZ_DISTRIBUTED
alignment.vert = xlwt.Alignment.VERT_CENTER  # May be: VERT_TOP, VERT_CENTER, VERT_BOTTOM, VERT_JUSTIFIED, VERT_DISTRIBUTED
book = xlwt.Workbook()  # 创建excel文件
style.alignment = alignment  # Add Alignment to Style
font.bold = True
style.font = font
border = xlwt.Borders()  #给单元格加框线
border.left = xlwt.Borders.THIN  #左
border.top=xlwt.Borders.THIN     #上
border.right=xlwt.Borders.THIN   #右
border.bottom=xlwt.Borders.THIN  #下
border.left_colour = 0x40  #设置框线颜色，0x40是黑色，颜色真的巨多，都晕了
border.right_colour = 0x40
border.top_colour = 0x40
border.bottom_colour = 0x40
style.borders = border
#写入sheet
wb = xlwt.Workbook()
ws = wb.add_sheet('sheet1')
ws.write(row,col,value,style)#这样就可以写入自己想要的格式了
wb.save('D:\\test.xls')



'''
(二） 关于写excel的格式控制，比如颜色等等
import xlwt
from datetime import datetime
  
font0 = xlwt.Font()
font0.name = 'Times New Roman'
font0.colour_index = 2
font0.bold = True
  
style0 = xlwt.XFStyle()
style0.font = font0
  
style1 = xlwt.XFStyle()
style1.num_format_str = 'D-MMM-YY'
  
wb = xlwt.Workbook()
ws = wb.add_sheet('A Test Sheet')
  
ws.write(0, 0, 'Test', style0)
ws.write(1, 0, datetime.now(), style1)
ws.write(2, 0, 1)
ws.write(2, 1, 1)
ws.write(2, 2, xlwt.Formula("A3+B3"))
  
wb.save('example.xls')

Examples Generating Excel Documents Using Python’s xlwt
Here are some simple examples using Python’s xlwt library to dynamically generate Excel documents.

Please note a useful alternative may be ezodf, which allows you to generate ODS (Open Document Spreadsheet) files for LibreOffice / OpenOffice. You can check them out at:http://packages.python.org/ezodf/index.html

The Simplest Example
import xlwt
workbook = xlwt.Workbook(encoding = 'ascii')
worksheet = workbook.add_sheet('My Worksheet')
worksheet.write(0, 0, label = 'Row 0, Column 0 Value')
workbook.save('Excel_Workbook.xls')

Formatting the Contents of a Cell
import xlwt
workbook = xlwt.Workbook(encoding = 'ascii')
worksheet = workbook.add_sheet('My Worksheet')
font = xlwt.Font() # Create the Font
font.name = 'Times New Roman'
font.bold = True
font.underline = True
font.italic = True
style = xlwt.XFStyle() # Create the Style
style.font = font # Apply the Font to the Style
worksheet.write(0, 0, label = 'Unformatted value')
worksheet.write(1, 0, label = 'Formatted value', style) # Apply the Style to the Cell
workbook.save('Excel_Workbook.xls')

Attributes of the Font Object
font.bold = True # May be: True, False
font.italic = True # May be: True, False
font.struck_out = True # May be: True, False
font.underline = xlwt.Font.UNDERLINE_SINGLE # May be: UNDERLINE_NONE, UNDERLINE_SINGLE, UNDERLINE_SINGLE_ACC, UNDERLINE_DOUBLE, UNDERLINE_DOUBLE_ACC
font.escapement = xlwt.Font.ESCAPEMENT_SUPERSCRIPT # May be: ESCAPEMENT_NONE, ESCAPEMENT_SUPERSCRIPT, ESCAPEMENT_SUBSCRIPT
font.family = xlwt.Font.FAMILY_ROMAN # May be: FAMILY_NONE, FAMILY_ROMAN, FAMILY_SWISS, FAMILY_MODERN, FAMILY_SCRIPT, FAMILY_DECORATIVE
font.charset = xlwt.Font.CHARSET_ANSI_LATIN # May be: CHARSET_ANSI_LATIN, CHARSET_SYS_DEFAULT, CHARSET_SYMBOL, CHARSET_APPLE_ROMAN, CHARSET_ANSI_JAP_SHIFT_JIS, CHARSET_ANSI_KOR_HANGUL, CHARSET_ANSI_KOR_JOHAB, CHARSET_ANSI_CHINESE_GBK, CHARSET_ANSI_CHINESE_BIG5, CHARSET_ANSI_GREEK, CHARSET_ANSI_TURKISH, CHARSET_ANSI_VIETNAMESE, CHARSET_ANSI_HEBREW, CHARSET_ANSI_ARABIC, CHARSET_ANSI_BALTIC, CHARSET_ANSI_CYRILLIC, CHARSET_ANSI_THAI, CHARSET_ANSI_LATIN_II, CHARSET_OEM_LATIN_I
font.colour_index = ?
font.get_biff_record = ?
font.height = 0x00C8 # C8 in Hex (in decimal) = 10 points in height.
font.name = ?
font.outline = ?
font.shadow = ?

Setting the Width of a Cell
import xltw
workbook = xlwt.Workbook()
worksheet = workbook.add_sheet('My Sheet')
worksheet.write(0, 0, 'My Cell Contents')
worksheet.col(0).width = 3333 # 3333 = 1" (one inch).
workbook.save('Excel_Workbook.xls')

Entering a Date into a Cell
import xlwt
import datetime
workbook = xlwt.Workbook()
worksheet = workbook.add_sheet('My Sheet')
style = xlwt.XFStyle()
style.num_format_str = 'M/D/YY' # Other options: D-MMM-YY, D-MMM, MMM-YY, h:mm, h:mm:ss, h:mm, h:mm:ss, M/D/YY h:mm, mm:ss, [h]:mm:ss, mm:ss.0
worksheet.write(0, 0, datetime.datetime.now(), style)
workbook.save('Excel_Workbook.xls')

Adding a Formula to a Cell
import xlwt
workbook = xlwt.Workbook()
worksheet = workbook.add_sheet('My Sheet')
worksheet.write(0, 0, 5) # Outputs 5
worksheet.write(0, 1, 2) # Outputs 2
worksheet.write(1, 0, xlwt.Formula('A1*B1')) # Should output "10" (A1[5] * A2[2])
worksheet.write(1, 1, xlwt.Formula('SUM(A1,B1)')) # Should output "7" (A1[5] + A2[2])
workbook.save('Excel_Workbook.xls')

Adding a Hyperlink to a Cell
import xlwt
workbook = xlwt.Workbook()
worksheet = workbook.add_sheet('My Sheet')
worksheet.write(0, 0, xlwt.Formula('HYPERLINK("http://www.google.com";"Google")')) # Outputs the text "Google" linking to http://www.google.com
workbook.save('Excel_Workbook.xls')

Merging Columns and Rows
import xlwt
workbook = xlwt.Workbook()
worksheet = workbook.add_sheet('My Sheet')
worksheet.write_merge(0, 0, 0, 3, 'First Merge') # Merges row 0's columns 0 through 3.
font = xlwt.Font() # Create Font
font.bold = True # Set font to Bold
style = xlwt.XFStyle() # Create Style
style.font = font # Add Bold Font to Style
worksheet.write_merge(1, 2, 0, 3, 'Second Merge', style) # Merges row 1 through 2's columns 0 through 3.
workbook.save('Excel_Workbook.xls')

Setting the Alignment for the Contents of a Cell
import xlwt
workbook = xlwt.Workbook()
worksheet = workbook.add_sheet('My Sheet')
alignment = xlwt.Alignment() # Create Alignment
alignment.horz = xlwt.Alignment.HORZ_CENTER # May be: HORZ_GENERAL, HORZ_LEFT, HORZ_CENTER, HORZ_RIGHT, HORZ_FILLED, HORZ_JUSTIFIED, HORZ_CENTER_ACROSS_SEL, HORZ_DISTRIBUTED
alignment.vert = xlwt.Alignment.VERT_CENTER # May be: VERT_TOP, VERT_CENTER, VERT_BOTTOM, VERT_JUSTIFIED, VERT_DISTRIBUTED
style = xlwt.XFStyle() # Create Style
style.alignment = alignment # Add Alignment to Style
worksheet.write(0, 0, 'Cell Contents', style)
workbook.save('Excel_Workbook.xls')

Adding Borders to a Cell
# Please note: While I was able to find these constants within the source code, on my system (using LibreOffice,) I was only presented with a solid line, varying from thin to thick; no dotted or dashed lines.
import xlwt
workbook = xlwt.Workbook()
worksheet = workbook.add_sheet('My Sheet')
borders = xlwt.Borders() # Create Borders
borders.left = xlwt.Borders.DASHED # May be: NO_LINE, THIN, MEDIUM, DASHED, DOTTED, THICK, DOUBLE, HAIR, MEDIUM_DASHED, THIN_DASH_DOTTED, MEDIUM_DASH_DOTTED, THIN_DASH_DOT_DOTTED, MEDIUM_DASH_DOT_DOTTED, SLANTED_MEDIUM_DASH_DOTTED, or 0x00 through 0x0D.
borders.right = xlwt.Borders.DASHED
borders.top = xlwt.Borders.DASHED
borders.bottom = xlwt.Borders.DASHED
borders.left_colour = 0x40
borders.right_colour = 0x40
borders.top_colour = 0x40
borders.bottom_colour = 0x40
style = xlwt.XFStyle() # Create Style
style.borders = borders # Add Borders to Style
worksheet.write(0, 0, 'Cell Contents', style)
workbook.save('Excel_Workbook.xls')

Setting the Background Color of a Cell
import xlwt
workbook = xlwt.Workbook()
worksheet = workbook.add_sheet('My Sheet')
pattern = xlwt.Pattern() # Create the Pattern
pattern.pattern = xlwt.Pattern.SOLID_PATTERN # May be: NO_PATTERN, SOLID_PATTERN, or 0x00 through 0x12
pattern.pattern_fore_colour = 5 # May be: 8 through 63. 0 = Black, 1 = White, 2 = Red, 3 = Green, 4 = Blue, 5 = Yellow, 6 = Magenta, 7 = Cyan, 16 = Maroon, 17 = Dark Green, 18 = Dark Blue, 19 = Dark Yellow , almost brown), 20 = Dark Magenta, 21 = Teal, 22 = Light Gray, 23 = Dark Gray, the list goes on...
style = xlwt.XFStyle() # Create the Pattern
style.pattern = pattern # Add Pattern to Style
worksheet.write(0, 0, 'Cell Contents', style)
workbook.save('Excel_Workbook.xls')

TODO: Things Left to Document
- Panes -- separate views which are always in view
- Border Colors (documented above, but not taking effect as it should)
- Border Widths (document above, but not working as expected)
- Protection
- Row Styles
- Zoom / Manification
- WS Props?
Source Code for reference available at: https://secure.simplistix.co.uk/svn/xlwt/trunk/xlwt/


功能描述：首先生成几个测试用的Excel文件，然后批量修改这些文件的格式，把表头加粗并设置为黑体，其他行字体为宋体，设置奇偶行颜色不同，并设置偶数行为从红到蓝的渐变背景色填充。

from random import sample

import openpyxl

from openpyxl.styles import Font, colors



def generateXlsx(num):

    for i in range(num):

        wb = openpyxl.Workbook()

        ws = wb.worksheets[0]

        # 添加表头

        ws.append(['字段'+str(_) for _ in range(1,6)])

        # 添加随机数据

        for _ in range(10):

            ws.append(sample(range(10000), 5))

        wb.save(str(i)+'.xlsx')



def batchFormat(num):

    for i in range(num):

        fn = str(i)+'.xlsx'

        wb = openpyxl.load_workbook(fn)

        ws = wb.worksheets[0]

        for irow, row in enumerate(ws.rows, start=1):

            if irow == 1:

                # 表头加粗、黑体

                font = Font('黑体', bold=True)

            elif irow%2 == 0:

                # 偶数行红色，宋体

                font = Font('宋体', color=colors.RED)

            else:

                # 奇数行浅蓝色，宋体

                font = Font('宋体', color='00CCFF')

            for cell in row:

                cell.font = font

                # 偶数行添加背景填充色，从红到蓝渐变

                if irow%2 == 0:

                    cell.fill = openpyxl.styles.fills.GradientFill(stop=['FF0000', '0000FF'])

        # 另存为新文件

        wb.save('new'+fn)



generateXlsx(5)

batchFormat(5)
--------------------- 
作者：dongfuguo 
来源：CSDN 
原文：https://blog.csdn.net/dongfuguo/article/details/78188388 
版权声明：本文为博主原创文章，转载请附上博文链接！

'''