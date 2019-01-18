# -*- coding: utf-8 -*-

import os

def readfilename(path):
    global lstfile
    filelist = os.listdir(path)
    #print(filelist)
    for files in filelist:
        filename1 = os.path.splitext(files)[1]  # 读取文件后缀名
        filename0 = os.path.splitext(files)[0]  #读取文件名
        #print(filename0)

        if '.txt' in filename1:
            filename2 = filename0
            m = filename1 == '.txt'
            print(m)
            print(filename2)

    if m:
        listfile = filename2.split('_')
        #print("list_file = ", listfile)
        for i in range(len(listfile)):
            if 'ct' in listfile[i]:
                ct_value = listfile[i+1]
                print("ct = ", ct_value)


path = "../source/"
readfilename(path)

