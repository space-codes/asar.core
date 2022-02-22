# -*- coding: utf-8 -*-

import os
import sqlite3

conn = sqlite3.connect('arabic-manuscripts.db')
folder='segdata'
os.chdir(folder)
label=0
for foldername in sorted(os.listdir(os.getcwd())):
    l=len(str(label))
    z=5-l
    a=''
    for i in range(z):
        a+='0'

    labelValue = a+str(label)
    os.rename(foldername,labelValue)
    # seed it into database
    conn.execute("INSERT INTO words (word,label) VALUES (?,?)", (foldername, int(labelValue)));
    label=label+1

conn.commit()
conn.close()
