# -*- coding: utf-8 -*-
"""
Created on Tue Sep  4 09:13:50 2018

@author: Moodliu
"""

import csv, os

while True :
	# 開啟 CSV 檔案
	with open('test.csv', 'r+' , newline='') as csvfile:
	
	  # 讀取 CSV 檔案內容
	  dic = csv.DictReader(csvfile)
	  
	  with open('ans.csv', 'w+' , newline='') as csvfile:
	    writer = csv.DictWriter(csvfile, fieldnames=dic.fieldnames)
	    writer.writeheader()
	    name = str(input('Enter student id : ') )
	    for row in dic:
	      if row['學號'] == name :
	        row['未到'] = 'F'
	        row['實到'] = 'T'
	        writer.writerow(row)
	      
	      else :
	        if row['未到'] == 'T':
	          row['未到'] = 'T'
	          row['實到'] = 'F'
	        writer.writerow(row)
	      print( row )
	      
	  csvfile.close()
	os.remove("test.csv")
	os.rename("ans.csv", "test.csv" )
