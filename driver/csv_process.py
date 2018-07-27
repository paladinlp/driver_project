import pandas as pd
import cv2
import numpy as np
import os
import sys
import pickle
from tqdm import tqdm

colunms=['c0','c1','c2','c3','c4','c5','c6','c7','c8','c9']
csv_path = 'csv/2018-06-29-13-20-09_all.csv'
files = pd.read_csv(csv_path)
modify_files = files.copy()

with open('pkl\\hashList_50.pk', 'rb') as f:
    hashList = pickle.load(f)

for file in tqdm(modify_files['img']):
    temp_list = [temp[0] for temp in  hashList[file] if temp[1] == 1]
    checked_list = []
    if(len(temp_list)!=1):
        old_item_typed = np.array(modify_files[modify_files['img'] == file][colunms]).argmax()
        checked_list    =[item for item in temp_list if old_item_typed == np.array(modify_files[modify_files['img'] == item][colunms]).argmax()]
        if (len(checked_list)!=0):
            s = (modify_files.loc[modify_files['img'].isin(checked_list)][colunms])
            modify_files.loc[modify_files['img'] == file, colunms] = np.array(s.mean())



import time
csv_columns =['img','c0','c1','c2','c3','c4','c5','c6','c7','c8','c9']
fileTime = time.strftime("csv\\%Y-%m-%d-%H-%M-%S", time.localtime())+'_modiyf.csv'
modify_files.to_csv(fileTime,index=False)



