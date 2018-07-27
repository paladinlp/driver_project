# -*-coding:utf-8-*-
__author__ = 'paladinlp'
__date__ = '2018/5/19 14:03'


CHECK_DRIVERS_NUMBER = 3
import os
import pandas as pd

from tqdm import tqdm
rootPath = 'c:\data\imgs'
for i in range(10):
    if not os.path.exists(rootPath + '\\valid' + '\\c%d' % i):
        os.mkdir(rootPath + '\\valid' + '\\c%d' % i)

import random




#回滚valid

# for i, filePath in tqdm(enumerate(os.listdir(rootPath + '\\valid'))):
#     files = os.listdir(rootPath + '\\valid\\'+filePath)
#     for file in files:
#         targetFile = (rootPath + '\\train' + '\\c%d\\' + file) % (i )
#         sourceFile = (rootPath + '\\valid' + '\\c%d\\'+ file) % (i )
#         if (not os.path.exists(targetFile)) and os.path.exists(sourceFile):
#             os.rename(sourceFile, targetFile)

#增加valid

random.seed(37)

drivers_to_files = pd.read_csv('driver_imgs_list.csv')
check_drivers = drivers_to_files.drop_duplicates(['subject'])['subject'].sample(n=CHECK_DRIVERS_NUMBER, random_state=37)
for driver in check_drivers:
    check_drivers_dataFrame = drivers_to_files.loc[drivers_to_files['subject'] == driver]
    for indexs in check_drivers_dataFrame.index:
        filePath = check_drivers_dataFrame.loc[indexs]
        sourceFile = (rootPath + '\\train' + '\\' + filePath['classname'] + '\\' + filePath['img'])
        targetFile = (rootPath + '\\valid' + '\\' + filePath['classname'] + '\\' + filePath['img'])
        if (not os.path.exists(targetFile)) and os.path.exists(sourceFile):
            os.rename(sourceFile, targetFile)



#回滚valid
# for i, filePath in tqdm(enumerate(os.listdir(rootPath + '\\valid'))):
#     files = os.listdir(rootPath + '\\valid\\'+filePath)
#     for file in files:
#         targetFile = (rootPath + '\\train' + '\\c%d\\' + file) % (i )
#         sourceFile = (rootPath + '\\valid' + '\\c%d\\'+ file) % (i )
#         if (not os.path.exists(targetFile)) and os.path.exists(sourceFile):
#             os.rename(sourceFile, targetFile)

