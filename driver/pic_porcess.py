import cv2
import numpy as np
import os
import sys
import pandas as pd
from tqdm import tqdm


def flatten(a):
    if not isinstance(a, (list, )):
        return [a]
    else:
        b = []
        for item in a:
            b += flatten(item)
    return b


def pHash(imgfile):
    """get image pHash value"""
    #加载并调整图片为32x32灰度图片
    img=cv2.imread(imgfile, 0)
    img=cv2.resize(img,(64,64),interpolation=cv2.INTER_LANCZOS4    )

        #创建二维列表
    h, w = img.shape[:2]
    vis0 = np.zeros((h,w), np.float32)
    vis0[:h,:w] = img       #填充数据

    #二维Dct变换
    vis1 = cv2.dct(cv2.dct(vis0))
    #cv.SaveImage('a.jpg',cv.fromarray(vis0)) #保存图片
    vis1.resize(32,32)

    #把二维list变成一维list
    img_list=flatten(vis1.tolist())

    #计算均值
    avg = sum(img_list)*1./len(img_list)
    avg_list = ['0' if i<avg else '1' for i in img_list]

    #得到哈希值
    return ''.join(['%x' % int(''.join(avg_list[x:x+4]),2) for x in range(0,32*32,4)])
def hammingDist(s1, s2):
    assert len(s1) == len(s2)
    return sum([ch1 != ch2 for ch1, ch2 in zip(s1, s2)])





def dHash(img):
    img = cv2.imread(img, 0)
    #缩放8*8
    img=cv2.resize(img,(9,8),interpolation=cv2.INTER_CUBIC)
    #转换灰度图
    gray=img
    hash_str=''
    #每行前一个像素大于后一个像素为1，相反为0，生成哈希
    for i in range(8):
        for j in range(8):
            if   gray[i,j]>gray[i,j+1]:
                hash_str=hash_str+'1'
            else:
                hash_str=hash_str+'0'
    return hash_str

def aHash(img):
    img = cv2.imread(img, 0)
    #缩放为8*8
    img=cv2.resize(img,(8,8),interpolation=cv2.INTER_CUBIC)
    #转换为灰度图
    gray=img
    #s为像素和初值为0，hash_str为hash值初值为''
    s=0
    hash_str=''
    #遍历累加求像素和
    for i in range(8):
        for j in range(8):
            s=s+gray[i,j]
    #求平均灰度
    avg=s/64
    #灰度大于平均值为1相反为0生成图片的hash值
    for i in range(8):
        for j in range(8):
            if  gray[i,j]>avg:
                hash_str=hash_str+'1'
            else:
                hash_str=hash_str+'0'
    return hash_str




rootPicPath = 'C:\\data\\data\\test1\\test\\'
HASH1=aHash(rootPicPath+'img_58.jpg')
HASH2=aHash(rootPicPath+'img_58.jpg')
# HASH2=pHash(rootPicPath+'img_58.jpg')
out_score = 1 - hammingDist(HASH1, HASH2)*1. / (32*32/4)

# print(out_score)


scoreList = []
hash_list = []
# for file in tqdm(os.listdir(rootPicPath)):
#     HASH = aHash(rootPicPath + file)
#     hash_list.append((file, HASH))

import pickle
# with open('hash.pk', 'wb') as f:
#     pickle.dump(hash_list, f)


# with open('hash.pk', 'rb') as f:
#     hash_list = pickle.load(f)

# for hash in hash_list:
#     out_score = 1 - hammingDist(HASH1, hash[1]) * 1. / (32 * 32 / 4)
#     scoreList.append((hash[0],out_score))
# scoreList = sorted(scoreList, key = lambda x:x[1],reverse = True)
# print(scoreList[:30])
with open('hash.pk', 'rb') as f:
    hash_list = pickle.load(f)

files_hashList ={}
for file in tqdm(os.listdir(rootPicPath)):
    scoreList =[]
    HASH1 = aHash(rootPicPath + file)
    for hash in hash_list:
        out_score = 1 - hammingDist(HASH1, hash[1]) * 1. / (32 * 32 / 4)
        if out_score >=0.99:
            scoreList.append((hash[0], out_score))
    scoreList = sorted(scoreList, key=lambda x: x[1], reverse=True)
    files_hashList[file] = (scoreList[:50])




# with open('hashList.pk', 'wb') as f:
#     pickle.dump(files_hashList, f)

# with open('hashList.pk', 'rb') as f:
#     hashList = pickle.load(f)
# print(hashList)

colunms=['c0','c1','c2','c3','c4','c5','c6','c7','c8','c9']
csv_path = '2018-06-28-17-01-21_all.csv'
files = pd.read_csv(csv_path)
modify_files = files.copy()


def change(x):

        return 0.0



s = modify_files.ix[:, colunms]
s[s < 0.0000001] = 0.0

modify_files.ix[:, colunms] = s




# for file in tqdm(modify_files['img']):
#     temp_list = [temp[0] for temp in  hashList[file] if temp[1] == 1]
#     checked_list = []
#     if(len(temp_list)!=1):
#         old_item_typed = np.array(modify_files[modify_files['img'] == file][colunms]).argmax()
#         checked_list =[item for item in temp_list if old_item_typed == np.array(modify_files[modify_files['img'] == item][colunms]).argmax()]
#         if (len(checked_list)!=0):
#             s = (modify_files.loc[modify_files['img'].isin(checked_list)][colunms])
#             modify_files.loc[modify_files['img'] == file, colunms] = np.array(s.mean())




import time
csv_columns =['img','c0','c1','c2','c3','c4','c5','c6','c7','c8','c9']
fileTime =  time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())+'.csv'

# modify_files.to_csv(fileTime,index=False)





# check_list =[]
# for item in scoreList[:30]:
#     # if item[1] == 1.0:
#     #     check_list.append(np.array(files[files['img']==item[0]][colunms]))
#     check_list.append(np.array(files[files['img'] == item[0]][colunms]))
# check_list = (np.array(check_list).reshape(-1,10))

# print(check_list)
# print (np.argmax(check_list,axis=1))
#     scoreList.append((file,out_score))
# scoreList = sorted(scoreList, key = lambda x:x[1],reverse = True)
# print(scoreList[:30])
# out_score = 1 - hammingDist(HASH1, HASH2) * 1. / (32 * 32 / 4)







# print ( np.array([int(num) for num in hash_list[1][1]]))
# pic_feature = np.array([int(num) for num in hash_list[0][1]])


# for i,hash in tqdm(enumerate(hash_list)):
#     if i!=0:
#         pic_feature=np.row_stack((pic_feature,np.array([int(num) for num in hash_list[i][1]])))

# with open('feature.pk', 'wb') as f:
#     pickle.dump(pic_feature, f)
# with open('feature.pk', 'rb') as f:
#     feature = pickle.load(f)



# from sklearn.cluster import KMeans
# from sklearn.externals import joblib
#
# clf = KMeans(n_clusters=100)
# s = clf.fit(feature)
#
# joblib.dump(clf , 'c:/km.pkl')


