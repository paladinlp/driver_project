# -*- coding: utf-8 -*-
"""
@author: Giba1
"""
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import sys
from skimage import io, transform
import matplotlib.animation as animation
import os
import imageio


train = pd.read_csv('C:\\Users\lp\Desktop\R Language\dog_cat\\2000\driver_imgs_list.csv' )
train['id'] = range(train.shape[0] )
fig = plt.figure(5)
subj = np.unique( train['subject'])[0]

for subj in np.unique( train['subject'])[:2]:

    imagem = train[ train['subject']==subj ]
    
    imgs = []
    t = imagem.values[0]
    for t in imagem.values:
        img = cv2.imread('C:\data\data\\train\\'+t[1]+'\\'+t[2],3)
        img = cv2.resize(img, (160, 120))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgs.append( img )
        
    ax = fig.add_subplot(111)
    ax.set_axis_off()
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1,wspace=None, hspace=None)  # removes white border
    # fname = 'MOVIE_subject_'+subj+'.mp4'
    imgs = [ (ax.imshow(img),
              ax.set_title(t[0]),
              ax.annotate(n_img,(5,5))) for n_img, img in enumerate(imgs) ]
    img_anim = animation.ArtistAnimation(fig, imgs, interval=125,
                                repeat_delay=None, blit=False)
    # print('Writing:', fname)
    # writer = animation.FFMpegWriter()
    # img_anim.save(fname, writer=writer)
    plt.show()
    # imageio.mimsave(subj, imgs, 'GIF', duration=0.1)

    # img_root = 'movie'  # 这里写你的文件夹路径，比如：/home/youname/data/img/,注意最后一个文件夹要有斜杠
    # fps = 24  # 保存视频的FPS，可以适当调整
    #
    # # 可以用(*'DVIX')或(*'X264'),如果都不行先装ffmepg: sudo apt-get install ffmepg
    # fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    # videoWriter = cv2.VideoWriter('saveVideo.avi', fourcc, fps, (160, 120))  # 最后一个是保存图片的尺寸
    #
    # for image in imgs:
    #
    #     videoWriter.write(image)
    # videoWriter.release()
    # break


print ('Now relax and watch some movies!!!')


# Any results you write to the current directory are saved as output.