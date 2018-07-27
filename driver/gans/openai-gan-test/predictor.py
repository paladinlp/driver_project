import tensorflow as tf
from model import save_images
import time
import pandas as pd
import numpy as np
from  tqdm import tqdm
from model import read_and_decode_with_labels




pd_win1 = pd.read_csv('C:/Users\lp\Desktop\csv/last_2018-06-25-15-23-13_win1.csv')
pd_title = (pd_win1['img'])

def creat_csv1(Y_predict,title):
    import time
    fileTime = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    # 写入CSV文件
    df_predict = pd.DataFrame(Y_predict, columns=['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9'])
    df_name_predict = title
    df_predict.insert(0, 'img', df_name_predict)
    fileName = 'C:\\Users\lp\Desktop\csv\\' + fileTime + '_win2' + '.csv'
    fileName1 = 'csv/' + fileTime + '_' + 'all.csv'
    df_predict.to_csv(fileName1, index=False, float_format='%.17f')
    print('****************************************************')
    print('CSV已经生成完毕，名称为：%s' % (fileName1))
    return fileName



def predictor(self, config): #79726
    """predictor DCGAN"""

    filename_queue = tf.train.string_input_producer(['imagenet_forTest128.tfrecords'])
    # filename_queue = tf.train.string_input_producer(['imagenet_train_labeled_forTest128.tfrecords'])
    get_image, get_label = read_and_decode_with_labels(filename_queue)

    images, sparse_labels = tf.train.batch([get_image, get_label],
                                                   batch_size =2 * self.batch_size,
                                                   num_threads=1,
                                                   name='test_images')

    _, _, _, _, _, class_logits = self.discriminator(images, reuse=True, prefix="joint")
    real_class_logits, gan_logits = tf.split(class_logits, [10, 1], 1)

    prediction = tf.nn.softmax(real_class_logits)

    # class_entropy = tf.nn.softmax(class_logits)
    #
    # prediction, _ = tf.split(class_entropy, [10, 1], 1)



    self.sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    prob_test = np.random.rand(1, 10)
    for i in tqdm(range(9966)):

        prob = self.sess.run([prediction])
        prob = np.array(prob).reshape(2*self.batch_size, 10)
        prob_test = np.row_stack((prob_test, prob))
        # prob_test = np.row_stack((s, prob))

    coord.request_stop()
    coord.join(threads)

    prob_test = np.delete(prob_test, 0, 0)
    prob_test = np.delete(prob_test, -1, 0)
    prob_test = np.delete(prob_test, -1, 0)

    fileName = creat_csv1(prob_test, pd_title)





