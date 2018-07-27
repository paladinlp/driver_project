import h5py
import numpy as np
from sklearn.utils import shuffle
np.random.seed(2017)

X_train = []
X_test = []

for filename in ["save_h5/gap_model_DenseNet201.h5", "save_h5/gap_model_inceptionResNetV2.h5",'save_h5/gap_model_inceptionV3.h5']:
    with h5py.File(filename, 'r') as h:
        X_train.append(np.array(h['train']))
        X_test.append(np.array(h['valid']))
        y_train = np.array(h['train_label'])
        y_test = np.array(h['valid_label'])

X_train = np.concatenate(X_train, axis=1)
X_test = np.concatenate(X_test, axis=1)

#转换为独热编码
n_values = np.max(y_train)+1

y_train = np.eye(n_values)[y_train]
y_test = np.eye(n_values)[y_test]

X_train, y_train = shuffle(X_train, y_train)

from sklearn.model_selection import train_test_split



from keras.models import *
from keras.layers import *

np.random.seed(2017)

input_tensor = Input(X_train.shape[1:])
x = Dropout(0.5)(input_tensor)
x = Dense(10, activation='softmax')(x)
model = Model(input_tensor, x)

from keras.optimizers import Adam

model.compile(
    optimizer=Adam(
        lr=0.0001,
    ),
    loss='categorical_crossentropy',
    metrics=['accuracy'])

model.fit(X_train, y_train, batch_size=128, nb_epoch=1000, validation_split=0.2)

#输入最终的正确率：



predictions = [np.argmax(model.predict(np.expand_dims(tensor, axis=0))) for tensor in X_test]

# 报告测试准确率
test_accuracy = 100*np.sum(np.array(predictions)==np.argmax(y_test, axis=1))/len(predictions)
print('Test accuracy: %.4f%%' % test_accuracy)












from keras.applications import *
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.callbacks import ModelCheckpoint


def model_predict_to_h5(rootFilePath, funtion_name, imageSize,):

    model = load_model(rootFilePath+'/saved_models/weights.best.'+funtion_name+'.hdf5')
    datagen =None

    if funtion_name == 'DenseNet201':
        print('设置' + funtion_name + '模型图片处理方式')
        datagen = ImageDataGenerator(
            preprocessing_function=densenet.preprocess_input)
    if funtion_name == 'inceptionResNetV2':
        print('设置' + funtion_name + '模型图片处理方式')
        datagen = ImageDataGenerator(
            preprocessing_function=inception_resnet_v2.preprocess_input)
    if funtion_name == 'InceptionV3':
        print('设置' + funtion_name + '模型图片处理方式')
        datagen = ImageDataGenerator(
            preprocessing_function=inception_v3.preprocess_input)
    if funtion_name == 'Xception':
        print('设置' + funtion_name + '模型图片处理方式')
        datagen = ImageDataGenerator(
            preprocessing_function=xception.preprocess_input)
    if funtion_name == 'ResNet50':
        print('设置' + funtion_name + '模型图片处理方式')
        datagen = ImageDataGenerator(
            preprocessing_function=resnet50.preprocess_input)
    if funtion_name == 'VGG19':
        print('设置' + funtion_name + '模型图片处理方式')
        datagen = ImageDataGenerator(
            preprocessing_function=vgg19.preprocess_input)

    validation_generator = datagen.flow_from_directory(
        rootFilePath + '/valid',
        target_size=imageSize,
        batch_size=1,
        class_mode='categorical',
        shuffle=False)

    train_generator = datagen.flow_from_directory(
        rootFilePath+'/train',  # this is the target directory
        target_size=imageSize,  # all images will be resized to 150x150
        batch_size=1,
        class_mode='categorical',
        shuffle=False)

    test_generator = datagen.flow_from_directory(
        rootFilePath+'/test1',
        target_size=imageSize,
        shuffle=False,
        batch_size=1,
        class_mode=None)
    train = model.predict_generator(train_generator, train_generator.samples)
    valid = model.predict_generator(validation_generator, validation_generator.samples)
    test = model.predict_generator(test_generator, test_generator.samples)

    with h5py.File(rootFilePath + '/saved_h5/gap_model_' + funtion_name + '.h5') as h:
        h.create_dataset("train", data=train)
        h.create_dataset('valid', data=valid)
        h.create_dataset("test", data=test)
        h.create_dataset("train_label", data=train_generator.classes)
        h.create_dataset("valid_label", data=validation_generator.classes)

































def load_model_training(
        rootFilePath,
        funtion_name,
        imageSize,
        train_generator_batch_size,
        valid_generator_batch_size,
        fine_tuning_fit_pre_samples,
        fine_tuning_fit_epochs,
        fine_tuning_valid_steps,
        imagesize,
        learning_rate=0.0001,
        model_opened_layers=None,
        backFileModelPath = None):
    if not backFileModelPath:
        fileModelPath = rootFilePath + '/saved_models/weights.best.' + funtion_name + '.hdf5'
    else:
        fileModelPath = backFileModelPath
    print('本次训练，所利用的数据集为：'+rootFilePath+'所加载的迁移模型为 ：' + funtion_name + '!!!，模型文件具体名称为：'+fileModelPath)


    # 训练和验证生成器的batch_size
    TRAIN_GENERATOR_BATCH_SIZE = train_generator_batch_size
    VALID_GENERATOR_BATCH_SIZE = valid_generator_batch_size

    # 迁移模型放开之后的每轮训练样本数、训练轮数、测试样本数
    FINETUNING_FIT_PRE_SAMPLES = fine_tuning_fit_pre_samples  # ( 最好是用生成器样本数除以生成器的batchsize)
    FINETUNING_FIT_EPOCHS = fine_tuning_fit_epochs
    FINETUNING_VALID_STEPS = fine_tuning_valid_steps

    # 迁移模型从第几次开始开放，None代表全部开放
    MODEL_OPENED_LAYERS = model_opened_layers
    print("**********************************************************")
    # 加载对应模型
    if os.path.exists(fileModelPath):
        model = load_model(fileModelPath)
        print('加载模型成功！')
    else:
        print('没有找到指定模型文件，程序结束')
        return 0

    #建立图片生成器，利用keras自带的，可以节约机器资源，提高效率, 要根据调用的模型，采用不同的书里函数
    datagen = None
    print("**********************************************************")
    if funtion_name == 'DenseNet201':
        print('设置' + funtion_name + '模型图片处理方式')
        datagen = ImageDataGenerator(
            preprocessing_function=densenet.preprocess_input)
    if funtion_name == 'inceptionResNetV2':
        print('设置' + funtion_name + '模型图片处理方式')
        datagen = ImageDataGenerator(
            preprocessing_function=inception_resnet_v2.preprocess_input)
    if funtion_name == 'InceptionV3':
        print('设置' + funtion_name + '模型图片处理方式')
        datagen = ImageDataGenerator(
            preprocessing_function=inception_v3.preprocess_input)
    if funtion_name == 'Xception':
        print('设置' + funtion_name + '模型图片处理方式')
        datagen = ImageDataGenerator(
            preprocessing_function=xception.preprocess_input)
    if funtion_name == 'ResNet50':
        print('设置' + funtion_name + '模型图片处理方式')
        datagen = ImageDataGenerator(
            preprocessing_function=resnet50.preprocess_input)
    if funtion_name == 'VGG19':
        print('设置' + funtion_name + '模型图片处理方式')
        datagen = ImageDataGenerator(
            preprocessing_function=vgg19.preprocess_input)
    if funtion_name == 'VGG16':
        print('设置' + funtion_name + '模型图片处理方式')
        datagen = ImageDataGenerator(
            preprocessing_function=vgg16.preprocess_input)
    if funtion_name == 'NASNetLarge':
        print('设置' + funtion_name + '模型图片处理方式')
        datagen = ImageDataGenerator(
            preprocessing_function=nasnet.preprocess_input)
    if funtion_name == 'NASNetMobile':
        print('设置' + funtion_name + '模型图片处理方式')
        datagen = ImageDataGenerator(
            preprocessing_function=nasnet.preprocess_input)
        # 设置图片生成器的路径已经其他相关信息
    print("**********************************************************")
    print('读取训练与验证图片！')

    train_generator = datagen.flow_from_directory(
        rootFilePath + '/train',  # this is the target directory
        target_size=imageSize,  # all images will be resized to 150x150
        batch_size=TRAIN_GENERATOR_BATCH_SIZE,
        class_mode='categorical')  # since we use binary_crossentropy loss, we need binary labels

    # this is a similar generator, for validation data
    validation_generator = datagen.flow_from_directory(
        rootFilePath + '/valid',
        target_size=imageSize,
        batch_size=VALID_GENERATOR_BATCH_SIZE,
        class_mode='categorical')

    # 设置放开训练层数：
    if MODEL_OPENED_LAYERS:
        print("**********************************************************")
        print('放开训练层数，本次训练从 %d 层之后开始放开！' % MODEL_OPENED_LAYERS)
    else:
        print("**********************************************************")
        print('本次训练迁移模型所有层均已经放开')

    for layer in model.layers[:MODEL_OPENED_LAYERS]:
        layer.trainable = False
    for layer in model.layers[MODEL_OPENED_LAYERS:]:
        layer.trainable = True

    model.evaluate_generator(validation_generator,
                             steps = FINETUNING_VALID_STEPS,
                             pickle_safe=False)

    from keras.optimizers import Adam
    model.compile(
        optimizer=Adam(
            lr=learning_rate,
        ),
        loss='categorical_crossentropy',
        metrics=['accuracy'])
    # 设置权重存储条件

    checkpointer = ModelCheckpoint(filepath=rootFilePath + '/saved_models/weights.retrained.best.' + funtion_name+'.hdf5',
                                   verbose=1, save_best_only=True)
    # 可以驯练放开之后的迁移模型，保存准确率最高的一个模型
    print("**********************************************************")
    print('开始fine tuning!!!')
    print('\n')
    model.fit_generator(
        train_generator,
        steps_per_epoch=FINETUNING_FIT_PRE_SAMPLES,
        epochs=FINETUNING_FIT_EPOCHS,
        validation_data=validation_generator,
        validation_steps=FINETUNING_VALID_STEPS,
        callbacks=[checkpointer],
        verbose=1)
    print(" 本次在数据集%s上的重训练结束，最优模型已经存储！！！" % rootFilePath)
