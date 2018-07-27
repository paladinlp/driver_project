# -*-coding:utf-8-*-
__author__ = 'paladinlp'
__date__ = '2018/5/19 15:25'



from keras.applications import *
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.callbacks import ModelCheckpoint

def inceptionResNetV2_own():
    print('inceptionResNetV2!!!')
    imageSize = (299, 299)
    # 训练和验证生成器的batch_size
    TRAIN_GENERATOR_BATCH_SIZE = 32
    VALID_GENERATOR_BATCH_SIZE = 1


    # FC训练时的每轮样本数，训练轮数，测试样本数
    FC_FIT_PRE_SAMPLES =200
    FC_FIT_EPOCHS =1
    FC_VALID_STEPS = 100

    #迁移模型放开之后的每轮训练样本数、训练轮数、测试样本数
    FINETUNING_FIT_PRE_SAMPLES =200#( 最好是用生成器样本数除以生成器的batchsize)
    FINETUNING_FIT_EPOCHS = 10
    FINETUNING_VALID_STEPS = 100

    #迁移模型从第几次开始开放，None代表全部开放

    MODEL_OPENED_LAYERS = 100


    base_model = InceptionResNetV2(weights='imagenet', include_top=False)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dense(1024, activation='relu')(x)
    # and a logistic layer -- let's say we have 200 classes
    predictions = Dense(10, activation='softmax')(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)
    print(len(model.layers))
    for layer in base_model.layers:
        layer.trainable = False

    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])


    train_datagen = ImageDataGenerator(
        preprocessing_function=inception_resnet_v2.preprocess_input)
    valid_datagen = ImageDataGenerator(
        preprocessing_function=inception_resnet_v2.preprocess_input)


    train_generator = train_datagen.flow_from_directory(
        'c:\data\imgs\\train',  # this is the target directory
        target_size=imageSize,  # all images will be resized to 150x150
        batch_size=TRAIN_GENERATOR_BATCH_SIZE,
        class_mode='categorical')  # since we use binary_crossentropy loss, we need binary labels

    # this is a similar generator, for validation data
    validation_generator = valid_datagen.flow_from_directory(
        'c:\data\imgs\\valid',
        target_size=imageSize,
        batch_size=VALID_GENERATOR_BATCH_SIZE,
        class_mode='categorical')

    model.fit_generator(
        train_generator,
        steps_per_epoch=FC_FIT_PRE_SAMPLES,
        epochs=FC_FIT_EPOCHS,
        validation_data=validation_generator,
        validation_steps=FC_VALID_STEPS)

    #设置放开训练层数：
    for layer in model.layers[:MODEL_OPENED_LAYERS]:
        layer.trainable = False

    for layer in model.layers[MODEL_OPENED_LAYERS:]:
        layer.trainable = True


    from keras.optimizers import Adam
    model.compile(
        optimizer=Adam(
            lr=0.0001,
            ),
        loss='categorical_crossentropy',
        metrics=['accuracy'])

    checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.inceptionResNetV2.hdf5',
                                   verbose=1, save_best_only=True)
    model.fit_generator(
        train_generator,
        steps_per_epoch=FINETUNING_FIT_PRE_SAMPLES,
        epochs=FINETUNING_FIT_EPOCHS,
        validation_data=validation_generator,
        validation_steps=FINETUNING_VALID_STEPS,
        callbacks=[checkpointer],
        verbose=1)


