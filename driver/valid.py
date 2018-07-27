from keras.models import load_model
from keras.applications import *
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.callbacks import ModelCheckpoint
import h5py


def model_DenseNet201_predict():

        model_DenseNet201 = load_model('saved_models/weights.best.DenseNet201.hdf5')
        imageSize = (224, 224)
        valid_datagen_DenseNet201 = ImageDataGenerator(
                preprocessing_function=densenet.preprocess_input)
        validation_generator = valid_datagen_DenseNet201.flow_from_directory(
                'c:\data\imgs\\valid',
                target_size=imageSize,
                batch_size=1,
                class_mode='categorical',
                shuffle=False)

        train_datagen_DenseNet201 = ImageDataGenerator(
                preprocessing_function=inception_v3.preprocess_input)

        train_generator = train_datagen_DenseNet201.flow_from_directory(
                'c:\data\imgs\\train',  # this is the target directory
                target_size=imageSize,  # all images will be resized to 150x150
                batch_size=1,
                class_mode='categorical',
                shuffle=False)
        test_generator = train_datagen_DenseNet201.flow_from_directory(
                'C:/Users\lp\Desktop\R Language\dog_cat/test/',
                target_size=imageSize,
                shuffle=False,
                batch_size=1,
                class_mode=None)
        train = model_DenseNet201.predict_generator(train_generator, train_generator.samples)
        valid = model_DenseNet201.predict_generator(validation_generator,validation_generator.samples)
        test = model_DenseNet201.predict_generator(test_generator,test_generator.samples)

        with h5py.File("save_h5/gap_model_DenseNet201.h5" ) as h:
                h.create_dataset("train", data=train)
                h.create_dataset('valid',data=valid)
                h.create_dataset("test", data=test)
                h.create_dataset("train_label", data=train_generator.classes)
                h.create_dataset("valid_label", data=validation_generator.classes)


def model_inceptionResNetV2_predict():
        model = load_model('saved_models/weights.best.inceptionResNetV2.hdf5')
        imageSize = (299, 299)
        datagen= ImageDataGenerator(
                preprocessing_function=densenet.preprocess_input)
        validation_generator = datagen.flow_from_directory(
                'c:\data\imgs\\valid',
                target_size=imageSize,
                batch_size=1,
                class_mode='categorical',
                shuffle=False)
        train_generator = datagen.flow_from_directory(
                'c:\data\imgs\\train',  # this is the target directory
                target_size=imageSize,  # all images will be resized to 150x150
                batch_size=1,
                class_mode='categorical',
                shuffle=False)
        test_generator = datagen.flow_from_directory(
                'C:/Users\lp\Desktop\R Language\dog_cat/test/',
                target_size=imageSize,
                shuffle=False,
                batch_size=1,
                class_mode=None)
        train = model.predict_generator(train_generator, train_generator.samples)
        valid = model.predict_generator(validation_generator, validation_generator.samples)
        test = model.predict_generator(test_generator, test_generator.samples)

        with h5py.File("save_h5/gap_model_inceptionResNetV2.h5" ) as h:
                h.create_dataset("train", data=train)
                h.create_dataset('valid',data=valid)
                h.create_dataset("test", data=test)
                h.create_dataset("train_label", data=train_generator.classes)
                h.create_dataset("valid_label", data=validation_generator.classes)



def model_inceptionV3_predict():
        model = load_model('saved_models/weights.best.InceptionV3.hdf5')
        imageSize = (299, 299)
        datagen= ImageDataGenerator(
                preprocessing_function=densenet.preprocess_input)
        validation_generator = datagen.flow_from_directory(
                'c:\data\imgs\\valid',
                target_size=imageSize,
                batch_size=1,
                class_mode='categorical',
                shuffle=False)
        train_generator = datagen.flow_from_directory(
                'c:\data\imgs\\train',  # this is the target directory
                target_size=imageSize,  # all images will be resized to 150x150
                batch_size=1,
                class_mode='categorical',
                shuffle=False)
        test_generator = datagen.flow_from_directory(
                'C:/Users\lp\Desktop\R Language\dog_cat/test/',
                target_size=imageSize,
                shuffle=False,
                batch_size=1,
                class_mode=None)
        train = model.predict_generator(train_generator, train_generator.samples)
        valid = model.predict_generator(validation_generator, validation_generator.samples)
        test = model.predict_generator(test_generator, test_generator.samples)

        with h5py.File("save_h5/gap_model_inceptionV3.h5") as h:
                h.create_dataset("train", data=train)
                h.create_dataset('valid',data=valid)
                h.create_dataset("test", data=test)
                h.create_dataset("train_label", data=train_generator.classes)
                h.create_dataset("valid_label", data=validation_generator.classes)


if __name__ == '__main__':
        model_DenseNet201_predict()
        model_inceptionResNetV2_predict()
        model_inceptionV3_predict()
