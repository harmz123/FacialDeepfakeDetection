from DNNGP import DNNGP
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D,MaxPool2D, BatchNormalization, Dropout, Reshape, Concatenate, LeakyReLU
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GaussianNoise
import math
import time
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')
# Height and width refer to the size of the image
# Channels refers to the amount of color channels (red, green, blue)

from tensorflow.keras.models import Model as KerasModel
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Dropout, Reshape, Concatenate, LeakyReLU
from tensorflow.keras.optimizers import Adam

IMGWIDTH = 256

image_dimensions = {'height':256, 'width':256, 'channels':3}

def meso():
    mod=Sequential()

    mod.add(Conv2D(filters=8, kernel_size=(3, 3), input_shape=(256, 256, 3), activation='relu'))
    mod.add(BatchNormalization())
    mod.add(MaxPool2D((2, 2),padding='same'))

    mod.add(Conv2D(filters=8, kernel_size=(5, 5),padding='same',  activation='relu'))
    mod.add(BatchNormalization())
    mod.add(MaxPool2D((2, 2),padding='same'))

    mod.add(Conv2D(filters=16, kernel_size=(5, 5),padding='same',  activation='relu'))
    mod.add(BatchNormalization())
    mod.add(MaxPool2D((2, 2),padding='same'))

    mod.add(Conv2D(filters=16, kernel_size=(5, 5),padding='same',  activation='relu'))
    mod.add(BatchNormalization())
    mod.add(MaxPool2D((4, 4),padding='same'))

    mod.add(Flatten())

    return mod

def get_train_data():
    dataGenerator = ImageDataGenerator(rescale=1. / 255)

    generator = dataGenerator.flow_from_directory(
        '/home/anonymous/PycharmProjects/FacialDeepfakeDetection/dataN1/FaceForensics_C23_Images_Train/Manipulated/Deepfakes',
        target_size=(256, 256),
        batch_size=1,
        class_mode='binary')

    # Re-checking class assignment after removing it
    print(generator.class_indices)

    generator2 = dataGenerator.flow_from_directory(
        '/home/anonymous/PycharmProjects/FacialDeepfakeDetection/dataN1/FaceForensics_C23_Images_Train/Pristine',
        target_size=(256, 256),
        batch_size=1,
        class_mode='binary')

    # Re-checking class assignment after removing it
    print(generator2.class_indices)


    # print("Total data size is: "+str(len(generator.labels)))


    xdata = []
    ydata = []


    for i in range(len(generator.labels)):
        # Loading next picture, generating prediction
        X, y = generator.next()
        color_img = np.squeeze(X)
        ydata.append(0)
        xdata.append(color_img)

    for i in range(len(generator2.labels)):
        # Loading next picture, generating prediction
        X, y = generator2.next()
        color_img = np.squeeze(X)
        ydata.append(1)
        xdata.append(color_img)

    xdata2 = np.array(xdata)
    ydata2 = np.array(ydata)
    # ydata2 = to_categorical(ydata2)
    # idx = np.random.permutation(len(xdata2))
    # xdata,ytrain =xdata2[idx][:1200], ydata2[idx][:1200]

    print(ydata2.shape)
    print(xdata2.shape)


    xdata2_shuffled, ydata2_shuffled = shuffle(xdata2, ydata2)
    # print(ydata2[:10])
    # print(ydata2[-10:])
    # print(ydata2_shuffled[:10])
    # print(ydata2_shuffled[-10:])
    xdata = []
    ydata = []
    xdata2 = np.array(xdata2_shuffled)
    ydata2 = np.array(ydata2_shuffled)
    xdata2_shuffled, ydata2_shuffled = [], []

    # COMMENT FOR NNGP
    # ydata2 = to_categorical(ydata2)
    # print(ydata2.shape)

    image_dimensions = {'height': 256, 'width': 256, 'channels': 3}

    # COMMENT FOR NNGP
    # meso = Meso42()

    # print(xdata2.shape)
    x_train = xdata2
    y_train = ydata2
    print(xdata2.shape)
    print(ydata2.shape)

    return xdata2, ydata2

sigb, sigw, layers = 2.1835419, 2.09, 66  # softmax log-loss

def get_test_data():
    dataGenerator = ImageDataGenerator(rescale=1. / 255)

    generator = dataGenerator.flow_from_directory(
        '/home/anonymous/PycharmProjects/FacialDeepfakeDetection/dataN1/FaceForensics_C23_Images_Test/Manipulated/Deepfakes',
        target_size=(256, 256),
        batch_size=1,
        class_mode='binary')

    # Re-checking class assignment after removing it
    print(generator.class_indices)

    generator2 = dataGenerator.flow_from_directory(
        '/home/anonymous/PycharmProjects/FacialDeepfakeDetection/dataN1/FaceForensics_C23_Images_Test/Pristine',
        target_size=(256, 256),
        batch_size=1,
        class_mode='binary')

    # Re-checking class assignment after removing it
    print(generator2.class_indices)
    sigb, sigw, layers = 2.1835419, 2.09, 66  # softmax log-loss

    # print("Total data size is: "+str(len(generator.labels)))

    xdata = []
    ydata = []
    start = time.time()

    for i in range(len(generator.labels)):
        # Loading next picture, generating prediction
        X, y = generator.next()
        color_img = np.squeeze(X)
        ydata.append(0)
        xdata.append(color_img)

    for i in range(len(generator2.labels)):
        # Loading next picture, generating prediction
        X, y = generator2.next()
        color_img = np.squeeze(X)
        ydata.append(1)
        xdata.append(color_img)

    xdata2 = np.array(xdata)
    ydata2 = np.array(ydata)
    # ydata2 = to_categorical(ydata2)
    # idx = np.random.permutation(len(xdata2))
    # xdata,ytrain =xdata2[idx][:1200], ydata2[idx][:1200]

    print(ydata2.shape)
    print(xdata2.shape)

    xdata2_shuffled, ydata2_shuffled = shuffle(xdata2, ydata2)
    # print(ydata2[:10])
    # print(ydata2[-10:])
    # print(ydata2_shuffled[:10])
    # print(ydata2_shuffled[-10:])
    xdata = []
    ydata = []
    xdata2 = np.array(xdata2_shuffled)
    ydata2 = np.array(ydata2_shuffled)
    xdata2_shuffled, ydata2_shuffled = [], []

    # COMMENT FOR NNGP
    # ydata2 = to_categorical(ydata2)
    # print(ydata2.shape)

    image_dimensions = {'height': 256, 'width': 256, 'channels': 3}

    # COMMENT FOR NNGP
    # meso = Meso42()

    # print(xdata2.shape)
    x_train = xdata2
    y_train = ydata2
    print(xdata2.shape)
    print(ydata2.shape)

    return xdata2, ydata2

model = meso()
x_train, y_train = get_train_data()
x_test, y_test = get_test_data()
x_test2=model.predict(x_test)
x_train2=model.predict(x_train)
x_test=[]
x_train=[]
print('train and test features shapes',x_train2.shape,x_test2.shape)

gp = DNNGP(x_train2,y_train,x_test2,sigb,sigw,layers)
gp.train()
predict = (gp.prediction())
print(predict.shape)


correct=0
test_count=len(predict)
for g in range(test_count):
    if y_test[g] == predict[g]:
        correct +=1

start = time.time()
print("Meso-NNGP Accuracy")
print(correct / len(predict))
finish = time.time()
elapsed = finish - start
print("Time")
print(elapsed)
print()


