import os, zipfile, io, re, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model, load_model
from keras.layers.core import Dense, Dropout
from keras.layers.pooling import GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg19 import VGG19
from keras.applications.inception_v3 import InceptionV3
from keras.applications.xception import Xception
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications import NASNetLarge

# Parameters
DATASET_DIR = "/Users/aueda/Python/GitHub/HGSOC.zip"  # path to Zip file
VALIDATION_SIZE = 0.18      # Train : validation : test = 0.70 : 0.15 : 0.15 (Train dirより validation 抽出)
BATCH_SIZE = 16             # Batch size
NUM_EPOCH = 25              # Number of epochs
LEARNING_RATE = 1E-4        # learning rate
BASE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# HE image normalization parameters
normalize_Io = 240
normalize_alpha = 0.1
normalize_beta = 0.15

# Data Generator
datagen = ImageDataGenerator(
    featurewise_center = False,
    samplewise_center = False,
    featurewise_std_normalization = False,
    samplewise_std_normalization = False,
    zca_whitening = False,
    rotation_range = 0,
    width_shift_range = 0,
    height_shift_range = 0,
    horizontal_flip = False,
    vertical_flip = False)


def importfiles():
    # import ZIP-compressed dataset files
    z = zipfile.ZipFile(DATASET_DIR)
    img_dirs = [x for x in z.namelist() if re.search("^train/.*/$", x)]
    img_dirs = [x.replace('train/', '') for x in img_dirs]
    img_dirs = [x.replace('/', '') for x in img_dirs]
    img_dirs.sort()
    classes = img_dirs
    num_classes = len(classes)
    print('---- {} classes detected : {} ----'.format(num_classes, classes))
    return z, classes, num_classes


def normalizeStaining(img, Io=normalize_Io, alpha=normalize_alpha, beta=normalize_beta):
    ''' Normalize staining appearence of H&E stained images

    Example use:
        see test.py

    Input:
        I: RGB input image
        Io: (optional) transmitted light intensity

    Output:
        Inorm: normalized image
        H: hematoxylin image
        E: eosin image

    Reference:
        A method for normalizing histology slides for quantitative analysis. M.
        Macenko et al., ISBI 2009
    '''

    HERef = np.array([[0.5626, 0.2159],
                      [0.7201, 0.8012],
                      [0.4062, 0.5581]])

    maxCRef = np.array([1.9705, 1.0308])

    # define height and width of image
    h, w, c = img.shape

    # reshape image
    img = img.reshape((-1, 3))

    # calculate optical density
    OD = -np.log((img.astype(np.float) + 1) / Io)

    # remove transparent pixels
    ODhat = OD[~np.any(OD < beta, axis=1)]

    # compute eigenvectors
    eigvals, eigvecs = np.linalg.eigh(np.cov(ODhat.T))

    # eigvecs *= -1

    # project on the plane spanned by the eigenvectors corresponding to the two
    # largest eigenvalues
    That = ODhat.dot(eigvecs[:, 1:3])

    phi = np.arctan2(That[:, 1], That[:, 0])

    minPhi = np.percentile(phi, alpha)
    maxPhi = np.percentile(phi, 100 - alpha)

    vMin = eigvecs[:, 1:3].dot(np.array([(np.cos(minPhi), np.sin(minPhi))]).T)
    vMax = eigvecs[:, 1:3].dot(np.array([(np.cos(maxPhi), np.sin(maxPhi))]).T)

    # a heuristic to make the vector corresponding to hematoxylin first and the
    # one corresponding to eosin second
    if vMin[0] > vMax[0]:
        HE = np.array((vMin[:, 0], vMax[:, 0])).T
    else:
        HE = np.array((vMax[:, 0], vMin[:, 0])).T

    # rows correspond to channels (RGB), columns to OD values
    Y = np.reshape(OD, (-1, 3)).T

    # determine concentrations of the individual stains
    C = np.linalg.lstsq(HE, Y, rcond=None)[0]

    # normalize stain concentrations
    maxC = np.array([np.percentile(C[0, :], 99), np.percentile(C[1, :], 99)])
    tmp = np.divide(maxC, maxCRef)
    C2 = np.divide(C, tmp[:, np.newaxis])

    # recreate the image using reference mixing matrix
    Inorm = np.multiply(Io, np.exp(-HERef.dot(C2)))
    Inorm[Inorm > 255] = 254
    Inorm = np.reshape(Inorm.T, (h, w, 3)).astype(np.uint8)

    # unmix hematoxylin and eosin
    H = np.multiply(Io, np.exp(np.expand_dims(-HERef[:, 0], axis=1).dot(np.expand_dims(C2[0, :], axis=0))))
    H[H > 255] = 254
    H = np.reshape(H.T, (h, w, 3)).astype(np.uint8)

    E = np.multiply(Io, np.exp(np.expand_dims(-HERef[:, 1], axis=1).dot(np.expand_dims(C2[1, :], axis=0))))
    E[E > 255] = 254
    E = np.reshape(E.T, (h, w, 3)).astype(np.uint8)



    return Inorm, H, E


def img2array(path, z, classes, num_classes):
    # import image files from ZIP-compressed dataset
    X = []
    y = []
    class_num = 0
    for class_name in classes:
        if class_num == num_classes : break
        imgfiles = [ x for x in z.namelist() if re.search("^" + path + "/" + class_name + "/.*jpg$", x)]
        for imgfile in imgfiles:
            # ZIPから画像読み込み
            image = Image.open(io.BytesIO(z.read(imgfile)))
            # RGB変換
            image = image.convert('RGB')
            # 画像から配列に変換
            data = np.asarray(image)
            X.append(data)
            y.append(classes.index(class_name))
        class_num += 1
    X = np.array(X).astype('float32')/255
    y = np.array(y)
    # one-hot
    y = to_categorical(y, num_classes = num_classes)
    return X, y

# Model
def model_fit(model_name, model, X_train, X_valid, y_train, y_valid, save_model = True):
    # EarlyStopping
    early_stopping = EarlyStopping(
        monitor = 'val_loss',
        patience = 10,
        verbose = 1
    )
    # ModelCheckpoint
    weights_dir = BASE_DIR + '/weights/'
    if os.path.exists(weights_dir) == False:os.mkdir(weights_dir)
    model_checkpoint = ModelCheckpoint(
        weights_dir + model_name + "_val_loss{val_loss:.3f}.hdf5",
        monitor = 'val_loss',
        verbose = 1,
        save_best_only = True,
        save_weights_only = True,
        period = 3
    )
    # reduce learning rate
    reduce_lr = ReduceLROnPlateau(
        monitor = 'val_loss',
        factor = 0.1,
        patience = 3,
        min_delta = 0.0001,
        verbose = 1
    )
    # log for TensorBoard
    logging = TensorBoard(log_dir = "log/")

    hist = model.fit_generator(
        datagen.flow(X_train, y_train, batch_size = BATCH_SIZE),
        steps_per_epoch = X_train.shape[0] // BATCH_SIZE,
        epochs = NUM_EPOCH,
        validation_data = (X_valid, y_valid),
        callbacks = [early_stopping, reduce_lr],
        shuffle = True,
        verbose = 1
    )
    # save models to the directories
    if save_model == True:
        model_dir = BASE_DIR + '/model/'
        if os.path.exists(model_dir) == False: os.mkdir(model_dir)
        model.save(model_dir + 'model_' + model_name + '.hdf5')
        # optimizerのない軽量モデルを保存（学習や評価不可だが、予測は可能）
        model.save(model_dir + 'model_' + model_name + '-opt.hdf5', include_optimizer=False)
    return hist


# Plotting learning performance
def learning_plot(title, hist):
    plt.figure(figsize = (18,6))
    # accuracy
    plt.subplot(1, 2, 1)
    plt.plot(hist.history["acc"], label = "acc", marker = "o")
    plt.plot(hist.history["val_acc"], label = "val_acc", marker = "o")
    #plt.yticks(np.arange())
    #plt.xticks(np.arange())
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.title(title)
    plt.legend(loc = "best")
    plt.grid(color = 'gray', alpha = 0.2)
    # loss
    plt.subplot(1, 2, 2)
    plt.plot(hist.history["loss"], label = "loss", marker = "o")
    plt.plot(hist.history["val_loss"], label = "val_loss", marker = "o")
    #plt.yticks(np.arange())
    #plt.xticks(np.arange())
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.title(title)
    plt.legend(loc = "best")
    plt.grid(color = "gray", alpha = 0.2)
    plt.show()
    plt.savefig(BASE_DIR + "/" + title + ".png")

# evaluation of the models
def model_evaluate(model, X_test, y_test):
    score = model.evaluate(X_test, y_test, verbose = 1)
    print("evaluate loss: {[0]:.4f}".format(score))
    print("evaluate acc: {[1]:.1%}".format(score))

# Deep learning methods
# ResNet50
def ResNet50_model(X_train, X_valid, y_train, y_valid,  X_test, y_test):
    model_name = "ResNet50"
    base_model = ResNet50(
        include_top = False,
        weights = "imagenet",
        input_shape = None
    )
    # 全結合層の新規構築
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    x = Dense(1024, activation = 'relu')(x)
    predictions = Dense(num_classes, activation = 'softmax')(x)
    model = Model(inputs = base_model.input, outputs = predictions)
    print("Model name: {} ; Layers: {}層".format(model_name, len(model.layers)))
    # # 40層までfreeze
    # for layer in model.layers[:40]:
    #     layer.trainable = False
    # # 40層以降、学習させる
    # for layer in model.layers[40:]:
    #     layer.trainable = True
    model.compile(
        optimizer=Adam(lr=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=["acc"]
    )
    hist = model_fit(model_name, model, X_train, X_valid, y_train, y_valid, save_model = True)
    learning_plot(model_name, hist)
    model_evaluate(model, X_test, y_test)

def VGG19_model(X_train, X_valid, y_train, y_valid,  X_test, y_test):
    model_name = "VGG19"
    base_model = VGG19(
        include_top = False,
        weights = "imagenet",
        input_shape = None
    )
    # 全結合層の新規構築
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    x = Dense(1024, activation = 'relu')(x)
    predictions = Dense(num_classes, activation = 'softmax')(x)
    model = Model(inputs = base_model.input, outputs = predictions)
    print("Model name: {} ; Layers: {}層".format(model_name, len(model.layers)))
    # # 17層までfreeze
    # for layer in model.layers[:17]:
    #     layer.trainable = False
    # # 18層以降、学習させる
    # for layer in model.layers[17:]:
    #     layer.trainable = True
    model.compile(
        optimizer=Adam(lr=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=["acc"]
    )
    hist = model_fit(model_name, model, X_train, X_valid, y_train, y_valid, save_model = True)
    learning_plot(model_name, hist)
    model_evaluate(model, X_test, y_test)


def InceptionV3_model(X_train, X_valid, y_train, y_valid,  X_test, y_test):
    model_name = "InceptionV3"
    base_model = InceptionV3(
        include_top = False,
        weights = "imagenet",
        input_shape = None
    )
    # 全結合層の新規構築
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    x = Dense(1024, activation = 'relu')(x)
    predictions = Dense(num_classes, activation = 'softmax')(x)
    model = Model(inputs = base_model.input, outputs = predictions)
    print("Model name: {} ; Layers: {}層".format(model_name, len(model.layers)))
    # # 249層までfreeze
    # for layer in model.layers[:249]:
    #     layer.trainable = False
    #     # Batch Normalization の freeze解除
    #     if layer.name.startswith('batch_normalization'):
    #         layer.trainable = True
    # # 250層以降、学習させる
    # for layer in model.layers[249:]:
    #     layer.trainable = True
    model.compile(
        optimizer=Adam(lr=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=["acc"]
    )
    hist = model_fit(model_name, model, X_train, X_valid, y_train, y_valid, save_model = True)
    learning_plot(model_name, hist)
    model_evaluate(model, X_test, y_test)


def Xception_model(X_train, X_valid, y_train, y_valid,  X_test, y_test):
    model_name = "Xception"
    base_model = Xception(
        include_top = False,
        weights = "imagenet",
        input_shape = None
    )
    # 全結合層の新規構築
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    x = Dense(1024, activation = 'relu')(x)
    predictions = Dense(num_classes, activation = 'softmax')(x)
    model = Model(inputs = base_model.input, outputs = predictions)
    print("Model name: {} ; Layers: {}層".format(model_name, len(model.layers)))
    # # 108層までfreeze
    # for layer in model.layers[:108]:
    #     layer.trainable = False
    #     # Batch Normalization の freeze解除
    #     if layer.name.startswith('batch_normalization'):
    #         layer.trainable = True
    #     if layer.name.endswith('bn'):
    #         layer.trainable = True
    # # 109層以降、学習させる
    # for layer in model.layers[108:]:
    #     layer.trainable = True
    model.compile(
        optimizer=Adam(lr=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=["acc"]
    )
    hist = model_fit(model_name, model, X_train, X_valid, y_train, y_valid, save_model = True)
    learning_plot(model_name, hist)
    model_evaluate(model, X_test, y_test)


def InceptionResNetV2_model(X_train, X_valid, y_train, y_valid,  X_test, y_test):
    model_name = "InceptionResNetV2"
    base_model = InceptionResNetV2(
        include_top = False,
        weights = "imagenet",
        input_shape = None
    )
    # 全結合層の新規構築
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    x = Dense(1024, activation = 'relu')(x)
    predictions = Dense(num_classes, activation = 'softmax')(x)
    model = Model(inputs = base_model.input, outputs = predictions)
    print("Model name: {} ; Layers: {}層".format(model_name, len(model.layers)))
    # # 249層までfreeze
    # for layer in model.layers[:249]:
    #     layer.trainable = False
    #     # Batch Normalization の freeze解除
    #     if layer.name.startswith('batch_normalization'):
    #         layer.trainable = True
    # # 250層以降、学習させる
    # for layer in model.layers[249:]:
    #     layer.trainable = True
    model.compile(
        optimizer=Adam(lr=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=["acc"]
    )
    hist = model_fit(model_name, model, X_train, X_valid, y_train, y_valid, save_model = True)
    learning_plot(model_name, hist)
    model_evaluate(model, X_test, y_test)


def NASNetLarge_model(X_train, X_valid, y_train, y_valid,  X_test, y_test):
    model_name = "NASNetLarge"
    base_model = NASNetLarge(
        include_top = False,
        weights = "imagenet",
        input_shape = None
    )
    # 全結合層の新規構築
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    x = Dense(1024, activation = 'relu')(x)
    predictions = Dense(num_classes, activation = 'softmax')(x)
    model = Model(inputs = base_model.input, outputs = predictions)
    print("Model name: {} ; Layers: {}層".format(model_name, len(model.layers)))
    # # 1000層までfreeze
    # for layer in model.layers[:1000]:
    #     layer.trainable = False
    #     # Batch Normalization の freeze解除
    #     if layer.name.startswith('batch_normalization'):
    #         layer.trainable = True
    # # 1000層以降、学習させる
    # for layer in model.layers[1000:]:
    #     layer.trainable = True
    model.compile(
        optimizer=Adam(lr=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=["acc"]
    )
    hist = model_fit(model_name, model, X_train, X_valid, y_train, y_valid, save_model = True)
    learning_plot(model_name, hist)
    model_evaluate(model, X_test, y_test)


def main():
    # import classes
    z, classes, num_classes = importfiles()
    # load data
    X_train, y_train = img2array("train", z, classes, num_classes)
    X_test, y_test = img2array("test", z, classes, num_classes)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, random_state = 0, stratify = y_train, test_size = VALIDATION_SIZE)
    print("Datasets: Train {} files, Validation {} files, Test {} files".format(y_train.shape[0], y_valid.shape[0], y_test.shape[0]))
    # learning models
    ## Loss: categorical crossentropy
    ## Optimizer: Adam (lr = 0.0001, epsilon = 1e-4)
    ## save weights with the best,val_loss, reduce LR on plateu
    ResNet50_model(X_train, X_valid, y_train, y_valid, X_test, y_test)
    # VGG19_model(X_train, X_valid, y_train, y_valid, X_test, y_test)
    # InceptionV3_model(X_train, X_valid, y_train, y_valid, X_test, y_test)
    # Xception_model(X_train, X_valid, y_train, y_valid, X_test, y_test)
    # InceptionResNetV2_model(X_train, X_valid, y_train, y_valid, X_test, y_test)
    # NASNetLarge_model(X_train, X_valid, y_train, y_valid, X_test, y_test)

if __name__ == "__main__":
    main()