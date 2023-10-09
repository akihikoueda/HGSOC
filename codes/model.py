import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import silence_tensorflow.auto
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications.nasnet import NASNetLarge
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=16, help='Size of the batch')
parser.add_argument('--num_epoch', type=int, default=50, help='Number of epochs')
parser.add_argument('--dataset_dir', type=str, default=None, help='Specify dataset directory')
parser.add_argument('--model_name', type=str, default=None, help='Specify model name')

args = parser.parse_args()
# Parameters
BASE_DIR = args.dataset_dir
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'model')
BATCH_SIZE = args.batch_size    # Batch size
NUM_EPOCH = args.num_epoch      # Number of epochs
LEARNING_RATE = 1E-4            # learning rate

# Data Generator
datagen=ImageDataGenerator(
    rescale=1. / 255,
    featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    rotation_range=0,
    width_shift_range=0,
    height_shift_range=0,
    horizontal_flip=False,
    vertical_flip=False)


# Model
def model_fit(model_name, model, train_size, val_size, save_model=True):
    # Load datasets
    train_generator = datagen.flow_from_directory(
        BASE_DIR + '/train',
        target_size=(331, 331),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True)
    val_generator = datagen.flow_from_directory(
        BASE_DIR + '/validation',
        target_size=(331, 331),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True)
    # EarlyStopping
    early_stopping = EarlyStopping(
        monitor='val_loss',
        min_delta=0.0,
        patience=8,
        mode='min',
        verbose=1,
        restore_best_weights=False
    )
    # reduce learning rate
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.1,
        patience=2,
        min_delta=0.0001,
        verbose=1
    )
    hist = model.fit(
        train_generator,
        steps_per_epoch=train_size//BATCH_SIZE,
        epochs=NUM_EPOCH,
        validation_data=val_generator,
        validation_steps=val_size//BATCH_SIZE,
        callbacks=[early_stopping, reduce_lr],
        shuffle=True,
        verbose=1)
    # save models to the directories
    if save_model:
        if os.path.exists(MODEL_DIR)==False: os.mkdir(MODEL_DIR)
        model.save(MODEL_DIR + '/' + model_name + '.h5')
    return hist

# Plotting learning performance
def learning_plot(title, hist):
    plt.figure(figsize = (18,6))
    # accuracy
    plt.subplot(1, 2, 1)
    plt.plot(hist.history["acc"], label="acc", marker="o")
    plt.plot(hist.history["val_acc"], label="val_acc", marker="o")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.title(title)
    plt.legend(loc="best")
    plt.grid(color='gray', alpha = 0.2)
    # loss
    plt.subplot(1, 2, 2)
    plt.plot(hist.history["loss"], label="loss", marker="o")
    plt.plot(hist.history["val_loss"], label="val_loss", marker="o")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.title(title)
    plt.legend(loc="best")
    plt.grid(color="gray", alpha=0.2)
    plt.show()
    plt.savefig(MODEL_DIR + '/' + title + '.png')


# evaluation of the models
def model_evaluate(model, test_size):
    test_generator = datagen.flow_from_directory(
        BASE_DIR + '/test',
        target_size=(331,331),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False)
    score = model.evaluate(test_generator, steps=test_size//BATCH_SIZE, verbose=0)
    print("evaluate loss: {[0]:.4f}".format(score))
    print("evaluate acc: {[1]:.1%}".format(score))


# Deep learning methods
def NASNetLarge_model(num_classes, train_size, val_size, test_size):
    model_name = "NASNetLarge"
    base_model = NASNetLarge(
        include_top = False,
        weights = "imagenet",
        input_shape = None
    )
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
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=["acc"]
    )
    if model_name is not None:
        model_name = args.model_name
    hist = model_fit(model_name, model, train_size, val_size, save_model=True)
    hist_df=pd.DataFrame(hist.history)
    hist_df.to_csv(MODEL_DIR+'/'+ model_name + '.csv')    
    learning_plot(model_name, hist)
    model_evaluate(model, test_size)

def main():
    train_size = len(glob.glob(BASE_DIR + '/train/*/*.jpg'))
    val_size = len(glob.glob(BASE_DIR + '/validation/*/*.jpg'))
    test_size = len(glob.glob(BASE_DIR + '/test/*/*.jpg'))
    num_classes = len(os.listdir(BASE_DIR + '/train'))
    # learning models
    ## Loss: categorical crossentropy
    ## Optimizer: Adam (lr = 0.0001, epsilon = 1e-4)
    ## save weights with the best,val_loss, reduce LR on plateu
    NASNetLarge_model(num_classes, train_size, val_size, test_size)

if __name__ == "__main__":
    main()
