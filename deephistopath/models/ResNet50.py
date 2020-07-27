from keras.preprocessing.image import ImageDataGenerator
from keras.applications.resnet50 import ResNet50
from keras.models import Sequential, Model
from keras.layers import Input, Flatten, Dense, Dropout
from keras.layers.pooling import GlobalAveragePooling2D
from keras import optimizers
import matplotlib.pyplot as plt

classes = ["IR", "MT", "SP", "PG", "Stroma"]
nb_classes = len(classes)
BASE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# BASE_DIR = '/content/gdrive/My Drive/HGSOC'  ## Google Colaboratory使用の場合
train_data_dir = BASE_DIR + '/dataset/train'
validation_data_dir = BASE_DIR + '/dataset/val'
nb_train_samples = 3760
nb_validation_samples = 925
img_width, img_height = 224, 224
batch_size = 16
num_epoch = 25
learning_rate = 1e-5

train_datagen = ImageDataGenerator(rescale=1.0 / 255)
validation_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = train_datagen.flow_from_directory(
  train_data_dir,
  target_size=(img_width, img_height),
  color_mode='rgb',
  classes=classes,
  class_mode='categorical',
  batch_size= batch_size)

validation_generator = validation_datagen.flow_from_directory(
  validation_data_dir,
  target_size=(img_width, img_height),
  color_mode='rgb',
  classes=classes,
  class_mode='categorical',
  batch_size= batch_size)

input_tensor = Input(shape=(img_width, img_height, 3))
ResNet50 = ResNet50(include_top=False, weights='imagenet',input_tensor=input_tensor)
top_model = Sequential()
top_model.add(GlobalAveragePooling2D())
top_model.add(Dense(1024, activation = 'relu'))
# top_model.add(Dropout(0.5))
top_model.add(Dense(nb_classes, activation='softmax'))
model = Model(input=ResNet50.input, output=top_model(ResNet50.output))

# # 出力側10層のみFine Tuningに設定
# for layer in model.layers[:40]:
#     layer.trainable = False

model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.Adam(lr=learning_rate),
              # optimizer=optimizers.SGD(lr=learning_rate, momentum=0.9),
              metrics=['acc'])

hist = model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples//batch_size,
    nb_epoch=num_epoch,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples//batch_size)

plt.figure(figsize = (18,6))

# accuracy
plt.subplot(1, 2, 1)
plt.plot(hist.history["acc"], label = "acc", marker = "o")
plt.plot(hist.history["val_acc"], label = "val_acc", marker = "o")
#plt.xticks(np.arange())
#plt.yticks(np.arange())
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.title("ResNet50")
plt.legend(loc = "best")
plt.grid(color = 'gray', alpha = 0.2)

# loss
plt.subplot(1, 2, 2)
plt.plot(hist.history["loss"], label = "loss", marker = "o")
plt.plot(hist.history["val_loss"], label = "val_loss", marker = "o")
#plt.xticks(np.arange())
#plt.yticks(np.arange())
plt.xlabel("epoch")
plt.ylabel("loss")
plt.title("ResNet50")
plt.legend(loc = "best")
plt.grid(color = 'gray', alpha = 0.2)

plt.show()