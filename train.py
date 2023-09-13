import numpy as np
import cv2
import keras
from keras import regularizers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from keras.optimizers import Adam
from keras.utils.vis_utils import plot_model
from keras.preprocessing.image import ImageDataGenerator

import warnings
warnings.filterwarnings("ignore")

train_dir = 'D:\\My(Kalp) 5th semester\\ML\\Project\\data\\train'
val_dir = 'D:\\My(Kalp) 5th semester\\ML\\Project\\data\\test'
train_datagen = ImageDataGenerator(rescale = 1./255, validation_split = 0.2)
val_datagen = ImageDataGenerator(rescale = 1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size = (48, 48),
    batch_size = 64,
    shuffle = True,
    color_mode = 'grayscale',
    class_mode = 'categorical'
)

validation_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size = (48, 48),
    batch_size = 64,
    shuffle = True,
    color_mode = 'grayscale',
    class_mode = 'categorical'
)

emotion_model = Sequential()
emotion_model.add(Conv2D(32, kernel_size = (3, 3), activation = 'relu', padding = 'same', input_shape = (48, 48, 1)))
emotion_model.add(Conv2D(64, kernel_size = (3, 3), padding = 'same', activation = 'relu'))
emotion_model.add(BatchNormalization())
emotion_model.add(MaxPooling2D(pool_size = (2, 2)))
emotion_model.add(Dropout(0.25))

emotion_model.add(Conv2D(128, kernel_size = (3, 3), padding = 'same', activation = 'relu', kernel_regularizer = regularizers.l2(0.01)))
emotion_model.add(BatchNormalization())
emotion_model.add(MaxPooling2D(pool_size = (2, 2)))
emotion_model.add(Dropout(0.25))

emotion_model.add(Conv2D(256, kernel_size = (3, 3), padding = 'same', activation = 'relu', kernel_regularizer = regularizers.l2(0.01)))
emotion_model.add(BatchNormalization())
emotion_model.add(MaxPooling2D(pool_size = (2, 2)))
emotion_model.add(Dropout(0.25))

emotion_model.add(Conv2D(512, kernel_size = (3, 3), padding = 'same', activation = 'relu', kernel_regularizer = regularizers.l2(0.01)))
emotion_model.add(BatchNormalization())
emotion_model.add(MaxPooling2D(pool_size = (2, 2)))
emotion_model.add(Dropout(0.25))

emotion_model.add(Flatten())
emotion_model.add(Dense(1024, activation = 'relu'))
emotion_model.add(Dropout(0.25))

emotion_model.add(Dense(256, activation = 'relu'))
emotion_model.add(Dropout(0.25))

emotion_model.add(Dense(7, activation = 'softmax'))

print(emotion_model.summary())
# plot model
# plot_model(emotion_model)

# compiling model

emotion_model.compile(loss='categorical_crossentropy',
                       optimizer = Adam(lr=0.0001, decay = 1e-6),
                       metrics = ['accuracy'])

emotion_model_info = emotion_model.fit_generator(
    train_generator,
    steps_per_epoch = train_generator.n // train_generator.batch_size,
    epochs = 5,
    validation_data = validation_generator,
    validation_steps = validation_generator.n // validation_generator.batch_size
)

emotion_model.save_weights('model.h5')

# plotting accuracy

# plt.plot(emotion_model_info.history['accuracy'])
# plt.plot(emotion_model_info.history['val_accuracy'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'val'], loc='upper left')
# plt.show()

# # plotting model loss

# plt.plot(emotion_model_info.history['loss'])
# plt.plot(emotion_model_info.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'val'], loc='upper right')
# plt.show()

# #Confution Matrix and Classification Report
# Y_pred = emotion_model.predict_generator(validation_generator, validation_generator.n // validation_generator.batch_size+1)
# y_pred = np.argmax(Y_pred, axis=1)
# # print('Confusion Matrix')
# # print(confusion_matrix(validation_generator.classes, y_pred))
# # print('Classification Report')

# plt.figure(figsize=(10, 10))
# ax = plt.axes()
# df_confusion = confusion_matrix(validation_generator.classes, y_pred)
# sns.heatmap(df_confusion, annot=True, annot_kws={"size": 10}, fmt='d',cmap="Blues", ax = ax )
# ax.set_title('Confusion Matrix for Imbalanced Dataset')
# plt.ylabel('True Value')
# plt.xlabel('Predicted Value')
# plt.show()

# target_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
# print(classification_report(validation_generator.classes, y_pred, target_names=target_names))