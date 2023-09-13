import cv2
import numpy as np
# from tkinter import *
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from keras import regularizers

# Model Creation

emotion_model = Sequential()
emotion_model.add(Conv2D(32, kernel_size = (3, 3), activation = 'relu', padding = 'same', kernel_regularizer = regularizers.l2(0.01), input_shape = (48, 48, 1)))
emotion_model.add(Conv2D(64, kernel_size = (3, 3), padding = 'same', activation = 'relu', kernel_regularizer = regularizers.l2(0.01)))
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
emotion_model.add(Dense(1024, activation = 'relu', kernel_regularizer = regularizers.l2(0.01)))
emotion_model.add(Dropout(0.25))

emotion_model.add(Dense(256, activation = 'relu', kernel_regularizer = regularizers.l2(0.01)))
emotion_model.add(Dropout(0.25))

emotion_model.add(Dense(7, activation = 'softmax'))

emotion_model.load_weights('D:\\My(Kalp) 5th semester\\ML\\Project\\models\\model_epoch_100.h5')

emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful",
                3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}


# start the webcam feed
#cap = cv2.VideoCapture(0)

# pass here your video path
# you may download one from here : https://www.pexels.com/video/three-girls-laughing-5273028/
cap = cv2.VideoCapture("D:\\My(Kalp) 5th semester\\ML\\Project\\Sample_videos\\pexels-cottonbro-5790184.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)
print("fps = ", fps)
while True:
    # Find haar cascade to draw bounding box around face
    ret, frame = cap.read()

    k = cv2.waitKey(100)
    if(k == 27):
        break        
    frame = cv2.resize(frame, (1280, 720))
    if not ret:
        break
    face_detector = cv2.CascadeClassifier(
        r'C:\\Users\\kalpp\\AppData\\Roaming\\Python\\Python39\\site-packages\\cv2\\data\\haarcascade_frontalface_default.xml')
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect faces available on camera
    num_faces = face_detector.detectMultiScale(
        gray_frame, scaleFactor=1.3, minNeighbors=5)

    # take each face available on the camera and Preprocess it
    for (x, y, w, h) in num_faces:
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)
        roi_gray_frame = gray_frame[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(
            cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

        # predict the emotions
        emotion_prediction = emotion_model.predict(cropped_img)
        maxindex = int(np.argmax(emotion_prediction))
        cv2.putText(frame, emotion_dict[maxindex], (x+5, y-20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    cv2.imshow('Emotion Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# exitButton = Button(root, text='Quit', fg="red", command=root.destroy, font=(
#         'arial', 25, 'bold')).pack(side=BOTTOM)