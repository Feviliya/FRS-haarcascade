import cv2
import numpy as np
import os
from random import shuffle
from tqdm import tqdm




def generate_dataset():
    face_classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    def face_cropped(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)

        if faces == ():
            return None
        for (x, y, w, h) in faces:
            cropped_face = img[y:y + h, x:x + w]
        return cropped_face

    cap = cv2.VideoCapture(0)
    img_id = 0

    while True:
        ret, frame = cap.read()
        if face_cropped(frame) is not None:
            img_id += 1
            face = cv2.resize(face_cropped(frame), (200, 200))
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            file_name_path = "Image/fevi" + str(img_id) + '.jpg'
            cv2.imwrite(file_name_path, face)
            cv2.putText(face, str(img_id), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("Cropped_Face", face)
            if cv2.waitKey(1) == 13 or int(img_id) == 20:
                break

    cap.release()
    cv2.destroyAllWindows()
    print("Collecting samples is completed !!!")

# generate_dataset()



import numpy as np

def my_label(image_name):
    name = image_name.split('.')[-2]
    # suppose your dataset contains three person
    if name == "fevi":
        return np.array([1, 0])




def my_data():
    data = []
    for img in tqdm(os.listdir("Image")):
        path=os.path.join("Image",img)
        img_data = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img_data = cv2.resize(img_data, (50,50))
        data.append([np.array(img_data), my_label(img)])
    shuffle(data)
    return data
data = my_data()

train = data[:14]
test = data[14:]
X_train = np.array([i[0] for i in train]).reshape(-1,50,50,1)
print(X_train.shape)
y_train = [i[1] for i in train]
X_test = np.array([i[0] for i in test]).reshape(-1,50,50,1)
print(X_test.shape)
y_test = [i[1] for i in test]


import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Input, Dropout, Flatten, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

tf.keras.backend.clear_session()

model = Sequential()
model.add(Input(shape=(50, 50, 1)))
model.add(Conv2D(32, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))  # Adjusted pool_size
model.add(Conv2D(64, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))  # Adjusted pool_size
model.add(Conv2D(128, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))  # Adjusted pool_size
model.add(Conv2D(64, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))  # Adjusted pool_size
model.add(Conv2D(32, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))  # Adjusted pool_size

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.8))
model.add(Dense(2, activation='softmax'))  # Assuming 2 classes: Feviliya and New person

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

model.fit(X_train, y_train, epochs=12, validation_data=(X_test, y_test))


def data_for_visualization():
    Vdata = []
    for img in tqdm(os.listdir("Image")):
        path = os.path.join("Image", img)
        img_num = img.split('.')[0]
        img_data = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img_data = cv2.resize(img_data, (50, 50))
        Vdata.append([np.array(img_data), img_num])
    shuffle(Vdata)
    return Vdata


Vdata = data_for_visualization()
import matplotlib.pyplot as plt  # installation command: pip install matplotlib

fig = plt.figure(figsize=(20, 20))
for num, data in enumerate(Vdata[:20]):
    img_data = data[0]
    y = fig.add_subplot(5, 5, num + 1)
    image = img_data
    data = img_data.reshape(50, 50, 1)
    model_out = model.predict([data])[0]

    if np.argmax(model_out) == 0:
        my_label = 'Feviliya'
    else:
        my_label = 'New person'

    y.imshow(image, cmap='gray')
    plt.title(my_label)

    y.axes.get_xaxis().set_visible(False)
    y.axes.get_yaxis().set_visible(False)
plt.show()