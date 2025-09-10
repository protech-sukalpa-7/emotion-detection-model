import cv2
import numpy as np
from keras.models import model_from_json, Sequential
import keras

# ✅ Register Sequential for backward compatibility
keras.utils.get_custom_objects().update({"Sequential": Sequential})

# ---- Load model ----
with open("facialemotionmodel.json", "r") as json_file:
    model_json = json_file.read()

model = model_from_json(model_json)
model.load_weights("facialemotionmodel.h5")

# ---- Load Haar Cascade ----
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

# ---- Feature Extraction ----
def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

# ---- Webcam ----
webcam = cv2.VideoCapture(0)
labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

while True:
    i, im = webcam.read()
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)  # ✅ use gray instead of im

    for (p, q, r, s) in faces:
        image = gray[q:q+s, p:p+r]
        cv2.rectangle(im, (p, q), (p+r, q+s), (255, 0, 0), 2)
        image = cv2.resize(image, (48, 48))
        img = extract_features(image)

        pred = model.predict(img, verbose=0)
        prediction_label = labels[pred.argmax()]

        cv2.putText(im, prediction_label, (p-10, q-10),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255), 2)

    cv2.imshow("Output", im)

    if cv2.waitKey(27) & 0xFF == 27:  # press ESC to exit
        break

webcam.release()
cv2.destroyAllWindows()
