# import cv2
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.models import model_from_json
# import streamlit as st
# from streamlit_webrtc import VideoTransformerBase, webrtc_streamer
# from PIL import Image

# # Load model
# with open("jsn_model.json", "r") as json_file:
#     loaded_model_json = json_file.read()
# model = model_from_json(loaded_model_json)
# model.load_weights('weights_model1.h5')

# # Loading the classifier from the file.
# face_haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# UPLOAD_FOLDER = 'static'
# ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# def allowed_file(filename):
#     """Checks the file format when file is uploaded"""
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# def Emotion_Analysis(image):
#     """It does prediction of Emotions found in the Image provided, saves as Images and returns them"""
#     gray_frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     faces = face_haar_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

#     if len(faces) == 0:
#         return image, None

#     for (x, y, w, h) in faces:
#         roi = gray_frame[y:y + h, x:x + w]
#         roi = cv2.resize(roi, (48, 48))
#         roi = roi.astype("float") / 255.0
#         roi = tf.expand_dims(roi, axis=-1)  # Adding channel dimension
#         roi = np.expand_dims(roi, axis=0)  # Adding batch dimension

#         prediction = model.predict(roi)
#         EMOTIONS_LIST = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
#         rec_col = {"Happy": (0, 255, 0), "Sad": (255, 0, 0), "Surprise": (255, 204, 55),
#                    "Angry": (0, 0, 255), "Disgust": (230, 159, 0), "Neutral": (0, 255, 255), "Fear": (128, 0, 128)}

#         pred_emotion = EMOTIONS_LIST[np.argmax(prediction)]
#         Text = str(pred_emotion)

#         cv2.rectangle(image, (x, y), (x + w, y + h), rec_col[str(pred_emotion)], 2)
#         cv2.rectangle(image, (x, y - 40), (x + w, y), rec_col[str(pred_emotion)], -1)
#         cv2.putText(image, Text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

#     return image, pred_emotion

# class EmotionDetector(VideoTransformerBase):
#     def transform(self, frame):
#         image = frame.to_ndarray(format="bgr24")
#         result_image, _ = Emotion_Analysis(image)
#         return result_image

# st.title('Emotion Detection App')

# st.sidebar.title("Options")

# # Options for manual upload or webcam capture
# upload_option = st.sidebar.selectbox("Choose Upload Option", ["Image Upload", "Webcam"])

# if upload_option == "Image Upload":
#     uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["png", "jpg", "jpeg", "gif"])

#     if uploaded_file is not None and allowed_file(uploaded_file.name):
#         image = Image.open(uploaded_file)
#         image = np.array(image.convert('RGB'))  # Ensure image is in RGB format
#         result = Emotion_Analysis(image)

#         if result is None:
#             st.image(image, caption="Uploaded Image", use_column_width=True)
#             st.error("No face detected")
#         else:
#             processed_image, pred_emotion = result
#             st.image(processed_image, caption=f"Predicted Emotion: {pred_emotion}", use_column_width=True)

# elif upload_option == "Webcam":
#     st.sidebar.write("Webcam Capture")
#     webrtc_streamer(key="example", video_transformer_factory=EmotionDetector)
# else:
#     st.write("Please select an option to start.")


import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import model_from_json
import streamlit as st
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer
from PIL import Image

# Load model
with open("jsn_model.json", "r") as json_file:
    loaded_model_json = json_file.read()
model = model_from_json(loaded_model_json)
model.load_weights('weights_model1.h5')

# Loading the classifier from the file
face_haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

UPLOAD_FOLDER = 'static'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    """Checks the file format when file is uploaded."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def Emotion_Analysis(image):
    """Predicts emotions from the image provided, marks detected faces, and returns the image."""
    gray_frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_haar_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    if len(faces) == 0:
        return image, None

    for (x, y, w, h) in faces:
        roi = gray_frame[y:y + h, x + w]
        roi = cv2.resize(roi, (48, 48))
        roi = roi.astype("float") / 255.0
        roi = tf.expand_dims(roi, axis=-1)  # Adding channel dimension
        roi = np.expand_dims(roi, axis=0)  # Adding batch dimension

        prediction = model.predict(roi)
        EMOTIONS_LIST = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
        rec_col = {"Happy": (0, 255, 0), "Sad": (255, 0, 0), "Surprise": (255, 204, 55),
                   "Angry": (0, 0, 255), "Disgust": (230, 159, 0), "Neutral": (0, 255, 255), "Fear": (128, 0, 128)}

        pred_emotion = EMOTIONS_LIST[np.argmax(prediction)]
        Text = str(pred_emotion)

        cv2.rectangle(image, (x, y), (x + w, y + h), rec_col[str(pred_emotion)], 2)
        cv2.rectangle(image, (x, y - 40), (x + w, y), rec_col[str(pred_emotion)], -1)
        cv2.putText(image, Text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    return image, pred_emotion

def resize_frame(frame, width=None, height=None):
    """Resize the frame to the given width and height while maintaining aspect ratio."""
    (h, w) = frame.shape[:2]
    if width is None and height is None:
        return frame
    if width is None:
        ratio = height / float(h)
        dim = (int(w * ratio), height)
    else:
        ratio = width / float(w)
        dim = (width, int(h * ratio))
    return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

class EmotionDetector(VideoTransformerBase):
    def transform(self, frame):
        image = frame.to_ndarray(format="bgr24")
        image = resize_frame(image, width=320)  # Resize frame to width 320 pixels
        result_image, _ = Emotion_Analysis(image)
        return result_image

st.title('Emotion Detection App')

st.sidebar.title("Options")

# Options for manual upload or webcam capture
upload_option = st.sidebar.selectbox("Choose Upload Option", ["Image Upload", "Webcam"])

if upload_option == "Image Upload":
    uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["png", "jpg", "jpeg", "gif"])

    if uploaded_file is not None and allowed_file(uploaded_file.name):
        image = Image.open(uploaded_file)
        image = np.array(image.convert('RGB'))  # Ensure image is in RGB format
        result = Emotion_Analysis(image)

        if result is None:
            st.image(image, caption="Uploaded Image", use_column_width=True)
            st.error("No face detected")
        else:
            processed_image, pred_emotion = result
            st.image(processed_image, caption=f"Predicted Emotion: {pred_emotion}", use_column_width=True)

elif upload_option == "Webcam":
    st.sidebar.write("Webcam Capture")
    webrtc_streamer(key="example", video_transformer_factory=EmotionDetector)
else:
    st.write("Please select an option to start.")

st.write("Make sure your browser supports WebRTC and is using HTTPS connection.")

