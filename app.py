import cv2
import  tensorflow as tf
# Importing image utilities from TensorFlow's Keras
#from tensorflow.keras.utils import img_to_array
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
from PIL import Image, ImageTk
from tensorflow.keras.models import model_from_json
import numpy as np
import tkinter as tk
from tkinter import filedialog

# Load the model architecture from the JSON file
with open('models/vgg16.json', 'r') as f:
    model_json = f.read()
model = model_from_json(model_json)
model.load_weights('models/vgg16 (1).h5')

# Initialize the face detector
face_haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Create the main window
root = tk.Tk()
root.title('Emotion Detection')
canvas = tk.Canvas(root, width=640, height=500)
canvas.pack()

# Create a label for the video player
video_label = tk.Label(root)
video_label.pack()

# Create a button to open the video file
def open_video():
    video_file = filedialog.askopenfilename(filetypes=[('Video Files', '*.webm')])
    if video_file:
        # Reset the emotion label
        emotion_label = ''
        play_video(video_file)

open_button = tk.Button(root, text='Open Video', command=open_video)
open_button.pack()

def play_video(video_file):
    # Open the video file
    cap = cv2.VideoCapture(video_file)

    emotions = {'Angry': 0, 'Disgust': 0, 'Fear': 0, 'Happy': 0, 'Sad': 0, 'Surprise': 0, 'Neutral': 0}

    while cap.isOpened():
        # Read a frame from the video file
        ret, frame = cap.read()

        if ret:
            # Process the frame with the VGG16 model
            gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)
            for (x, y, w, h) in faces_detected:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), thickness=7)
                roi_gray = gray_img[y:y + w, x:x + h]  # cropping region of interest i.e. face area from  image
                roi_gray = cv2.resize(roi_gray, (48, 48))
                img_pixels = image.img_to_array(roi_gray)

                img_pixels = np.expand_dims(img_pixels, axis=0)
                img_pixels /= 255
                predictions = model.predict(img_pixels)

                # Get the predicted emotional state
                emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
                emotion_label = emotion_labels[np.argmax(predictions[0])]

                # Update the emotions dictionary
                emotions[emotion_label] += 1

                # Display the predicted emotional state on the frame
                cv2.putText(frame, emotion_label, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(frame, "angry : " + str(round((predictions[0][0] * 100), 2)) + " %", (50, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, 155, 0)
                cv2.putText(frame, "disgust : " + str(round((predictions[0][1] * 100), 2)) + " %", (50, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, 155, 0)
                cv2.putText(frame, "fear : " + str(round((predictions[0][2] * 100), 2)) + " %", (50, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, 155, 1)
                cv2.putText(frame, "happy : " + str(round((predictions[0][3] * 100), 2)) + " %", (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, 155, 1)
                cv2.putText(frame, "sad : " + str(round((predictions[0][4] * 100), 2)) + " %", (50, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, 155, 1)
                cv2.putText(frame, "surprise : " + str(round((predictions[0][5] * 100), 2)) + " %", (50, 140),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, 155, 1)
                cv2.putText(frame, "neutral : " + str(round((predictions[0][6] * 100), 2)) + " %", (50, 160),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, 155, 1)

            # Display the frame in the GUI
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            img = ImageTk.PhotoImage(img)
            canvas.create_image(0, 0, image=img, anchor=tk.NW)
            root.update()

            # Exit if the user closes the window
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    # Release the video capture object and close the window
    cap.release()

    # Generate the pie chart image
    plt.figure(figsize=(5, 5))
    plt.pie(list(emotions.values()), labels=list(emotions.keys()), autopct='%1.1f%%')
    plt.title('Emotions Distribution')
    plt.savefig('emotions_pie_chart.png')

    # Display the pie chart image in the GUI
    pie_chart_img = Image.open('emotions_pie_chart.png')
    pie_chart_img = pie_chart_img.resize((300, 300))
    pie_chart_img = ImageTk.PhotoImage(pie_chart_img)
    pie_chart_label = tk.Label(root, image=pie_chart_img)
    pie_chart_label.pack()

# Run the main loop
root.mainloop()
