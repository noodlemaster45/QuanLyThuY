import threading
import time
from tkinter import *
import cv2
import numpy as np
from keras.models import load_model
from PIL import Image, ImageTk
img_counter = 0
    # Load the model and labels
def initCam():
    model = load_model("converted_keras\keras_model.h5", compile=False)
    class_names = [line.split(' ', 1)[1].strip() for line in open("converted_keras\labels.txt", "r")]
    
    # Set up GUI window
    root = Tk()
    root.geometry("770x520")
    root.title("Object Detection")

    # Create label to display camera feed
    camera_label = Label(root)
    camera_label.pack(side=TOP)

    # Create label to display prediction result
    prediction_label = Label(root, text="No prediction yet", font=("Arial Bold", 20))
    prediction_label.pack(side=BOTTOM)

    # Define function to update camera feed label
    def update_camera_label():
        ret, image = camera.read()
        image = cv2.resize(image, (350, 350))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        photo = ImageTk.PhotoImage(image=Image.fromarray(image))
        camera_label.configure(image=photo)
        camera_label.image = photo
        camera_label.after(10, update_camera_label)

    # Define function to run prediction in a separate thread
    def run_prediction():
        while True:
            ret, image = camera.read()
            image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
            image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)
            image = (image / 127.5) - 1
            prediction = model.predict(image)
            index = np.argmax(prediction)
            class_name = class_names[index]
            confidence_score = prediction[0][index] * 100
            prediction_text = f"Class: {class_name}\nConfidence: {confidence_score:.2f}%"
            prediction_label.configure(text=prediction_text)
            time.sleep(0.1)
            

    # Start camera and update camera feed label
    camera = cv2.VideoCapture(0)
    update_camera_label()

    # Start prediction thread
    prediction_thread = threading.Thread(target=run_prediction)
    prediction_thread.daemon = True
    prediction_thread.start()

    # Start GUI event loop
    root.mainloop()

    # Release camera and close GUI window
    
    camera.release()
    cv2.destroyAllWindows()
initCam()