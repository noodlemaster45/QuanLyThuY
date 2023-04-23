from keras.models import load_model  # TensorFlow is required for Keras to work
import cv2  # Install opencv-python
import numpy as np
def initCam():
    
    img_path = "images/"
    img_counter = 0 
    # Disable scientific notation for clarity
    np.set_printoptions(suppress=True)

    # Load the model
    model = load_model("converted_keras\keras_model.h5", compile=False)

    # Load the labels
    class_names = open("converted_keras\labels.txt", "r").readlines()

    # CAMERA can be 0 or 1 based on default camera of your computer
    camera = cv2.VideoCapture(0)

    while True:
        # Grab the webcamera's image.
        ret, image = camera.read()

        # Resize the raw image into (224-height,224-width) pixels
        image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
        savedIm = image
        # Show the image in a window
        cv2.imshow("Webcam Image", image)

        # Make the image a numpy array and reshape it to the models input shape.
        image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)

        # Normalize the image array
        image = (image / 127.5) - 1

        # Predicts the model
        prediction = model.predict(image)
        index = np.argmax(prediction)
        class_name = class_names[index]
        confidence_score = prediction[0][index]

        # Print prediction and confidence score
        print("Class:", class_name[2:],end ="")
        print("Confidence Score:", str(np.round(confidence_score * 100))[:-2], "%")
        
        class_name = class_name[2:].strip()
        img_name = class_name.strip()+str(img_counter)
        
        # Listen to the keyboard for presses.
        keyboard_input = cv2.waitKey(1)
        # 27 is the ASCII for the esc key on your keyboard.
        if keyboard_input == 27:
            break
        elif keyboard_input == 32:
            # SPACE pressed
            img_name =img_path+img_name+ ".jpg"
            cv2.imwrite(img_name, savedIm)
            print("{} written!".format(img_name))
            # img = cv2.imread(img_name)
            # cv2.imshow("picture",img)
            img_counter += 1
            
            break
    camera.release()
    cv2.destroyAllWindows()
    return img_name, class_name
   
