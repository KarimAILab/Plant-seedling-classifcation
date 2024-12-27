import os
from fastapi import FastAPI, File, UploadFile, HTTPException
import tensorflow as tf
import cv2
import numpy as np
import kagglehub
import shutil

app = FastAPI()
plant_species = ["Black-grass", "Charlock", "Cleavers", "Common Chickweed", "Common wheat", 
                 "Fat Hen", "Loose Silky-bent", "Maize", "Scentless Mayweed", "Shepherds Purse", 
                 "Small-flowered Cranesbill", "Sugar beet"]

Root_directory = os.getcwd()

if not os.listdir(Root_directory+"/model"): 

    path = kagglehub.model_download("cyberkarim/plantseednet/tensorFlow2/default",path='modelcnn1.keras')
    # Define the source file path and destination folder
    source = path
    Root_directory = os.getcwd()
    destination = Root_directory + "/model/modelcnn1.keras"


    try:
        # Move the file
        shutil.move(source, destination)
        print(f"File loaded successfully to {destination}")
    except FileNotFoundError:
        print("model source file not found.")
    except PermissionError:
        print("Permission denied. Unable to load model file.")
    except Exception as e:
        print(f"An error occurred: {e}")

model = tf.keras.models.load_model(Root_directory + "/model/modelcnn1.keras")
model.summary()
 


# Load image from bytes and resize it
async def load_img(file):
    # Read the uploaded file as bytes
    image_bytes = await file.read()

    # Convert image bytes to a NumPy array
    np_arr = np.frombuffer(image_bytes, np.uint8)

    # Decode the image (OpenCV reads it as BGR)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize image
    resized = cv2.resize(img, (100, 100), interpolation=cv2.INTER_AREA)
    return resized

# Process the image (create a mask)
def process_img(img):
    img1 = cv2.GaussianBlur(img, (7, 7), 0)
    img2 = cv2.cvtColor(np.float32(img1), cv2.COLOR_RGB2HSV)

    # Green segment threshold
    green_segment_values = 50
    hsv_lower_bound = np.array([100 - green_segment_values, 0, 0])
    hsv_upper_bound = np.array([100 + green_segment_values, 255, 240])

    mask = cv2.inRange(img2, hsv_lower_bound, hsv_upper_bound)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Rescale mask for the model
    mask = tf.keras.layers.Rescaling(scale=1./255)(mask)
    mask = tf.reshape(mask, (1, 100, 100, 1))  # Add batch dimension
    return mask

# Endpoint to accept image and return prediction
@app.post("/predict/")
async def make_prediction(file: UploadFile = File(...)):
    # Load and process the image
    image = await load_img(file)
    image = process_img(image)

    # Make a prediction using the model
    output = model.predict(image)
    specie_index = np.argmax(output, axis=1)

    return {"class": plant_species[specie_index[0]]}

# To run the FastAPI app, use `uvicorn main:app --reload`