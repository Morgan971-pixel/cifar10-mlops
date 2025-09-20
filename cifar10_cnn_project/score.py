import json
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import load_model

# Called when the service is loaded
def init():
    global model
    # AZUREML_MODEL_DIR is an environment variable created during deployment. 
    # It is the path to the model folder (./azureml-models/$MODEL_NAME/$VERSION)
    # For a registered model, this is the folder containing the model file.
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'cifar10_cnn_model.h5')
    model = load_model(model_path)

# Called when a request is received
def run(raw_data):
    try:
        # Decode the raw data (assuming JSON input of a list of image data)
        data = json.loads(raw_data) # Expecting a list of image data
        
        # Convert to numpy array and normalize
        # Assuming input is a list of 32x32x3 images
        input_images = np.array(data).astype('float32') / 255.0

        # Make predictions
        predictions = model.predict(input_images)

        # Return the predictions as JSON
        return json.dumps(predictions.tolist())
    except Exception as e:
        error = str(e)
        return json.dumps({"error": error})
