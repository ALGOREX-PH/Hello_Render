from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
#from tensorflow.keras.models import load_model
#import numpy as np
#from PIL import Image
#import io

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hello World"}

@app.get("/hello")
def hello():
    return {"message": "Hello from the FastAPI cron job!"}
# Load the CNN model
#model = load_model("Model/proj_apple.h5")

# Define input size (assuming your CNN model expects a specific image size, e.g., 224x224)
#IMG_SIZE = (224, 224)

# Helper function to preprocess the image
#def preprocess_image(image: Image.Image) -> np.ndarray:
#    image = image.resize(IMG_SIZE)
#    image = np.array(image)
#    # If the model expects grayscale or RGB images, make adjustments accordingly
#    if len(image.shape) == 2:
#       image = np.expand_dims(image, axis=-1)  # Add channel dimension for grayscale
#    image = image / 255.0  # Normalize pixel values
#    image = np.expand_dims(image, axis=0)  # Add batch dimension
#    return image

# API route for prediction
#@app.post("/predict")
#async def predict(file: UploadFile = File(...)):
#    try:
        # Read image from the uploaded file
#        contents = await file.read()
#        image = Image.open(io.BytesIO(contents)).convert("RGB")
#        processed_image = preprocess_image(image)

        # Make prediction
#       prediction = model.predict(processed_image)
#       predicted_class = np.argmax(prediction, axis=1).item()

#        return {"prediction": int(predicted_class)}

#    except Exception as e:
#        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

# Run the application
#if __name__ == "__main__":
#    import uvicorn
#    uvicorn.run(app, host="0.0.0.0", port=8000)
