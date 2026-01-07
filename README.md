# Potato Plant Disease Prediction

A deep learning-based system for automated detection and classification of potato leaf diseases. The system can identify Early Blight, Late Blight, and Healthy potato leaves using convolutional neural networks.

## Features

- Real-time disease classification from uploaded images
- Confidence scores for predictions
- Web interface built with Streamlit
- RESTful API using FastAPI
- Multiple model versions for production and testing

## Technologies

- TensorFlow & Keras
- Streamlit
- FastAPI
- NumPy & PIL
- Python

## Project Structure

```
├── app.py                
├── fastapi/              
│   ├── main.py
│   └── requirements.txt
├── training/            
│   └── training.ipynb
├── saved_models/          
└── Data/                  
    └── PlantVillage/
        ├── Potato___Early_blight/
        ├── Potato___Late_blight/
        └── Potato___healthy/
```

## Dataset

The project uses the PlantVillage Potato Leaf Dataset:
- Early Blight: 1,000 images
- Late Blight: 1,000 images
- Healthy: 152 images

## API Endpoints

**GET /ping** - Health check endpoint

**POST /predict** - Disease prediction endpoint
- Accepts image file upload
- Returns predicted class and confidence score

Example response:
```json
{
  "class": "Early Blight",
  "confidence": 0.9876
}
```

## Model Information

The project uses CNN models trained with TensorFlow/Keras. Models are saved in both Keras format (.keras) and SavedModel format. The production model (1.keras) is used for predictions in both the web app and API.
