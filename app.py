"""
Voice Deepfake Detection API
FastAPI application for detecting AI-generated vs Human voice samples
"""

import os
import base64
import tempfile
from typing import Dict, Any
from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
import uvicorn
from dotenv import load_dotenv

from model_loader import DeepfakeDetector
from utils import validate_audio_file, cleanup_temp_file

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Voice Deepfake Detection API",
    description="Detect AI-generated vs Human voice samples across 5 languages",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the detector (lazy loading)
detector = None

# Supported languages
SUPPORTED_LANGUAGES = {"ta", "en", "hi", "ml", "te"}
LANGUAGE_NAMES = {
    "ta": "Tamil",
    "en": "English", 
    "hi": "Hindi",
    "ml": "Malayalam",
    "te": "Telugu"
}

# API Configuration
API_KEY = os.getenv("API_KEY", "your-secret-api-key-here")


class DetectionRequest(BaseModel):
    """Request model for voice detection"""
    audio_base64: str = Field(..., description="Base64 encoded MP3 audio file")
    language: str = Field(..., description="Language code (ta/en/hi/ml/te)")
    
    @validator('language')
    def validate_language(cls, v):
        if v not in SUPPORTED_LANGUAGES:
            raise ValueError(f"Language must be one of {SUPPORTED_LANGUAGES}")
        return v
    
    @validator('audio_base64')
    def validate_audio_base64(cls, v):
        if not v or len(v) < 100:
            raise ValueError("Invalid or empty audio data")
        return v


class DetectionResponse(BaseModel):
    """Response model for voice detection"""
    result: str = Field(..., description="AI_GENERATED or HUMAN")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    language: str = Field(..., description="Language name")
    explanation: str = Field(..., description="Brief reasoning for the classification")


@app.on_event("startup")
async def startup_event():
    """Initialize the model on startup"""
    global detector
    print("Loading deepfake detection model...")
    detector = DeepfakeDetector()
    print("Model loaded successfully!")


@app.get("/")
async def root():
    """Root endpoint - API information"""
    return {
        "service": "Voice Deepfake Detection API",
        "version": "1.0.0",
        "status": "active",
        "supported_languages": list(LANGUAGE_NAMES.values()),
        "endpoints": {
            "detection": "/detect",
            "health": "/health"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    model_status = "loaded" if detector and detector.is_loaded() else "not_loaded"
    return {
        "status": "healthy",
        "model_status": model_status,
        "supported_languages": list(SUPPORTED_LANGUAGES)
    }


def verify_api_key(x_api_key: str = Header(...)):
    """Verify the API key from request header"""
    if x_api_key != API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key"
        )
    return x_api_key


@app.post("/detect", response_model=DetectionResponse)
async def detect_deepfake(
    request: DetectionRequest,
    api_key: str = Header(..., alias="x-api-key")
):
    """
    Detect if a voice sample is AI-generated or human
    
    Args:
        request: DetectionRequest containing base64 audio and language
        api_key: API key for authentication (x-api-key header)
    
    Returns:
        DetectionResponse with classification result and confidence
    """
    # Verify API key
    verify_api_key(api_key)
    
    # Check if model is loaded
    if not detector or not detector.is_loaded():
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please try again in a moment."
        )
    
    temp_file_path = None
    
    try:
        # Decode base64 audio
        try:
            audio_bytes = base64.b64decode(request.audio_base64)
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid base64 encoding: {str(e)}"
            )
        
        # Validate audio data
        if len(audio_bytes) < 1000:  # Minimum reasonable file size
            raise HTTPException(
                status_code=400,
                detail="Audio file is too small or corrupted"
            )
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
            temp_file.write(audio_bytes)
            temp_file_path = temp_file.name
        
        # Validate audio file format
        if not validate_audio_file(temp_file_path):
            raise HTTPException(
                status_code=400,
                detail="Invalid audio file format. Must be MP3."
            )
        
        # Run detection
        result = detector.predict(temp_file_path, request.language)
        
        # Build response
        response = DetectionResponse(
            result=result["classification"],
            confidence=result["confidence"],
            language=LANGUAGE_NAMES[request.language],
            explanation=result["explanation"]
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Detection failed: {str(e)}"
        )
    finally:
        # Cleanup temporary file
        if temp_file_path:
            cleanup_temp_file(temp_file_path)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler"""
    return {
        "error": "Internal server error",
        "detail": str(exc)
    }


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=port,
        reload=False
    )