"""
Utility functions for Voice Deepfake Detection API
"""

import os
import magic
from pathlib import Path


def validate_audio_file(file_path: str) -> bool:
    """
    Validate that the file is a valid audio file (MP3)
    
    Args:
        file_path: Path to the audio file
        
    Returns:
        True if valid, False otherwise
    """
    try:
        # Check file exists
        if not os.path.exists(file_path):
            return False
        
        # Check file size (must be between 1KB and 50MB)
        file_size = os.path.getsize(file_path)
        if file_size < 1000 or file_size > 50 * 1024 * 1024:
            return False
        
        # Check file extension
        if not file_path.lower().endswith(('.mp3', '.wav', '.m4a')):
            return False
        
        # Check MIME type using python-magic
        try:
            mime = magic.from_file(file_path, mime=True)
            valid_mimes = [
                'audio/mpeg',
                'audio/mp3',
                'audio/wav',
                'audio/x-wav',
                'audio/wave',
                'audio/x-m4a',
                'audio/mp4'
            ]
            if mime not in valid_mimes:
                return False
        except Exception as e:
            # If python-magic fails, rely on extension
            print(f"MIME check warning: {e}")
        
        return True
        
    except Exception as e:
        print(f"Validation error: {e}")
        return False


def cleanup_temp_file(file_path: str) -> None:
    """
    Safely delete a temporary file
    
    Args:
        file_path: Path to the temporary file
    """
    try:
        if file_path and os.path.exists(file_path):
            os.unlink(file_path)
    except Exception as e:
        print(f"Cleanup warning: {e}")


def get_audio_duration(file_path: str) -> float:
    """
    Get duration of audio file in seconds
    
    Args:
        file_path: Path to audio file
        
    Returns:
        Duration in seconds
    """
    try:
        import librosa
        duration = librosa.get_duration(path=file_path)
        return duration
    except Exception as e:
        print(f"Duration check error: {e}")
        return 0.0


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted string
    """
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"


def ensure_directory(directory: str) -> None:
    """
    Ensure a directory exists, create if not
    
    Args:
        directory: Directory path
    """
    Path(directory).mkdir(parents=True, exist_ok=True)


def is_valid_language_code(language: str) -> bool:
    """
    Check if language code is valid
    
    Args:
        language: Language code
        
    Returns:
        True if valid, False otherwise
    """
    valid_codes = {'ta', 'en', 'hi', 'ml', 'te'}
    return language in valid_codes