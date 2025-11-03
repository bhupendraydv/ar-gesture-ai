"""FastAPI Backend for Assistive Gesture & Facial AI Communication System"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime, timezone
from typing import Optional, List
import os
import logging
from dotenv import load_dotenv

try:
    from storage import MongoDBStorage
except ImportError as e:
    print(f"Error importing storage module: {e}")
    raise

# Countries routes
try:
    from .countries_routes import router as countries_router  # type: ignore
except Exception:
    # When running as module-less (e.g., uvicorn backend.app:app), fallback to direct import
    try:
        from countries_routes import router as countries_router  # type: ignore
    except Exception:
        countries_router = None  # Will handle later

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Gesture AI Communication API",
    description="API for Assistive Gesture & Facial Expression Recognition",
    version="1.0.0"
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount countries router if available
if countries_router is not None:
    app.include_router(countries_router)
else:
    logger.warning("Countries router not available; /countries endpoints will be unavailable")

# Database
try:
    db = MongoDBStorage(
        uri=os.getenv("MONGO_URI", "mongodb://localhost:27017"),
        db_name=os.getenv("DB_NAME", "gesture_ai")
    )
    logger.info("MongoDB connection initialized")
except Exception as e:
    logger.error(f"Failed to initialize MongoDB: {e}")
    # Continue without database
    db = None

# Data Models
class Event(BaseModel):
    """Event model for gesture/expression logging"""
    gesture: str
    expression: str
    confidence: float
    timestamp: Optional[str] = None

class EventResponse(BaseModel):
    """Response model for events"""
    id: str
    gesture: str
    expression: str
    confidence: float
    timestamp: str

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    service: str
    timestamp: str

@app.get("/health", response_model=HealthResponse)
def health_check():
    """Health check endpoint"""
    try:
        return {
            "status": "healthy",
            "service": "Gesture AI Communication Backend",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail="Health check failed")

@app.post("/log_event", response_model=EventResponse)
def log_event(event: Event):
    """Log a gesture/expression event"""
    try:
        if not event.timestamp:
            event.timestamp = datetime.now(timezone.utc).isoformat()
        
        # Validate input
        if not event.gesture or len(event.gesture.strip()) == 0:
            raise HTTPException(status_code=400, detail="Gesture cannot be empty")
        if not event.expression or len(event.expression.strip()) == 0:
            raise HTTPException(status_code=400, detail="Expression cannot be empty")
        if event.confidence < 0 or event.confidence > 1:
            raise HTTPException(status_code=400, detail="Confidence must be between 0 and 1")
        
        if db is None:
            logger.warning("MongoDB not available, returning mock response")
            return EventResponse(
                id="mock_id",
                gesture=event.gesture,
                expression=event.expression,
                confidence=event.confidence,
                timestamp=event.timestamp
            )
        
        # Store in database
        result = db.insert_event({
            "gesture": event.gesture,
            "expression": event.expression,
            "confidence": event.confidence,
            "timestamp": event.timestamp
        })
        
        return EventResponse(
            id=str(result),
            gesture=event.gesture,
            expression=event.expression,
            confidence=event.confidence,
            timestamp=event.timestamp
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error logging event: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to log event: {str(e)}")

@app.get("/events", response_model=List[EventResponse])
def get_events(limit: int = 100, offset: int = 0):
    """Get logged events"""
    try:
        if limit < 1 or limit > 1000:
            raise HTTPException(status_code=400, detail="Limit must be between 1 and 1000")
        if offset < 0:
            raise HTTPException(status_code=400, detail="Offset must be non-negative")
        
        if db is None:
            logger.warning("MongoDB not available, returning empty list")
            return []
        
        events = db.get_events(limit=limit, offset=offset)
        return [
            EventResponse(
                id=str(e.get("_id")),
                gesture=e.get("gesture", ""),
                expression=e.get("expression", ""),
                confidence=e.get("confidence", 0),
                timestamp=e.get("timestamp", "")
            )
            for e in events
        ]
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving events: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve events: {str(e)}")

@app.delete("/events")
def clear_events():
    """Clear all events"""
    try:
        if db is None:
            logger.warning("MongoDB not available, cannot clear events")
            return {"status": "warning", "message": "MongoDB not available"}
        
        db.clear_events()
        return {"status": "success", "message": "All events cleared"}
    except Exception as e:
        logger.error(f"Error clearing events: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to clear events: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
