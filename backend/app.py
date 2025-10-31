"""FastAPI Backend for Assistive Gesture & Facial AI Communication System"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime, timezone
from typing import Optional, List
import os
from dotenv import load_dotenv

from storage import MongoDBStorage

load_dotenv()

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

# Database
db = MongoDBStorage(
    uri=os.getenv("MONGO_URI", "mongodb://localhost:27017"),
    db_name=os.getenv("DB_NAME", "gesture_ai")
)

# Data Models
class Event(BaseModel):
    gesture: str
    expression: str
    confidence: float
    timestamp: Optional[str] = None

class EventResponse(BaseModel):
    id: Optional[str] = None
    gesture: str
    expression: str
    confidence: float
    timestamp: str

# Routes
@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Gesture AI Communication Backend",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

@app.post("/log_event", response_model=EventResponse)
def log_event(event: Event):
    """Log a gesture/expression event"""
    if not event.timestamp:
        event.timestamp = datetime.now(timezone.utc).isoformat()
    
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

@app.get("/events", response_model=List[EventResponse])
def get_events(limit: int = 100, offset: int = 0):
    """Get logged events"""
    events = db.get_events(limit=limit, offset=offset)
    return [
        EventResponse(
            id=str(e.get("_id")),
            gesture=e.get("gesture"),
            expression=e.get("expression"),
            confidence=e.get("confidence"),
            timestamp=e.get("timestamp")
        )
        for e in events
    ]

@app.delete("/events")
def clear_events():
    """Clear all events"""
    db.clear_events()
    return {"status": "success", "message": "All events cleared"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
