"""MongoDB Storage Module"""

from pymongo import MongoClient
from pymongo.collection import Collection
from typing import List, Dict, Any
from datetime import datetime, timezone

class MongoDBStorage:
    """MongoDB storage handler for gesture events"""
    
    def __init__(self, uri: str = "mongodb://localhost:27017", db_name: str = "gesture_ai"):
        """Initialize MongoDB connection
        
        Args:
            uri: MongoDB connection string
            db_name: Database name
        """
        try:
            self.client = MongoClient(uri, serverSelectionTimeoutMS=5000)
            # Verify connection
            self.client.admin.command('ping')
            self.db = self.client[db_name]
            self.events_collection: Collection = self.db['events']
            print(f"Connected to MongoDB: {db_name}")
        except Exception as e:
            print(f"Failed to connect to MongoDB: {e}")
            print("Running in offline mode - events will not be persisted")
            self.client = None
            self.db = None
            self.events_collection = None
    
    def insert_event(self, event: Dict[str, Any]) -> Any:
        """Insert an event into the database
        
        Args:
            event: Event dictionary with gesture, expression, confidence, timestamp
            
        Returns:
            Inserted document ID
        """
        if self.events_collection is None:
            # In offline mode, return a dummy ID
            return "offline_" + datetime.now(timezone.utc).isoformat()
        
        result = self.events_collection.insert_one(event)
        return result.inserted_id
    
    def get_events(self, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """Retrieve events from database
        
        Args:
            limit: Maximum number of events to return
            offset: Number of events to skip
            
        Returns:
            List of event documents
        """
        if self.events_collection is None:
            return []
        
        return list(self.events_collection.find().skip(offset).limit(limit))
    
    def get_events_by_gesture(self, gesture: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get events filtered by gesture type
        
        Args:
            gesture: Gesture name to filter by
            limit: Maximum number of events to return
            
        Returns:
            List of event documents
        """
        if self.events_collection is None:
            return []
        
        return list(self.events_collection.find({"gesture": gesture}).limit(limit))
    
    def clear_events(self) -> bool:
        """Clear all events from database
        
        Returns:
            True if successful
        """
        if self.events_collection is None:
            return False
        
        self.events_collection.delete_many({})
        return True
    
    def close(self):
        """Close MongoDB connection"""
        if self.client:
            self.client.close()
