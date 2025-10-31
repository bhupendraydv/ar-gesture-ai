"""MongoDB Storage Module for gesture and expression event logging"""

from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.errors import ServerSelectionTimeoutError, OperationFailure
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
from bson.objectid import ObjectId
import logging

logger = logging.getLogger(__name__)


class MongoDBStorage:
    """MongoDB storage handler for gesture events"""
    
    def __init__(self, uri: str = "mongodb://localhost:27017", db_name: str = "gesture_ai"):
        """Initialize MongoDB connection
        
        Args:
            uri: MongoDB connection string
            db_name: Database name
        """
        
        self.client = None
        self.db = None
        self.events_collection = None
        
        try:
            self.client = MongoClient(uri, serverSelectionTimeoutMS=5000)
            
            # Verify connection
            self.client.admin.command('ping')
            self.db = self.client[db_name]
            self.events_collection: Collection = self.db['events']
            
            logger.info(f"Successfully connected to MongoDB: {db_name}")
        except ServerSelectionTimeoutError as e:
            logger.error(f"Failed to connect to MongoDB (timeout): {e}")
            logger.warning("Running in offline mode - events will not be persisted")
        except OperationFailure as e:
            logger.error(f"MongoDB operation failed: {e}")
            logger.warning("Running in offline mode - events will not be persisted")
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            logger.warning("Running in offline mode - events will not be persisted")
    
    def insert_event(self, event: Dict[str, Any]) -> Optional[str]:
        """Insert an event into the database
        
        Args:
            event: Event dictionary with gesture, expression, confidence, timestamp
            
        Returns:
            Inserted document ID (as string) or None if failed
        """
        try:
            if self.events_collection is None:
                logger.warning("MongoDB not connected, cannot insert event")
                return None
            
            # Ensure timestamp is in ISO format
            if 'timestamp' not in event:
                event['timestamp'] = datetime.now(timezone.utc).isoformat()
            
            result = self.events_collection.insert_one(event)
            logger.debug(f"Event inserted with ID: {result.inserted_id}")
            return str(result.inserted_id)
        except OperationFailure as e:
            logger.error(f"MongoDB operation failed while inserting event: {e}")
            return None
        except Exception as e:
            logger.error(f"Error inserting event: {e}")
            return None
    
    def get_events(self, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """Get events from database
        
        Args:
            limit: Maximum number of events to return
            offset: Number of events to skip
            
        Returns:
            List of event documents
        """
        try:
            if self.events_collection is None:
                logger.warning("MongoDB not connected, returning empty list")
                return []
            
            # Validate parameters
            limit = max(1, min(limit, 1000))  # Limit between 1 and 1000
            offset = max(0, offset)
            
            events = list(
                self.events_collection
                .find({})
                .skip(offset)
                .limit(limit)
                .sort("timestamp", -1)  # Sort by timestamp descending
            )
            
            logger.debug(f"Retrieved {len(events)} events (limit={limit}, offset={offset})")
            return events
        except OperationFailure as e:
            logger.error(f"MongoDB operation failed while retrieving events: {e}")
            return []
        except Exception as e:
            logger.error(f"Error retrieving events: {e}")
            return []
    
    def get_events_by_gesture(self, gesture: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get events filtered by gesture
        
        Args:
            gesture: Gesture name to filter by
            limit: Maximum number of events to return
            
        Returns:
            List of event documents
        """
        try:
            if self.events_collection is None:
                logger.warning("MongoDB not connected, returning empty list")
                return []
            
            if not gesture or not isinstance(gesture, str):
                logger.warning("Invalid gesture parameter")
                return []
            
            events = list(
                self.events_collection
                .find({"gesture": gesture})
                .limit(limit)
                .sort("timestamp", -1)
            )
            
            logger.debug(f"Retrieved {len(events)} events for gesture '{gesture}'")
            return events
        except OperationFailure as e:
            logger.error(f"MongoDB operation failed while filtering by gesture: {e}")
            return []
        except Exception as e:
            logger.error(f"Error retrieving events by gesture: {e}")
            return []
    
    def clear_events(self) -> bool:
        """Clear all events from database
        
        Returns:
            True if successful, False otherwise
        """
        try:
            if self.events_collection is None:
                logger.warning("MongoDB not connected, cannot clear events")
                return False
            
            result = self.events_collection.delete_many({})
            logger.info(f"Cleared {result.deleted_count} events from database")
            return True
        except OperationFailure as e:
            logger.error(f"MongoDB operation failed while clearing events: {e}")
            return False
        except Exception as e:
            logger.error(f"Error clearing events: {e}")
            return False
    
    def get_event_by_id(self, event_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific event by ID
        
        Args:
            event_id: Event MongoDB ObjectId (as string)
            
        Returns:
            Event document or None if not found
        """
        try:
            if self.events_collection is None:
                logger.warning("MongoDB not connected, cannot retrieve event")
                return None
            
            if not event_id or not isinstance(event_id, str):
                logger.warning("Invalid event_id parameter")
                return None
            
            try:
                object_id = ObjectId(event_id)
            except Exception as e:
                logger.warning(f"Invalid ObjectId format: {e}")
                return None
            
            event = self.events_collection.find_one({"_id": object_id})
            return event
        except OperationFailure as e:
            logger.error(f"MongoDB operation failed while retrieving event: {e}")
            return None
        except Exception as e:
            logger.error(f"Error retrieving event: {e}")
            return None
    
    def close(self):
        """Close MongoDB connection"""
        try:
            if self.client:
                self.client.close()
                logger.info("MongoDB connection closed")
        except Exception as e:
            logger.error(f"Error closing MongoDB connection: {e}")
