# Assistive Gesture & Facial AI Communication System

🔥 **Help speech-disabled people communicate using hand gestures + facial expressions.**

A production-grade full-stack AI system that detects and classifies hand gestures and facial expressions in real-time, converting them to meaningful text output with optional text-to-speech conversion. All events are logged to MongoDB for analytics and insights.

---

## 🎯 Features

✅ **Real-time Hand Gesture Recognition**
- Detects 21 hand landmarks using MediaPipe
- Recognizes: Hello, Help, Yes, No, Stop
- Provides confidence scores
- Modular design for easy gesture addition

✅ **Facial Expression Recognition**
- Detects: Neutral, Happy, Sad, Angry
- Real-time processing with MediaPipe FaceMesh
- Lightweight and responsive

✅ **FastAPI Backend**
- `/health` - Health check endpoint
- `/log_event` - Log gesture + expression events
- `/events` - Retrieve logged events
- CORS-enabled for cross-origin requests

✅ **MongoDB Integration**
- Persistent event storage
- Query and retrieve communication history
- Works in offline mode if MongoDB unavailable

✅ **Real-time HUD Overlay**
- Displays gesture and expression
- Shows confidence scores
- FPS counter
- Clean, accessible UI

✅ **Optional TTS (Text-to-Speech)**
- Convert gestures to speech
- Accessible audio feedback

---

## 🛠️ Tech Stack

- **Backend:** Python + FastAPI + Uvicorn
- **ML/Computer Vision:** MediaPipe + OpenCV + scikit-learn
- **Database:** MongoDB (pymongo)
- **Frontend:** Python + OpenCV (real-time HUD)
- **OS:** Cross-platform (Windows, macOS, Linux)

**Dependencies:**
```
fastapi==0.104.1
uvicorn==0.24.0
mediapipe==0.10.0
opencv-python==4.8.1.78
numpy==1.24.3
scikit-learn==1.3.2
pymongo==4.6.0
pyttsx3==2.90  # Optional TTS
```

---

## 📁 Project Structure

```
ar-gesture-ai/
├── backend/
│   ├── app.py                     # FastAPI main application
│   ├── gesture_recognition.py     # Gesture detection module
│   ├── facial_expression.py       # Expression recognition module
│   ├── storage.py                 # MongoDB storage handler
│   └── models/                    # Pre-trained ML models
│
├── ui/
│   ├── main.py                    # UI application entry point
│   └── hud_elements.py            # HUD overlay renderer
│
├── data/
│   ├── gestures/                  # Training data for gestures
│   └── faces/                     # Training data for expressions
│
├── requirements.txt               # Python dependencies
├── README.md                      # This file
└── .gitignore                     # Git ignore rules
```

---

## 🚀 Installation

### Prerequisites
- Python 3.8+
- pip or conda
- MongoDB (optional, runs in offline mode without it)
- Webcam or camera device

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/bhupendraydv/ar-gesture-ai.git
cd ar-gesture-ai
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Setup MongoDB (Optional)**
```bash
# Using Docker (recommended)
docker run -d -p 27017:27017 --name gesture-ai-db mongo:latest

# Or install locally
# macOS: brew install mongodb-community
# Ubuntu: sudo apt-get install mongodb
# Windows: Download from https://www.mongodb.com/try/download/community
```

5. **Create .env file (Optional)**
```bash
echo 'MONGO_URI=mongodb://localhost:27017' > .env
echo 'DB_NAME=gesture_ai' >> .env
```

---

## 🏃 Running the Application

### Start the Backend API

```bash
cd backend
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

**API Documentation:** `http://localhost:8000/docs` (Swagger UI)

### Start the UI

In a new terminal:

```bash
python ui/main.py
```

**Controls:**
- `q` - Quit application
- `r` - Record current gesture + expression to database

---

## 📡 API Endpoints

### 1. Health Check
```bash
GET /health

Response:
{
  "status": "healthy",
  "service": "Gesture AI Communication Backend",
  "timestamp": "2025-10-31T10:15:00Z"
}
```

### 2. Log Event
```bash
POST /log_event

Request:
{
  "gesture": "Hello",
  "expression": "Happy",
  "confidence": 0.92,
  "timestamp": "2025-10-31T10:15:30Z"
}

Response:
{
  "id": "507f1f77bcf86cd799439011",
  "gesture": "Hello",
  "expression": "Happy",
  "confidence": 0.92,
  "timestamp": "2025-10-31T10:15:30Z"
}
```

### 3. Get Events
```bash
GET /events?limit=100&offset=0

Response:
[
  {
    "id": "507f1f77bcf86cd799439011",
    "gesture": "Hello",
    "expression": "Happy",
    "confidence": 0.92,
    "timestamp": "2025-10-31T10:15:30Z"
  },
  ...
]
```

### 4. Clear Events
```bash
DELETE /events

Response:
{
  "status": "success",
  "message": "All events cleared"
}
```

---

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the project root:

```
MONGO_URI=mongodb://localhost:27017
DB_NAME=gesture_ai
API_HOST=0.0.0.0
API_PORT=8000
```

### Gesture Customization

To add new gestures, edit `backend/gesture_recognition.py`:

```python
class GestureRecognizer:
    GESTURES = ["Hello", "Help", "Yes", "No", "Stop", "YourNewGesture"]
```

Then train the model with new gesture samples.

---

## 📊 Database Schema

MongoDB Collection: `events`

```javascript
{
  "_id": ObjectId("..."),
  "gesture": String,          // e.g., "Hello"
  "expression": String,       // e.g., "Happy"
  "confidence": Number,       // 0.0 - 1.0
  "timestamp": ISODate("...") // ISO 8601 format
}
```

---

## 🧠 ML Models

### Gesture Recognition
- **Algorithm:** RandomForest Classifier (scikit-learn)
- **Features:** 21 hand landmarks (42D)
- **Training:** Requires labeled gesture samples
- **Output:** Gesture class + confidence score

### Facial Expression Recognition
- **Algorithm:** Heuristic-based on facial landmarks
- **Features:** Mouth distance, eye openness
- **Processing:** Real-time with MediaPipe FaceMesh
- **Output:** Expression class + confidence score

---

## 🔄 Workflow

1. **Capture:** Camera captures video frames
2. **Detect:** MediaPipe detects hand and face landmarks
3. **Recognize:** ML models classify gesture and expression
4. **Display:** HUD overlay shows results in real-time
5. **Log:** Events are sent to FastAPI backend
6. **Store:** MongoDB persists events for history

---

## 🚨 Troubleshooting

### Camera not detected
```bash
# Try different camera IDs
python ui/main.py  # Default: Camera 0
```

### MongoDB connection error
The system will run in offline mode without MongoDB. Events won't be persisted.

### Low FPS
- Reduce model complexity
- Check system resources
- Update GPU drivers if available

### Poor gesture recognition
- Ensure good lighting
- Maintain consistent gesture positions
- Train models with more samples

---

## 📝 Usage Examples

### Python Client
```python
import requests

api_url = "http://localhost:8000"

# Log an event
event = {
    "gesture": "Help",
    "expression": "Sad",
    "confidence": 0.88
}
response = requests.post(f"{api_url}/log_event", json=event)
print(response.json())

# Get events
response = requests.get(f"{api_url}/events?limit=10")
for event in response.json():
    print(f"{event['gesture']} - {event['expression']}")
```

### cURL
```bash
# Check health
curl http://localhost:8000/health

# Log event
curl -X POST http://localhost:8000/log_event \
  -H "Content-Type: application/json" \
  -d '{"gesture":"Yes","expression":"Neutral","confidence":0.95}'

# Get events
curl http://localhost:8000/events?limit=5
```

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push and create a Pull Request

---

## 📄 License

MIT License - See LICENSE file for details

---

## 🙏 Acknowledgments

- **MediaPipe:** Google's excellent computer vision framework
- **scikit-learn:** Machine learning library
- **FastAPI:** Modern Python web framework
- **OpenCV:** Computer vision library

---

## 📞 Support

For issues, questions, or suggestions, please open an issue on GitHub.

---

**Made with ❤️ for accessibility and inclusivity**
