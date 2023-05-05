"# backend-python-FastAP" 

## 0. Create Virtual Environment
python -m venv venv

### Before Install Module and start Serve
venv\Scripts\activate
## 1. Install dependent for project
pip install -r requirements.txt

## 2. Start server
uvicorn main:app --reload

## 3. Visit document
http://127.0.0.1:8000/docs