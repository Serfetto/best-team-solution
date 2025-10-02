Simple Streamlit UI for testing the Reviews API

Prerequisites
- Python 3.10+
- FastAPI service running locally (see `service/api.py`)

Install
- pip install streamlit requests

Run
- streamlit run ui_streamlit/app.py
- Optionally set `REVIEWS_API_URL` env var (defaults to http://127.0.0.1:8000)

Notes
- This UI only calls the API; it does not modify project code.
- Endpoints used: `/analyze`, `/taxonomy` (GET/PUT), `/report`.
