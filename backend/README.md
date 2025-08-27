# Backend

Python service built with FastAPI and LangGraph.

## Running Locally
1. Create and activate a virtual environment.
2. Install dependencies: `pip install -r requirements.txt` (requirements file to be defined).
3. Set required environment variables:
   - `OPENAI_API_KEY` – API key for model access.
4. Start the API server: `uvicorn app:app --reload`.

## Development
- Lint with `flake8`.
- Run tests with `pytest`.
