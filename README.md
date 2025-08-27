# Interview-scheduling-agent-system

This project is the starting point for an intelligent agentic AI system. It provides a minimal backend and frontend that can be extended into a full interview scheduling assistant.

## Project Structure

- **backend** – [FastAPI](https://fastapi.tiangolo.com/) application that exposes a simple API.
- **frontend** – Static HTML page that fetches data from the backend and displays it in the browser.

## Backend

1. Install dependencies:
   ```bash
   pip install -r backend/requirements.txt
   ```
2. Run the development server:
   ```bash
   uvicorn backend.app:app --reload
   ```

The server will start on `http://localhost:8000/` and returns a greeting message.

## Frontend

Open `frontend/index.html` in a web browser. It will request the greeting from the backend and display it on the page.

## Next Steps

This repository currently contains only a minimal example. Future work may include:

- Implementing interview scheduling logic and storage.
- Expanding the frontend with a modern framework like React or Next.js.
- Integrating agentic AI components using LangChain, LangGraph, and Azure OpenAI.

Contributions are welcome!
