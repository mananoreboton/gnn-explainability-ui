# Explainability UI for GNN models

## Demo
You can see the demo [here](https://explainer.m30m.me)

## How to run this project

### Production (built frontend + backend)
1. Clone the repo
2. Install Python dependencies: `uv sync` ([uv](https://docs.astral.sh/uv/) required)
3. Build the frontend: `cd web && npm install && npm run build && cd ..`
4. Start the server: `uv run python web_service.py`
5. Open `http://localhost:5000` (server listens on all interfaces)

### Development (hot-reload frontend)
1. Start the backend: `uv run python web_service.py` (in project root)
2. In another terminal: `cd web && npm run serve`
3. Open `http://localhost:8080` (frontend talks to backend at `http://localhost:5000`)


## How to add new experiments
There is an experiment template in the `experiments` folder which you can use as a starting point.
Copy `experiment.py` and modify it as necessary.
