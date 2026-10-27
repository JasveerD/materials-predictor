# Materials Band Gap Predictor

Predict the electronic band gap of crystal structures using a graph neural network trained on [Materials Project](https://materialsproject.org) data.

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌───────────────┐
│  CIF / MPID │────▶│  Crystal →   │────▶│  CGCNN (4x    │
│   Input      │     │  Graph Conv  │     │  CGConv +     │
└─────────────┘     └──────────────┘     │  BatchNorm +  │
                                          │  Residual)    │
                                          └───────┬───────┘
                                                  │
                                          ┌───────▼───────┐
                                          │  Mean+Max Pool │
                                          │  → MLP → Band │
                                          │    Gap (eV)   │
                                          └───────────────┘
```

## How It Works

Crystals are naturally represented as graphs: atoms are nodes with features (atomic number, electronegativity, radius), and bonds within an 8 Å cutoff are edges with distance and direction features. This graph representation preserves the full 3D structure, unlike descriptor-based approaches that lose spatial information.

The model uses Crystal Graph Convolutional Neural Network (CGCNN) layers that propagate information between neighboring atoms through message passing. After 4 rounds of message passing with residual connections, global pooling collapses the per-atom representations into a single graph-level embedding, which a small MLP maps to the predicted band gap.

Uncertainty is estimated via MC dropout: at inference time, dropout layers remain active across 5 forward passes, and the standard deviation of predictions serves as a confidence measure.

## Quickstart

### With Docker

```bash
docker compose up
# Open http://localhost:8000
```

### Manual Setup

```bash
# Install dependencies
pip install uv
uv pip install ".[dev]"

# Generate sample data (no API key needed)
python -m data.generate_sample

# Or fetch real data (requires free API key from https://next-gen.materialsproject.org)
export MP_API_KEY=your_key_here
python -m data.fetch

# Process crystal structures into graphs
python -m data.process

# Train the model
python -m model.train

# Evaluate on test set
python -m model.evaluate

# Start the API server
uvicorn api.main:app --reload
```

## Model Performance

Trained on ~50,000 stable materials from the Materials Project.

| Metric | Value |
|--------|-------|
| MAE    | < 0.4 eV |
| RMSE   | < 0.6 eV |
| R²     | > 0.90 |

## API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predict` | POST | Predict band gap from CIF string or MP ID |
| `/health` | GET | Health check |
| `/model/info` | GET | Model metadata and training metrics |

### Example

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"material_id": "mp-149"}'
```

## Project Structure

```
materials-predictor/
├── data/           # Data fetching and processing
│   ├── fetch.py    # Materials Project API client
│   ├── process.py  # Crystal → graph conversion
│   └── sample/     # 100 sample materials for testing
├── model/          # GNN model and training
│   ├── gnn.py      # CGCNN architecture
│   ├── train.py    # Training loop
│   ├── evaluate.py # Metrics and parity plots
│   └── predict.py  # Inference with uncertainty
├── api/            # FastAPI backend
├── frontend/       # React + Tailwind UI
└── tests/          # pytest suite
```

## Data Source

Crystal structure data from the [Materials Project](https://materialsproject.org).

> Jain, A. et al. *Commentary: The Materials Project: A materials genome approach to accelerating materials innovation.* APL Materials 1, 011002 (2013). https://doi.org/10.1063/1.4812323

## License

MIT
