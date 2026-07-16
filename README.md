# 🤖 Emergent Behavior Sandbox (SwarmScope)

An advanced flight simulator and pre-deployment safety dashboard for multi-agent LLM systems. It executes Monte Carlo simulations locally to detect conversation vulnerabilities—such as **Loops**, **Deadlocks**, **Collusion**, and **Goal Drift**—before shipping agents to production.

---

## 🚀 Key Features

* **Pluggable Agent Simulation Engine**: Simulates conversation streams between $N$ agents locally or using cloud backends.
* **Deterministic Fallback (DummyBackend)**: Works out-of-the-box with zero API keys or external LLM setups by utilizing rule-based dialogue.
* **Multi-Backend Support**: Out-of-the-box support for Ollama, OpenAI, and Anthropic API connections.
* **Pluggable Vulnerability Detectors**:
  * **Loop Detector**: Uses Jaccard word-set similarities to track cycles or repeating turns.
  * **Deadlock Detector**: Monitors numerical proposal stagnation and refusal-oriented vocabulary.
  * **Collusion Detector**: Evaluates semantic agreement metrics that violate original constraints or objectives.
  * **Goal Drift Detector**: Tracks semantic divergence from defined goals across history.
* **Monte Carlo Batch Analytics**: Runs automated batches with randomized temperature jitter, shuffled agent sequences, and phrasing changes to capture statistical failure frequencies.
* **Streamlit Safety Dashboard**: Visualizes communication networks, score timelines, and historical run data.
* **Scikit-Learn Predictive Model**: Pre-deploys a lightweight Random Forest classifier trained on run parameters (backend, scenario, turns, temperature) to calculate risk probability instantly.
* **Counterfactual Replay Engine**: Dialogue debugger that automatically replays failed conversation streams under modified parameters (restricted memory, temperature drops, prompt adjustments) to determine the best safety mitigation strategy.

---

## 🗺️ Project Structure

```text
swarmscope/
├── requirements.txt            # Project Python dependencies
├── README.md                   # Elevator pitch & setup instructions
├── Dockerfile                  # API and Dashboard image definition
├── docker-compose.yml          # Container configuration orchestrator
├── simulate.py                 # Simulation runner CLI entrypoint
├── train_model.py              # ML classifier trainer CLI entrypoint
├── scenarios/                  # YAML-based Scenario Library
│   ├── negotiation.yaml        # Vintage car buyer-seller negotiation
│   ├── resource_allocation.yaml # Grid energy sharing dispute
│   └── debate_consensus.yaml   # Software database architecture debate
├── sandbox/                    # Core package
│   ├── config.py               # Environmental configuration loader
│   ├── schemas.py              # Pydantic typing and validation models
│   ├── agents.py               # Agent identity, memory, and routing class
│   ├── simulation.py           # Core simulator engine and run sequence
│   ├── prompts.py              # Dialogue prompt templates
│   ├── utils.py                # File handling and ID utilities
│   ├── api.py                  # FastAPI routing & API endpoint definition
│   ├── backends/               # LLM adapters (Dummy, Ollama, OpenAI, Anthropic)
│   ├── storage/                # DuckDB persistence driver
│   ├── detectors/              # Pluggable vulnerability scanners
│   ├── analytics/              # Batch runner implementation
│   ├── predictive_model/       # Scikit-Learn training pipelines
│   └── dashboard/              # Streamlit frontend views
└── tests/                      # Automated unit/integration test suites
```

---

## 🛠️ Getting Started (5-Minute Setup)

### 1. Installation

Set up a virtual environment and install the required libraries:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Run a Local Simulation

To execute a single conversation run locally using the rule-based backend:

```bash
python simulate.py --scenario negotiation --turns 10 --backend dummy
```

To run a batch Monte Carlo simulation of 5 runs with parameter jitter:

```bash
python simulate.py --scenario negotiation --turns 10 --backend dummy --batch 5
```

### 3. Launch the Safety Dashboard

Launch the interactive dashboard to visualize graphs and run history:

```bash
streamlit run sandbox/dashboard/app.py
```

### 4. Train the Predictive Model

Train the machine learning model on historical database entries (or synthetic features if empty):

```bash
python train_model.py
```

---

## 🐳 Docker Deployment

To spin up both the FastAPI backend (`port 8000`) and the Streamlit UI (`port 8501`) together:

```bash
docker-compose up --build
```

---

## 🧪 Running Tests

Execute the test suites with pytest:

```bash
pytest tests/
```

---

## 🔌 Client SDKs (Polyglot Integration)

We ship client SDK adapters so you can audit agent applications in any environment. The SDKs communicate directly with the SwarmScope FastAPI server:

### 1. TypeScript & Node.js SDK (`sdk/ts/` & `sdk/js/`)
A fully-typed client wrapper using native fetch. To run the Node.js demo script:
```bash
node sdk/js/demo.js
```

### 2. Go SDK (`sdk/go/`)
A native Go implementation for auditing microservices. To verify compilation:
```bash
go run sdk/go/client.go
```

### 3. Rust Client Library (`sdk/rust/`)
A Tokio-based asynchronous client wrapper using reqwest. To verify compilation:
```bash
cd sdk/rust && cargo check
```
