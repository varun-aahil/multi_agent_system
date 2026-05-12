# Intelligent Multi-Agent Travel Planner ✈️

A production-ready Multi-Agent System (MAS) built with **LangChain** and **LangGraph**, designed to automate travel planning. The system utilizes multiple specialized AI agents working in coordination to transform a simple user request into a comprehensive travel guide, complete with a day-by-day itinerary and a detailed budget breakdown.

The system is powered by local LLMs via **Ollama**, ensuring privacy and cost-efficiency.

## 🚀 Key Features

- **Multi-Agent Orchestration**: Uses LangGraph to manage a stateful workflow between specialized agents.
- **Local LLM Support**: Optimized for local execution using Ollama (default: `gemma3:12b`).
- **Stateful Design**: Maintains a robust state throughout the planning process, handling validation and error recovery.
- **Dual Interface**:
    - **CLI Mode**: Interactive terminal-based planning.
    - **Web Mode**: A polished UI built with Streamlit.
- **Specialized Agents**:
    - **Extraction Agent**: Analyzes user intent and extracts key entities (destination, duration, budget).
    - **Itinerary Agent**: Generates structured, time-stamped daily plans.
    - **Budget Agent**: Estimates costs and provides a markdown breakdown.
    - **Formatter Agent**: Consolidates agent outputs into a beautiful final guide.

---

## 🏗️ Architecture

The system follows a directed graph workflow:

1.  **START** ➡️ **Analyzer** (Extracts details & validates request)
2.  **Analyzer** ➡️ **Planner** (Generates Itinerary)
3.  **Planner** ➡️ **Budget** (Estimates Costs)
4.  **Budget** ➡️ **Formatter** (Creates Final Guide)
5.  **Formatter** ➡️ **END**

*Conditional paths exist to handle invalid requests or errors gracefully.*

---

## 🛠️ Setup & Installation

### 1. Prerequisites
- **Python 3.10+**
- **Ollama**: [Download and install Ollama](https://ollama.com/).
- **Model**: Pull the recommended model:
  ```bash
  ollama pull gemma3:12b
  ```

### 2. Install Dependencies
Clone this repository and install the required packages:
```bash
pip install -r requirements.txt
```

---

## 🔧 Troubleshooting

- **Ollama Connection**: Ensure Ollama is running locally before starting the app. The system defaults to `http://localhost:11434`.
- **Model Not Found**: If the system cannot find `gemma3:12b`, it will attempt to use the first available model in your Ollama list. You can pull more models using `ollama pull <model_name>`.
- **Dependency Issues**: If you encounter import errors, ensure you are using a virtual environment and have run `pip install -r requirements.txt`.

---

## 📖 Usage

### Option 1: Streamlit Web Interface (Recommended)
Run the interactive web application:
```bash
streamlit run streamlit_app.py
```

### Option 2: CLI Interface
Run the system directly in your terminal:
```bash
python multi_agent_system.py
```

---

## 📂 Project Structure

- `multi_agent_system.py`: Core logic, LangGraph definition, and Agent implementations.
- `streamlit_app.py`: The Streamlit frontend.
- `requirements.txt`: Project dependencies.

---

## ⚙️ Configuration
The system automatically detects available models in your local Ollama instance. You can select your preferred model via the Streamlit sidebar or let the CLI pick the best available one.

---

Developed with ❤️ using LangChain & LangGraph.
