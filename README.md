# Python Code Generation Assistant

A professional Python code generation agent that produces efficient, well-documented Python code based on user requirements. Built with FastAPI, Azure OpenAI GPT-4.1, and robust observability, it validates requirements, requests clarification if ambiguous, and enforces runtime guardrails for safety and compliance.

---

## Quick Start

### 1. Create a virtual environment:
```
python -m venv .venv
```

### 2. Activate the virtual environment:

**Windows:**
```
.venv\Scripts\activate
```

**macOS/Linux:**
```
source .venv/bin/activate
```

### 3. Install dependencies:
```
pip install -r requirements.txt
```

### 4. Environment setup:
Copy `.env.example` to `.env` and fill in all required values.
```
cp .env.example .env
```

### 5. Running the agent

- Direct execution:
  ```
  python code/agent.py
  ```
- As a FastAPI server:
  ```
  uvicorn code.agent:app --reload --host 0.0.0.0 --port 8000
  ```

---

## Environment Variables

**Agent Identity**
- `AGENT_NAME`
- `AGENT_ID`
- `PROJECT_NAME`
- `PROJECT_ID`
- `SERVICE_NAME`
- `SERVICE_VERSION`

**General**
- `ENVIRONMENT`

**Azure Key Vault**
- `USE_KEY_VAULT`
- `KEY_VAULT_URI`
- `AZURE_USE_DEFAULT_CREDENTIAL`

**Azure Authentication**
- `AZURE_TENANT_ID`
- `AZURE_CLIENT_ID`
- `AZURE_CLIENT_SECRET`

**LLM Configuration**
- `MODEL_PROVIDER`
- `LLM_MODEL`
- `LLM_TEMPERATURE`
- `LLM_MAX_TOKENS`
- `LLM_MODELS` (optional, JSON array for token pricing)

**API Keys / Secrets**
- `OPENAI_API_KEY`
- `AZURE_OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`
- `GOOGLE_API_KEY`
- `AZURE_CONTENT_SAFETY_KEY`

**Service Endpoints**
- `AZURE_OPENAI_ENDPOINT`
- `AZURE_CONTENT_SAFETY_ENDPOINT`
- `AZURE_SEARCH_ENDPOINT`
- `AZURE_SEARCH_API_KEY`
- `AZURE_SEARCH_INDEX_NAME`

**Observability DB (Azure SQL)**
- `OBS_DATABASE_TYPE`
- `OBS_AZURE_SQL_SERVER`
- `OBS_AZURE_SQL_DATABASE`
- `OBS_AZURE_SQL_PORT`
- `OBS_AZURE_SQL_USERNAME`
- `OBS_AZURE_SQL_PASSWORD`
- `OBS_AZURE_SQL_SCHEMA`
- `OBS_AZURE_SQL_TRUST_SERVER_CERTIFICATE`

**Agent-Specific**
- `VALIDATION_CONFIG_PATH`
- `CONTENT_SAFETY_ENABLED` (optional)
- `CONTENT_SAFETY_SEVERITY_THRESHOLD` (optional)

---

## API Endpoints

### **GET** `/health`
Health check endpoint.

**Response:**
```
{
  "status": "ok"
}
```

---

### **POST** `/query`
Generate Python code based on user requirements.

**Request body:**
```
{
  "requirements": "string (required)"
}
```

**Response:**
```
{
  "success": true|false,
  "code": "string|null",          // Generated Python code or clarification/fallback message
  "error": "string|null",         // Error message if any
  "tool_calls_made": []           // Always an empty list for this agent
}
```

---

### **Exception Handlers**

#### Validation Error (422)
**Response:**
```
{
  "success": false,
  "error": "Input validation error",
  "details": [...],
  "tips": [
    "Ensure your JSON is well-formed.",
    "Check for missing required fields.",
    "Remove trailing commas and fix quotes if present.",
    "Requirements must not be empty and less than 50,000 characters."
  ]
}
```

#### Malformed JSON (400)
**Response:**
```
{
  "success": false,
  "error": "Malformed JSON in request body",
  "details": "...",
  "tips": [
    "Ensure your JSON is well-formed.",
    "Check for missing or extra commas, brackets, or quotes.",
    "Use double quotes for keys and string values."
  ]
}
```

---

## Running Tests

### 1. Install test dependencies (if not already installed):
```
pip install pytest pytest-asyncio
```

### 2. Run all tests:
```
pytest tests/
```

### 3. Run a specific test file:
```
pytest tests/test_<module_name>.py
```

### 4. Run tests with verbose output:
```
pytest tests/ -v
```

### 5. Run tests with coverage report:
```
pip install pytest-cov
pytest tests/ --cov=code --cov-report=term-missing
```

---

## Deployment with Docker

### 1. Prerequisites: Ensure Docker is installed and running.

### 2. Environment setup: Copy `.env.example` to `.env` and configure all required environment variables.

### 3. Build the Docker image:
```
docker build -t Python Code Generation Assistant -f deploy/Dockerfile .
```

### 4. Run the Docker container:
```
docker run -d --env-file .env -p 8000:8000 --name Python Code Generation Assistant Python Code Generation Assistant
```

### 5. Verify the container is running:
```
docker ps
```

### 6. View container logs:
```
docker logs Python Code Generation Assistant
```

### 7. Stop the container:
```
docker stop Python Code Generation Assistant
```

---

## Notes

- All run commands must use the `code/` prefix (e.g., `python code/agent.py`, `uvicorn code.agent:app ...`).
- See `.env.example` for all required and optional environment variables.
- The agent requires access to LLM API keys and (optionally) Azure SQL for observability.
- For production, configure Key Vault and secure credentials as needed.

---

**Python Code Generation Assistant** — Professional, safe, and efficient Python code generation from natural language requirements.