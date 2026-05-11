import asyncio
import asyncio as _asyncio

import time as _time
from observability.observability_wrapper import (
    trace_agent, trace_step, trace_step_sync, trace_model_call, trace_tool_call,
)
from config import settings as _obs_settings

import logging as _obs_startup_log
from contextlib import asynccontextmanager
from observability.instrumentation import initialize_tracer

_obs_startup_logger = _obs_startup_log.getLogger(__name__)

from modules.guardrails.content_safety_decorator import with_content_safety

GUARDRAILS_CONFIG = {
    'content_safety_enabled': True,
    'runtime_enabled': True,
    'content_safety_severity_threshold': 3,
    'check_toxicity': True,
    'check_jailbreak': True,
    'check_pii_input': False,
    'check_credentials_output': True,
    'check_output': True,
    'check_toxic_code_output': True,
    'sanitize_pii': False
}

import logging
import json
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ValidationError, field_validator
from pathlib import Path

import openai

from config import Config

# =========================
# Constants
# =========================

SYSTEM_PROMPT = (
    "You are a professional Python developer and code generation assistant. "
    "Your primary responsibility is to generate accurate, efficient, and well-documented Python code based on the user's requirements. "
    "Always ensure that your code adheres to Python best practices and PEP 8 standards. "
    "If the user's requirement is unclear or ambiguous, politely request clarification before proceeding. "
    "Provide code with appropriate comments and explanations when necessary. "
    "If you cannot fulfill the request, respond with a clear and professional message indicating the limitation."
)
OUTPUT_FORMAT = (
    "- Output only the Python code in a properly formatted code block.\n"
    "- Include comments and docstrings where appropriate.\n"
    "- If clarification is needed, ask the user for more details.\n"
    "- If unable to generate code, provide a professional fallback message."
)
FALLBACK_RESPONSE = (
    "I'm unable to generate Python code for the provided requirement. Please provide more details or clarify your request."
)
FEW_SHOT_EXAMPLES = [
    "Write a Python function to calculate the factorial of a number.",
    "Create a script that reads a CSV file and prints the number of rows."
]
VALIDATION_CONFIG_PATH = Config.VALIDATION_CONFIG_PATH or str(Path(__file__).parent / "validation_config.json")

# =========================
# Input/Output Models
# =========================

class GenerateCodeRequest(BaseModel):
    requirements: str = Field(..., description="Natural language code requirements")

    @field_validator("requirements")
    @classmethod
    def validate_requirements(cls, v):
        if not v or not v.strip():
            raise ValueError("Requirements must not be empty.")
        if len(v.strip()) > 50000:
            raise ValueError("Requirements must be less than 50,000 characters.")
        return v.strip()

class QueryResponse(BaseModel):
    success: bool = Field(..., description="Whether the request was successful")
    code: Optional[str] = Field(None, description="Generated Python code or clarification/fallback message")
    error: Optional[str] = Field(None, description="Error message if any")
    tool_calls_made: Optional[List[str]] = Field(None, description="List of tool calls made (always empty for this agent)")

# =========================
# Utility: LLM Output Sanitizer
# =========================

import re as _re

_FENCE_RE = _re.compile(r"```(?:\w+)?\s*\n(.*?)```", _re.DOTALL)
_LONE_FENCE_START_RE = _re.compile(r"^```\w*$")
_WRAPPER_RE = _re.compile(
    r"^(?:"
    r"Here(?:'s| is)(?: the)? (?:the |your |a )?(?:code|solution|implementation|result|explanation|answer)[^:]*:\s*"
    r"|Sure[!,.]?\s*"
    r"|Certainly[!,.]?\s*"
    r"|Below is [^:]*:\s*"
    r")",
    _re.IGNORECASE,
)
_SIGNOFF_RE = _re.compile(
    r"^(?:Let me know|Feel free|Hope this|This code|Note:|Happy coding|If you)",
    _re.IGNORECASE,
)
_BLANK_COLLAPSE_RE = _re.compile(r"\n{3,}")

def _strip_fences(text: str, content_type: str) -> str:
    """Extract content from Markdown code fences."""
    fence_matches = _FENCE_RE.findall(text)
    if fence_matches:
        if content_type == "code":
            return "\n\n".join(block.strip() for block in fence_matches)
        for match in fence_matches:
            fenced_block = _FENCE_RE.search(text)
            if fenced_block:
                text = text[:fenced_block.start()] + match.strip() + text[fenced_block.end():]
        return text
    lines = text.splitlines()
    if lines and _LONE_FENCE_START_RE.match(lines[0].strip()):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()

def _strip_trailing_signoffs(text: str) -> str:
    """Remove conversational sign-off lines from the end of code output."""
    lines = text.splitlines()
    while lines and _SIGNOFF_RE.match(lines[-1].strip()):
        lines.pop()
    return "\n".join(lines).rstrip()

@with_content_safety(config=GUARDRAILS_CONFIG)
def sanitize_llm_output(raw: str, content_type: str = "code") -> str:
    """
    Generic post-processor that cleans common LLM output artefacts.
    Args:
        raw: Raw text returned by the LLM.
        content_type: 'code' | 'text' | 'markdown'.
    Returns:
        Cleaned string ready for validation, formatting, or direct return.
    """
    if not raw:
        return ""
    text = _strip_fences(raw.strip(), content_type)
    text = _WRAPPER_RE.sub("", text, count=1).strip()
    if content_type == "code":
        text = _strip_trailing_signoffs(text)
    return _BLANK_COLLAPSE_RE.sub("\n\n", text).strip()

# =========================
# Service Classes
# =========================

class LLMService:
    """
    Handles interaction with Azure OpenAI GPT-4.1, including prompt construction, few-shot examples, and response parsing.
    """
    def __init__(self):
        self.client = None

    def get_client(self):
        if self.client is None:
            api_key = Config.AZURE_OPENAI_API_KEY
            if not api_key:
                raise ValueError("AZURE_OPENAI_API_KEY not configured")
            self.client = openai.AsyncAzureOpenAI(
                api_key=api_key,
                api_version="2024-02-01",
                azure_endpoint=Config.AZURE_OPENAI_ENDPOINT,
            )
        return self.client

    @with_content_safety(config=GUARDRAILS_CONFIG)
    @trace_agent(agent_name=_obs_settings.AGENT_NAME, project_name=_obs_settings.PROJECT_NAME)
    async def call_llm(
        self,
        system_prompt: str,
        user_prompt: str,
        few_shot_examples: Optional[List[str]] = None,
        clarified_requirement: Optional[str] = None,
    ) -> str:
        """
        Calls Azure OpenAI GPT-4.1 with constructed prompt and parameters.
        Handles API errors, timeouts, retries up to 3 times with exponential backoff.
        """
        client = self.get_client()
        model = Config.LLM_MODEL or "gpt-4.1"
        _llm_kwargs = Config.get_llm_kwargs()
        messages = [
            {"role": "system", "content": system_prompt + "\n\nOutput Format: " + OUTPUT_FORMAT},
        ]
        if few_shot_examples:
            for ex in few_shot_examples:
                messages.append({"role": "user", "content": ex})
        if clarified_requirement:
            messages.append({"role": "user", "content": clarified_requirement})
        else:
            messages.append({"role": "user", "content": user_prompt})

        last_exception = None
        for attempt in range(3):
            _t0 = _time.time()
            try:
                response = await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    **_llm_kwargs
                )
                content = response.choices[0].message.content
                try:
                    trace_model_call(
                        provider="azure",
                        model_name=model,
                        prompt_tokens=getattr(getattr(response, "usage", None), "prompt_tokens", 0) or 0,
                        completion_tokens=getattr(getattr(response, "usage", None), "completion_tokens", 0) or 0,
                        latency_ms=int((_time.time() - _t0) * 1000),
                        response_summary=content[:200] if content else "",
                    )
                except Exception:
                    pass
                return content
            except Exception as e:
                last_exception = e
                await self._backoff(attempt)
        raise RuntimeError(f"LLM API call failed after 3 attempts: {last_exception}")

    async def _backoff(self, attempt: int):
        delay = 2 ** attempt
        await self._sleep(delay)

    async def _sleep(self, seconds: int):
        await asyncio.sleep(seconds)

class RequirementValidator:
    """
    Validates user requirements for clarity and actionability; requests clarification if ambiguous.
    """
    def validate_requirement(self, user_requirement: str) -> (bool, Optional[str]):
        """
        Checks if user requirement is clear and actionable.
        Returns (valid/invalid, clarified requirement or prompt for clarification).
        """
        if not user_requirement or not user_requirement.strip():
            return False, "Please provide a clear and specific Python code requirement."
        # Simple ambiguity check: too short, vague, or generic
        ambiguous_phrases = [
            "something", "anything", "code", "script", "program", "do it", "make it work", "fix", "help", "example"
        ]
        text = user_requirement.strip().lower()
        if len(text) < 10 or any(phrase in text for phrase in ambiguous_phrases):
            return False, "Your requirement is unclear. Please provide more details about the desired functionality, inputs, and expected outputs."
        return True, user_requirement.strip()

class ResponseFormatter:
    """
    Formats LLM output according to output_format instructions (code block, comments, docstrings, fallback messages).
    """
    def format_response(self, llm_response: str) -> str:
        """
        Ensures code is in a properly formatted code block, includes comments/docstrings, and adheres to output_format instructions.
        Returns fallback message if formatting fails.
        """
        try:
            code = sanitize_llm_output(llm_response, content_type="code")
            if not code or len(code.strip()) == 0:
                return FALLBACK_RESPONSE
            return code
        except Exception:
            return FALLBACK_RESPONSE

class AuditLogger:
    """
    Logs requests, responses, errors, and system events for compliance and monitoring.
    """
    def __init__(self):
        self.logger = logging.getLogger("agent.audit")
        self.logger.setLevel(logging.INFO)

    def log_event(self, event_type: str, event_data: Dict[str, Any]):
        """
        Logs events for audit and monitoring.
        """
        try:
            self.logger.info(f"{event_type}: {json.dumps(event_data, default=str)}")
        except Exception as e:
            self.logger.warning(f"Failed to log event {event_type}: {e}")

class ErrorHandler:
    """
    Manages errors, retries, and fallback behaviors according to business rules.
    """
    def handle_error(self, error_code: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Implements retry logic, fallback behaviors, and hard stop triggers.
        """
        if error_code == "INVALID_REQUIREMENT":
            return "Your requirement is unclear. Please provide more details about the desired Python code."
        elif error_code == "CODE_GENERATION_ERROR":
            return FALLBACK_RESPONSE
        else:
            return "An unexpected error occurred. Please try again later."

class ToolRegistry:
    """
    Manages OpenAI function-calling tools (empty for this agent, but included for extensibility).
    """
    def __init__(self):
        self.tools = {}

    def register_tool(self, name: str, tool):
        self.tools[name] = tool

    def get_tool(self, name: str):
        return self.tools.get(name)

# =========================
# Main Agent Class
# =========================

class PythonCodeGenerationAgent:
    """
    Orchestrates the end-to-end flow: receives user input, validates requirements, invokes LLM, formats output, handles errors.
    """
    def __init__(self):
        self.llm_service = LLMService()
        self.validator = RequirementValidator()
        self.formatter = ResponseFormatter()
        self.logger = AuditLogger()
        self.error_handler = ErrorHandler()
        self.tool_registry = ToolRegistry()

    @with_content_safety(config=GUARDRAILS_CONFIG)
    async def process_user_query(self, user_requirement: str) -> Dict[str, Any]:
        """
        Receives and processes user input, orchestrates validation and code generation.
        Returns dict with keys: success, code, error, tool_calls_made.
        """
        async with trace_step(
            "validate_requirement",
            step_type="parse",
            decision_summary="Validate user requirement for clarity",
            output_fn=lambda r: f"valid={r[0]}"
        ) as step:
            is_valid, clarified = self.validator.validate_requirement(user_requirement)
            step.capture((is_valid, clarified))
            self.logger.log_event("requirement_validation", {
                "input": user_requirement,
                "is_valid": is_valid,
                "clarified": clarified
            })
            if not is_valid:
                error_msg = self.error_handler.handle_error("INVALID_REQUIREMENT", {"input": user_requirement})
                return {
                    "success": False,
                    "code": None,
                    "error": error_msg,
                    "tool_calls_made": []
                }

        async with trace_step(
            "llm_code_generation",
            step_type="llm_call",
            decision_summary="Call LLM to generate Python code",
            output_fn=lambda r: f"llm_response={str(r)[:80]}"
        ) as step:
            try:
                _t0 = _time.time()
                llm_response = await self.llm_service.call_llm(
                    system_prompt=SYSTEM_PROMPT,
                    user_prompt=user_requirement,
                    few_shot_examples=FEW_SHOT_EXAMPLES,
                    clarified_requirement=clarified
                )
                step.capture(llm_response)
                self.logger.log_event("llm_response", {
                    "llm_response": llm_response[:200]
                })
            except Exception as e:
                self.logger.log_event("llm_error", {
                    "error": str(e),
                    "input": user_requirement
                })
                error_msg = self.error_handler.handle_error("CODE_GENERATION_ERROR", {"error": str(e)})
                return {
                    "success": False,
                    "code": None,
                    "error": error_msg,
                    "tool_calls_made": []
                }

        async with trace_step(
            "format_response",
            step_type="format",
            decision_summary="Format LLM output as Python code",
            output_fn=lambda r: f"code_len={len(r) if r else 0}"
        ) as step:
            try:
                code = self.formatter.format_response(llm_response)
                step.capture(code)
                self.logger.log_event("code_generated", {
                    "code": code[:200]
                })
                return {
                    "success": True,
                    "code": code,
                    "error": None,
                    "tool_calls_made": []
                }
            except Exception as e:
                self.logger.log_event("formatting_error", {
                    "error": str(e),
                    "llm_response": llm_response[:200]
                })
                error_msg = self.error_handler.handle_error("CODE_GENERATION_ERROR", {"error": str(e)})
                return {
                    "success": False,
                    "code": None,
                    "error": error_msg,
                    "tool_calls_made": []
                }

# =========================
# FastAPI App & Endpoints
# =========================

@asynccontextmanager
async def _obs_lifespan(application):
    """Initialise observability on startup, clean up on shutdown."""
    try:
        _obs_startup_logger.info('')
        _obs_startup_logger.info('========== Agent Configuration Summary ==========')
        _obs_startup_logger.info(f'Environment: {getattr(Config, "ENVIRONMENT", "N/A")}')
        _obs_startup_logger.info(f'Agent: {getattr(Config, "AGENT_NAME", "N/A")}')
        _obs_startup_logger.info(f'Project: {getattr(Config, "PROJECT_NAME", "N/A")}')
        _obs_startup_logger.info(f'LLM Provider: {getattr(Config, "MODEL_PROVIDER", "N/A")}')
        _obs_startup_logger.info(f'LLM Model: {getattr(Config, "LLM_MODEL", "N/A")}')
        _cs_endpoint = getattr(Config, 'AZURE_CONTENT_SAFETY_ENDPOINT', None)
        _cs_key = getattr(Config, 'AZURE_CONTENT_SAFETY_KEY', None)
        if _cs_endpoint and _cs_key:
            _obs_startup_logger.info('Content Safety: Enabled (Azure Content Safety)')
            _obs_startup_logger.info(f'Content Safety Endpoint: {_cs_endpoint}')
        else:
            _obs_startup_logger.info('Content Safety: Not Configured')
        _obs_startup_logger.info('Observability Database: Azure SQL')
        _obs_startup_logger.info(f'Database Server: {getattr(Config, "OBS_AZURE_SQL_SERVER", "N/A")}')
        _obs_startup_logger.info(f'Database Name: {getattr(Config, "OBS_AZURE_SQL_DATABASE", "N/A")}')
        _obs_startup_logger.info('===============================================')
        _obs_startup_logger.info('')
    except Exception as _e:
        _obs_startup_logger.warning('Config summary failed: %s', _e)

    _obs_startup_logger.info('')
    _obs_startup_logger.info('========== Content Safety & Guardrails ==========')
    if GUARDRAILS_CONFIG.get('content_safety_enabled'):
        _obs_startup_logger.info('Content Safety: Enabled')
        _obs_startup_logger.info(f'  - Severity Threshold: {GUARDRAILS_CONFIG.get("content_safety_severity_threshold", "N/A")}')
        _obs_startup_logger.info(f'  - Check Toxicity: {GUARDRAILS_CONFIG.get("check_toxicity", False)}')
        _obs_startup_logger.info(f'  - Check Jailbreak: {GUARDRAILS_CONFIG.get("check_jailbreak", False)}')
        _obs_startup_logger.info(f'  - Check PII Input: {GUARDRAILS_CONFIG.get("check_pii_input", False)}')
        _obs_startup_logger.info(f'  - Check Credentials Output: {GUARDRAILS_CONFIG.get("check_credentials_output", False)}')
    else:
        _obs_startup_logger.info('Content Safety: Disabled')
    _obs_startup_logger.info('===============================================')
    _obs_startup_logger.info('')

    _obs_startup_logger.info('========== Initializing Agent Services ==========')
    # 1. Observability DB schema (imports are inside function — only needed at startup)
    try:
        from observability.database.engine import create_obs_database_engine
        from observability.database.base import ObsBase
        import observability.database.models  # noqa: F401
        _obs_engine = create_obs_database_engine()
        ObsBase.metadata.create_all(bind=_obs_engine, checkfirst=True)
        _obs_startup_logger.info('✓ Observability database connected')
    except Exception as _e:
        _obs_startup_logger.warning('✗ Observability database connection failed (metrics will not be saved)')
    # 2. OpenTelemetry tracer (initialize_tracer is pre-injected at top level)
    try:
        _t = initialize_tracer()
        if _t is not None:
            _obs_startup_logger.info('✓ Telemetry monitoring enabled')
        else:
            _obs_startup_logger.warning('✗ Telemetry monitoring disabled')
    except Exception as _e:
        _obs_startup_logger.warning('✗ Telemetry monitoring failed to initialize')
    _obs_startup_logger.info('=================================================')
    _obs_startup_logger.info('')
    yield

app = FastAPI(
    title="Python Code Generation Assistant",
    description="Generates Python code based on user requirements using Azure OpenAI GPT-4.1.",
    version=Config.SERVICE_VERSION if hasattr(Config, "SERVICE_VERSION") else "1.0.0",
    lifespan=_obs_lifespan
)

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}

@app.exception_handler(ValidationError)
@with_content_safety(config=GUARDRAILS_CONFIG)
async def validation_exception_handler(request: Request, exc: ValidationError):
    return JSONResponse(
        status_code=422,
        content={
            "success": False,
            "error": "Input validation error",
            "details": exc.errors(),
            "tips": [
                "Ensure your JSON is well-formed.",
                "Check for missing required fields.",
                "Remove trailing commas and fix quotes if present.",
                "Requirements must not be empty and less than 50,000 characters."
            ]
        }
    )

@app.exception_handler(json.decoder.JSONDecodeError)
@with_content_safety(config=GUARDRAILS_CONFIG)
async def json_decode_exception_handler(request: Request, exc: json.decoder.JSONDecodeError):
    return JSONResponse(
        status_code=400,
        content={
            "success": False,
            "error": "Malformed JSON in request body",
            "details": str(exc),
            "tips": [
                "Ensure your JSON is well-formed.",
                "Check for missing or extra commas, brackets, or quotes.",
                "Use double quotes for keys and string values."
            ]
        }
    )

@app.post("/query", response_model=QueryResponse)
@with_content_safety(config=GUARDRAILS_CONFIG)
async def query_endpoint(req: GenerateCodeRequest):
    """
    Main endpoint for Python code generation.
    """
    agent = PythonCodeGenerationAgent()
    try:
        result = await agent.process_user_query(user_requirement=req.requirements)
        # Sanitize output before returning
        code = sanitize_llm_output(result.get("code", ""), content_type="code") if result.get("code") else None
        return QueryResponse(
            success=result.get("success", False),
            code=code,
            error=result.get("error"),
            tool_calls_made=result.get("tool_calls_made", [])
        )
    except Exception as e:
        logging.getLogger("agent").error(f"Unhandled error in /query: {e}", exc_info=True)
        return QueryResponse(
            success=False,
            code=None,
            error="An unexpected error occurred. Please try again later.",
            tool_calls_made=[]
        )

# =========================
# Entrypoint
# =========================

async def _run_agent():
    """Entrypoint: runs the agent with observability (trace collection only)."""
    import uvicorn

    # Unified logging config — routes uvicorn, agent, and observability through
    # the same handler so all telemetry appears in a single consistent stream.
    _LOG_CONFIG = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "()": "uvicorn.logging.DefaultFormatter",
                "fmt": "%(levelprefix)s %(name)s: %(message)s",
                "use_colors": None,
            },
            "access": {
                "()": "uvicorn.logging.AccessFormatter",
                "fmt": '%(levelprefix)s %(client_addr)s - "%(request_line)s" %(status_code)s',
            },
        },
        "handlers": {
            "default": {
                "formatter": "default",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stderr",
            },
            "access": {
                "formatter": "access",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
            },
        },
        "loggers": {
            "uvicorn":        {"handlers": ["default"], "level": "INFO", "propagate": False},
            "uvicorn.error":  {"level": "INFO"},
            "uvicorn.access": {"handlers": ["access"], "level": "INFO", "propagate": False},
            "agent":          {"handlers": ["default"], "level": "INFO", "propagate": False},
            "__main__":       {"handlers": ["default"], "level": "INFO", "propagate": False},
            "observability": {"handlers": ["default"], "level": "INFO", "propagate": False},
            "config": {"handlers": ["default"], "level": "INFO", "propagate": False},
            "azure":   {"handlers": ["default"], "level": "WARNING", "propagate": False},
            "urllib3": {"handlers": ["default"], "level": "WARNING", "propagate": False},
        },
    }

    config = uvicorn.Config(
        "agent:app",
        host="0.0.0.0",
        port=8080,
        reload=False,
        log_level="info",
        log_config=_LOG_CONFIG,
    )
    server = uvicorn.Server(config)
    await server.serve()


if __name__ == "__main__":
    _asyncio.run(_run_agent())