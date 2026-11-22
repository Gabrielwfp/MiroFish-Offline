"""
Configuration Management
Loads configuration from .env file in project root directory
"""

import os
from dotenv import load_dotenv

# Load .env file from project root
# Path: MiroFish/.env (relative to backend/app/config.py)
project_root_env = os.path.join(os.path.dirname(__file__), '../../.env')

if os.path.exists(project_root_env):
    load_dotenv(project_root_env, override=True)
else:
    # If no .env in root, try to load environment variables (for production)
    load_dotenv(override=True)


# Computed at module level: class-body comprehensions can't see class attributes
_LLM_OPERATIONS = ('ontology', 'ner', 'profile', 'sim_config', 'report', 'chat')
_NUM_CTX_DEFAULT = int(os.environ.get('OLLAMA_NUM_CTX_DEFAULT') or os.environ.get('OLLAMA_NUM_CTX') or '32768')
_NUM_CTX_BY_OPERATION = {
    op: int(os.environ.get(f'OLLAMA_NUM_CTX_{op.upper()}') or _NUM_CTX_DEFAULT) for op in _LLM_OPERATIONS
}


class Config:
    """Flask configuration class"""

    # Flask configuration
    SECRET_KEY = os.environ.get('SECRET_KEY', 'mirofish-secret-key')
    DEBUG = os.environ.get('FLASK_DEBUG', 'True').lower() == 'true'

    # JSON configuration - disable ASCII escaping to display Chinese directly (not as \uXXXX)
    JSON_AS_ASCII = False

    # LLM configuration (unified OpenAI format)
    LLM_API_KEY = os.environ.get('LLM_API_KEY')
    LLM_BASE_URL = os.environ.get('LLM_BASE_URL', 'http://localhost:11434/v1')
    LLM_MODEL_NAME = os.environ.get('LLM_MODEL_NAME', 'qwen2.5:32b')

    # Transport for chat calls: 'openai' (OpenAI-compatible /v1, default) or
    # 'ollama_native' (Ollama /api/chat, the only way options.num_ctx is honored;
    # Ollama's MLX runner ignores num_ctx on either API, llama.cpp/GGUF honors it).
    LLM_API_MODE = os.environ.get('LLM_API_MODE', 'openai').strip().lower()
    LLM_OPERATIONS = _LLM_OPERATIONS
    # Context window per operation (ollama_native only). OLLAMA_NUM_CTX_<OPERATION>
    # overrides the default; the legacy OLLAMA_NUM_CTX is accepted as the default.
    OLLAMA_NUM_CTX_DEFAULT = _NUM_CTX_DEFAULT
    OLLAMA_NUM_CTX_BY_OPERATION = _NUM_CTX_BY_OPERATION
    # Preflight token estimate = chars / ratio. Measured real ratios per operation in the
    # GGUF lab run: 4.0 (sim_config) to 5.5 (NER); 3.5 keeps a ~12% margin below the worst case.
    CONTEXT_CHARS_PER_TOKEN = float(os.environ.get('CONTEXT_CHARS_PER_TOKEN', '3.5'))

    # Simulation agents (CAMEL/OASIS, always OpenAI-compatible /v1, which cannot send
    # num_ctx): model name may differ from LLM_MODEL_NAME, e.g. an Ollama model derived
    # with `PARAMETER num_ctx` so /v1 calls load it at that context.
    SIMULATION_LLM_MODEL_NAME = os.environ.get('SIMULATION_LLM_MODEL_NAME') or LLM_MODEL_NAME
    # Send reasoning_effort="none" from simulation agents (posts + interviews)
    SIMULATION_DISABLE_THINKING = os.environ.get('SIMULATION_DISABLE_THINKING', 'false').lower() == 'true'

    @classmethod
    def num_ctx_for(cls, operation):
        """Context window for an LLM operation (falls back to the default)."""
        return cls.OLLAMA_NUM_CTX_BY_OPERATION.get(operation, cls.OLLAMA_NUM_CTX_DEFAULT)

    # Neo4j configuration
    NEO4J_URI = os.environ.get('NEO4J_URI', 'bolt://localhost:7687')
    NEO4J_USER = os.environ.get('NEO4J_USER', 'neo4j')
    NEO4J_PASSWORD = os.environ.get('NEO4J_PASSWORD', 'mirofish')

    # Embedding configuration
    EMBEDDING_MODEL = os.environ.get('EMBEDDING_MODEL', 'qwen3-embedding:0.6b')
    # OpenAI-compatible base URL including /v1 (omlx, Ollama and others serve /v1/embeddings)
    EMBEDDING_BASE_URL = os.environ.get('EMBEDDING_BASE_URL', 'http://localhost:11434/v1')
    EMBEDDING_API_KEY = os.environ.get('EMBEDDING_API_KEY') or LLM_API_KEY
    # Must match the model's output size AND the Neo4j vector indexes
    # (qwen3-embedding:0.6b=1024, nomic-embed-text=768). Every response is checked against it.
    EMBEDDING_DIMENSIONS = int(os.environ.get('EMBEDDING_DIMENSIONS', '1024'))

    # File upload configuration
    MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50MB
    UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), '../uploads')
    ALLOWED_EXTENSIONS = {'pdf', 'md', 'txt', 'markdown'}

    # Text processing configuration
    DEFAULT_CHUNK_SIZE = 500  # Default chunk size
    DEFAULT_CHUNK_OVERLAP = 50  # Default overlap size

    # OASIS simulation configuration
    OASIS_DEFAULT_MAX_ROUNDS = int(os.environ.get('OASIS_DEFAULT_MAX_ROUNDS', '10'))
    OASIS_SIMULATION_DATA_DIR = os.path.join(os.path.dirname(__file__), '../uploads/simulations')

    # OASIS platform available actions configuration
    OASIS_TWITTER_ACTIONS = [
        'CREATE_POST', 'LIKE_POST', 'REPOST', 'FOLLOW', 'DO_NOTHING', 'QUOTE_POST'
    ]
    OASIS_REDDIT_ACTIONS = [
        'LIKE_POST', 'DISLIKE_POST', 'CREATE_POST', 'CREATE_COMMENT',
        'LIKE_COMMENT', 'DISLIKE_COMMENT', 'SEARCH_POSTS', 'SEARCH_USER',
        'TREND', 'REFRESH', 'DO_NOTHING', 'FOLLOW', 'MUTE'
    ]

    # Report Agent configuration
    REPORT_AGENT_MAX_TOOL_CALLS = int(os.environ.get('REPORT_AGENT_MAX_TOOL_CALLS', '5'))
    REPORT_AGENT_MAX_REFLECTION_ROUNDS = int(os.environ.get('REPORT_AGENT_MAX_REFLECTION_ROUNDS', '2'))
    REPORT_AGENT_TEMPERATURE = float(os.environ.get('REPORT_AGENT_TEMPERATURE', '0.5'))

    @classmethod
    def validate(cls):
        """Validate required configuration"""
        errors = []
        if not cls.LLM_API_KEY:
            errors.append("LLM_API_KEY not configured (set to any non-empty value, e.g. 'ollama')")
        if not cls.NEO4J_URI:
            errors.append("NEO4J_URI not configured")
        if not cls.NEO4J_PASSWORD:
            errors.append("NEO4J_PASSWORD not configured")
        if cls.LLM_API_MODE not in ('openai', 'ollama_native'):
            errors.append(f"LLM_API_MODE must be 'openai' or 'ollama_native', got '{cls.LLM_API_MODE}'")
        return errors
