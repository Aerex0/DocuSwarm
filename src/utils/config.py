"""Configuration management utility"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    """Global configuration manager"""

    def __init__(self):
        self.project_root = Path(__file__).parent.parent.parent
        self.configs_dir = self.project_root / "configs"

        # Load all configs
        self.groq_config = self._load_yaml("groq.yaml")
        self.agents_config = self._load_yaml("agents.yaml")
        self.chromadb_config = self._load_yaml("chromadb.yaml")
        self.llamaparse_config = self._load_yaml("llamaparse.yaml")

        # Environment variables
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        self.llamaparse_api_key = os.getenv("LLAMAPARSE_API_KEY")
        self.tavily_api_key = os.getenv("TAVILY_API_KEY")

        # Paths
        self.data_dir = Path(os.getenv("DATA_DIR", "./data"))
        self.chromadb_path = Path(os.getenv("CHROMADB_PATH", "./data/chromadb"))
        self.cache_dir = Path(os.getenv("CACHE_DIR", "./data/cache"))
        self.processed_dir = Path(os.getenv("PROCESSED_DIR", "./data/processed"))

        # Ensure directories exist
        self.chromadb_path.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        # System settings
        self.environment = os.getenv("ENVIRONMENT", "development")
        self.log_level = os.getenv("LOG_LEVEL", "INFO")
        self.debug = os.getenv("DEBUG", "False").lower() == "true"

        # Model settings
        self.model_reasoning = os.getenv(
            "GROQ_MODEL_REASONING", "llama-3.3-70b-versatile"
        )
        self.model_embedding = os.getenv("GROQ_MODEL_EMBEDDING", "nomic-embed-text")
        self.model_fallback = os.getenv("GROQ_MODEL_FALLBACK", "llama-3.1-8b-instant")

        # LLM parameters
        self.temperature = float(os.getenv("TEMPERATURE", "0.1"))
        self.max_tokens = int(os.getenv("MAX_TOKENS", "8000"))
        self.top_p = float(os.getenv("TOP_P", "0.95"))

        # Retrieval settings
        self.default_top_k = int(os.getenv("DEFAULT_TOP_K", "10"))
        self.similarity_threshold = float(os.getenv("SIMILARITY_THRESHOLD", "0.7"))
        self.use_hybrid_search = (
            os.getenv("USE_HYBRID_SEARCH", "True").lower() == "true"
        )

        # Chunking settings
        self.default_chunk_size = int(os.getenv("DEFAULT_CHUNK_SIZE", "1024"))
        self.default_chunk_overlap = int(os.getenv("DEFAULT_CHUNK_OVERLAP", "128"))
        self.chunking_strategy = os.getenv("CHUNKING_STRATEGY", "multimodal")

    def _load_yaml(self, filename: str) -> Dict[str, Any]:
        """Load YAML configuration file"""
        filepath = self.configs_dir / filename
        if not filepath.exists():
            return {}

        with open(filepath, "r") as f:
            return yaml.safe_load(f) or {}

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key"""
        return getattr(self, key, default)


# Global config instance
config = Config()
