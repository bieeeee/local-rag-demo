from pathlib import Path

# LLM used by Ollama to generate answers. Must match a model name already pulled via `ollama pull`.
MODEL = "gemma3:4b"

BASE_DIR = Path(__file__).resolve().parent.parent
DOCS_DIR = BASE_DIR / "docs"            # Root directory of documents to index
DB_DIR = BASE_DIR / "chroma_db"         # Vector DB storage location (created on first run if missing)
INSTRUCTION_FILE = BASE_DIR / "instruction.yaml"