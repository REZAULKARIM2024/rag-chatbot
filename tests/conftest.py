import sys
from pathlib import Path

# Make project modules (ingest, chatbot, agent_graph) importable from tests/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
