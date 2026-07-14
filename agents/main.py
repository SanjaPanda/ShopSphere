import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

from shared.logging_config import setup_logger

load_dotenv()

from agents.graph import router_graph

logger = setup_logger("main")

initial_state = {
    "user_query": "Is WFH available?",
}

logger.info("main started with query: %s", initial_state["user_query"])
for event in router_graph.stream(initial_state):
    logger.info("main received event: %s", event)
    print(event)
