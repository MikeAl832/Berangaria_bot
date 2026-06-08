import logging
from config import MEM0_CONFIG, DEBUG
import json

logger = logging.getLogger(__name__)

memory = None

try:
    from mem0 import Memory
    
    if DEBUG:
        logger.info("🔧 [yellow]Mem0 конфигурация:[/]")
        config_debug = {
            "llm_provider": MEM0_CONFIG["llm"]["provider"],
            "llm_model": MEM0_CONFIG["llm"]["config"]["model"],
            "embedder_provider": MEM0_CONFIG["embedder"]["provider"],
            "embedder_model": MEM0_CONFIG["embedder"]["config"]["model"],
            "vector_store": MEM0_CONFIG["vector_store"]["provider"],
        }
        logger.info(f"   {json.dumps(config_debug, indent=2, ensure_ascii=False)}")
    
    memory = Memory.from_config(MEM0_CONFIG)
    logger.info("✅ [bright_green]Mem0 инициализирован[/]")
except Exception as e:
    logger.error(f"⚠️ Mem0 недоступен: [red]{e}[/]")

