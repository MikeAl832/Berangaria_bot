import logging
from config import MEM0_CONFIG
import json
import time

logger = logging.getLogger(__name__)

memory = None


def initialize_memory(attempts: int = 1, delay_seconds: float = 2.0):
    """Инициализирует Mem0 с retry, чтобы пережить старт Qdrant в Docker."""
    global memory
    if memory is not None:
        return memory

    config_debug = {
        "llm_provider": MEM0_CONFIG["llm"]["provider"],
        "llm_model": MEM0_CONFIG["llm"]["config"]["model"],
        "embedder_provider": MEM0_CONFIG["embedder"]["provider"],
        "embedder_model": MEM0_CONFIG["embedder"]["config"]["model"],
        "vector_store": MEM0_CONFIG["vector_store"]["provider"],
    }
    logger.debug(f"🔧 Mem0 конфигурация:\n{json.dumps(config_debug, indent=2, ensure_ascii=False)}")

    total_attempts = max(1, attempts)
    for attempt in range(1, total_attempts + 1):
        try:
            from mem0 import Memory

            memory = Memory.from_config(MEM0_CONFIG)
            logger.info("✅ [bright_green]Mem0 инициализирован[/]")
            return memory
        except Exception as exc:
            memory = None
            if attempt >= total_attempts:
                logger.error(
                    f"⚠️ Mem0 недоступен после {total_attempts} попыток: [red]{exc}[/]"
                )
                return None
            logger.warning(
                f"⏳ Mem0/Qdrant ещё не готовы: {exc}; "
                f"повтор {attempt + 1}/{total_attempts} через {delay_seconds:.1f}с"
            )
            time.sleep(max(0.0, delay_seconds))

    return None
