import logging
from config import MEM0_CONFIG

logger = logging.getLogger(__name__)

memory = None

try:
    from mem0 import Memory
    memory = Memory.from_config(MEM0_CONFIG)
    logger.info("✅ Mem0 инициализирован")
except Exception as e:
    logger.error(f"⚠️ Mem0 недоступен: {e}")
