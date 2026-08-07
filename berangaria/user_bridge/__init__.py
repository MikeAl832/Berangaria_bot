"""Read-only MTProto bridge: surface other bots' group messages into Bot API pipeline.

Failures here must never take down the main bot. When disabled or misconfigured the
package is a no-op.
"""

from berangaria.user_bridge.runtime import start_user_bridge, stop_user_bridge

__all__ = ["start_user_bridge", "stop_user_bridge"]
