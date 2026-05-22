"""Kit --exec hook: toggle isaacsim.examples.interactive after N frames (Examples Browser / user_examples).

Registered from start_isaac_sim.sh via --/app/python/scriptFolders/99=... and --exec <this file>.
Disable: ISAAC_RELOAD_INTERACTIVE_EXAMPLES=0. Tune: ISAAC_RELOAD_INTERACTIVE_FRAMES=200 (default 120).
"""
from __future__ import annotations

import os

import carb
import omni.kit.app
from omni.kit.async_engine import run_coroutine

_EXT_ID = "isaacsim.examples.interactive"
_FRAMES = int(os.environ.get("ISAAC_RELOAD_INTERACTIVE_FRAMES", "120"))


async def _reload_after_frames() -> None:
    app = omni.kit.app.get_app()
    for _ in range(max(1, _FRAMES)):
        await app.next_update_async()
    mgr = app.get_extension_manager()
    if not mgr.is_extension_enabled(_EXT_ID):
        carb.log_warn(f"{_EXT_ID} is not enabled; skipping Examples Browser reload hook.")
        return
    carb.log_info(f"Reloading {_EXT_ID} (ISAAC_RELOAD_INTERACTIVE_EXAMPLES hook)")
    mgr.set_extension_enabled_immediate(_EXT_ID, False)
    await app.next_update_async()
    mgr.set_extension_enabled_immediate(_EXT_ID, True)


run_coroutine(_reload_after_frames())
