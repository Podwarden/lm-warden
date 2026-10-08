# app/cache_obs/__init__.py
"""Prompt-cache observation: what was reusable, what was reused, and why not.

Pure and import-light on purpose: nothing here imports app.proxy or app.stats,
so a future cross-node router can use the index on its own (spec §6).
"""
