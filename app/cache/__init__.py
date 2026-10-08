"""HF cache management subsystem.

Operator-facing API + UI for inspecting and reclaiming HuggingFace model
cache that vllm-warden has downloaded to ``Settings.hf_cache_dir``.

See ``docs/superpowers/specs/2026-05-20-hf-cache-management-design.md``
for the design rationale and vllm-warden#114 for the issue.
"""
