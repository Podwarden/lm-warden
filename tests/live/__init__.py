"""Live priority + queueing proof against a real vllm-warden deployment.

Opt-in only (`pytest -m live`); never collected by the default `pytest tests`
/ `pytest tests/unit` run -- see pyproject.toml's `-m 'not live'` addopt and
`tests/live/conftest.py::pytest_collection_modifyitems`. See README.md for
the env contract and how to run this against a real deployment.
"""
