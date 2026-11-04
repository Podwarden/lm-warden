-- Prompt diet (2026-10-28): a key's opt-in cut of unused tool definitions and carried reasoning
-- from chat bodies before the engine. JSON {"strip_tools": [names], "strip_reasoning": bool}; NULL = off.
ALTER TABLE api_tokens ADD COLUMN prompt_diet TEXT;
-- What the diet cut from this request: JSON {"tools", "tools_chars", "reason_chars"}; NULL = nothing cut.
ALTER TABLE request_history ADD COLUMN prompt_diet TEXT;
-- Debug capture: epoch seconds until which this key's request bodies and responses are recorded
-- (app/proxy/capture.py); NULL = off.
ALTER TABLE api_tokens ADD COLUMN capture_until REAL;
