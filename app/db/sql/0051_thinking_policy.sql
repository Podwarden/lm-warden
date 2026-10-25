-- Thinking policy (spec 2026-10-25 §17.3): 'client' = the request decides (default), 'off' = the proxy
-- forces chat_template_kwargs.enable_thinking=false. Effective policy: off if the token OR the model says off.
ALTER TABLE api_tokens ADD COLUMN thinking_policy TEXT NOT NULL DEFAULT 'client';
ALTER TABLE models ADD COLUMN thinking_policy TEXT NOT NULL DEFAULT 'client';
-- 1 = this request's thinking was forced off by policy, 0 = not forced, NULL = unknown (older rows).
ALTER TABLE request_history ADD COLUMN thinking_forced_off INTEGER;
