-- Prefix-cache hit per finished request.
--
-- cached_tokens: MEASURED, usage.prompt_tokens_details.cached_tokens from the
-- engine's final usage frame (vLLM reports it only when launched with
-- --enable-prompt-tokens-details). NULL = the engine did not say; not 0.
-- cache_est_tokens: what the dashboard ESTIMATED while the request was in
-- flight (app/proxy/dp_affinity.py::PrefixMemory), kept so the estimate can be
-- audited against the measurement.
ALTER TABLE request_history ADD COLUMN cached_tokens INTEGER;
ALTER TABLE request_history ADD COLUMN cache_est_tokens INTEGER;
