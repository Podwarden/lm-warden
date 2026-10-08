CREATE INDEX idx_model_samples_minute ON model_samples(minute);
CREATE INDEX idx_gpu_samples_minute ON gpu_samples(minute);
CREATE INDEX idx_api_tokens_prefix ON api_tokens(prefix);
CREATE INDEX idx_api_tokens_revoked ON api_tokens(revoked_at);
