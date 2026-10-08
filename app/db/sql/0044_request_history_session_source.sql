-- Which signal identified the conversation of a request, and its parent.
--
-- session_source: claude_code_header | anthropic_metadata | x_session_id |
-- x_session_affinity | session_id_header | prompt_cache_key | openai_user |
-- client_request_id | prompt_hash (app/proxy/session_id.py). NULL when the
-- request carried nothing to key on, and on rows from before this migration.
-- parent_session_id: the parent of a subagent's session (OpenCode, Codex);
-- display only.
ALTER TABLE request_history ADD COLUMN session_source TEXT;
ALTER TABLE request_history ADD COLUMN parent_session_id TEXT;
