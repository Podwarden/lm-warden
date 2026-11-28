-- 0068_reasoning_budget.sql
--
-- Reasoning budget (#348): bound a reasoning model's thinking.
-- The effective budget is min(model, key, request); NULL = no limit at that layer.
--
-- models.reasoning_budget          tokens of thinking a request may use, whatever the client asks
-- models.reasoning_budget_mode     what happens when it is spent: 'force_answer' (close the
--                                  reasoning block, let the model answer) or 'stop' (end the request)
-- models.reasoning_effort_budgets  JSON {"low": n, "medium": n, "high": n, "minimal": n}: what a
--                                  request's reasoning_effort means in tokens on this model; NULL = defaults
-- api_tokens.reasoning_budget      the key's cap
-- api_tokens.reasoning_budget_mode the key's mode; NULL = use the model's
--
-- The history outcome needs no column: a request the budget decided is recorded as
-- request_history.finish_reason = 'reasoning_budget'.
-- Rollback: SQLite >= 3.35 ALTER TABLE ... DROP COLUMN on each of the five columns.
ALTER TABLE models ADD COLUMN reasoning_budget INTEGER;
ALTER TABLE models ADD COLUMN reasoning_budget_mode TEXT NOT NULL DEFAULT 'force_answer';
ALTER TABLE models ADD COLUMN reasoning_effort_budgets TEXT;
ALTER TABLE api_tokens ADD COLUMN reasoning_budget INTEGER;
ALTER TABLE api_tokens ADD COLUMN reasoning_budget_mode TEXT;
