-- Measured prefix-cache hits per model per minute, for the Stats page's
-- cached-vs-computed prompt chart.
--
-- cached_tokens: sum of the engine-reported usage.prompt_tokens_details
-- .cached_tokens of the requests that FINISHED in the minute (a request is
-- credited to its finish minute, like prompt_tokens). cached_measured_requests:
-- how many of the minute's requests reported it at all. A request whose engine
-- did not report it adds nothing to either and so counts as computed; rows
-- written before this migration read 0/0, which the UI renders as "not
-- measured", not as "nothing was cached".
ALTER TABLE model_samples ADD COLUMN cached_tokens INTEGER NOT NULL DEFAULT 0;
ALTER TABLE model_samples ADD COLUMN cached_measured_requests INTEGER NOT NULL DEFAULT 0;
