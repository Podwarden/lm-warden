-- Operator-visible outcome of the boot-time parallel-layout reconcile (#286).
--
-- layout_notice_level: 'info' (a legacy row was promoted/cleaned) or 'warning'
-- (a row still carries parallel flags in extra_args that could not be fixed
-- automatically and needs an operator edit). NULL = nothing to show.
-- layout_notice: the human-readable message. Cleared when the row is clean.
ALTER TABLE models ADD COLUMN layout_notice_level TEXT;
ALTER TABLE models ADD COLUMN layout_notice TEXT;
