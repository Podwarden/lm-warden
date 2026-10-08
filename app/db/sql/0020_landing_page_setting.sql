-- 0020_landing_page_setting.sql
--
-- Issue #155 — unified-port architecture; default changed 2026-10-07.
--
-- Seed the `landing_page_enabled` runtime setting. A NEW install starts with
-- the public website OFF: LM Warden is open source, and only lmwarden.com
-- wants a public marketing site at its root. Operators opt in by PATCHing
-- `/api/settings/runtime` with `{"landing_page_enabled": true}` (Settings ->
-- Networking).
--
-- Why editing an applied migration is safe here: app/db/migrations.py runs
-- each file once, recorded by filename in schema_migrations, with no
-- checksum. An existing install applied this file long ago (when it seeded
-- 'true') and never runs it again, so its stored value is untouched; only a
-- fresh database runs it and gets 'false'.
--
-- Rollback: `DELETE FROM settings WHERE key = 'landing_page_enabled';`
-- (a missing row also means off; see app/landing/routes.py::_is_enabled).

INSERT OR IGNORE INTO settings(key, value)
VALUES ('landing_page_enabled', 'false');
