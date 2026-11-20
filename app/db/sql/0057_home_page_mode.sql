-- 0057_home_page_mode.sql
--
-- Spec 2026-11-04 (site split and home page): `home_page_mode`
-- ('off' | 'builtin' | 'custom') replaces `landing_page_enabled`, and
-- `home_page_off_action` ('console' | '404') says what `off` shows.
--
-- No existing install may change what its address shows:
--   landing_page_enabled 'true'              -> builtin
--   'false', or no row at all                -> off
-- A NEW install defaults to builtin (the welcome page). This reverses the
-- 2026-10-07 "new installs start with the site off".
--
-- "New" cannot be read from the data (a fresh database runs 0001..0057 in one
-- batch, so 0020 has already seeded landing_page_enabled='false' by now).
-- apply_migrations() records whether `schema_migrations` was empty before the
-- call in temp.migration_context.fresh, and this file reads it.
--
-- An interrupted first boot (a fresh DB that crashed mid-migration) leaves
-- schema_migrations non-empty, so the next boot reads it as an upgrade and
-- gets `off` -- conservative by design.
--
-- `landing_page_enabled` is left in place, inert: nothing reads it after this.
--
-- Rollback: DELETE FROM settings WHERE key IN ('home_page_mode', 'home_page_off_action');

INSERT OR IGNORE INTO settings(key, value)
SELECT 'home_page_mode',
       CASE
         WHEN (SELECT fresh FROM temp.migration_context) = 1 THEN 'builtin'
         WHEN (SELECT lower(trim(value)) FROM settings WHERE key = 'landing_page_enabled')
              IN ('true', '1', 'yes', 'on') THEN 'builtin'
         ELSE 'off'
       END;

INSERT OR IGNORE INTO settings(key, value) VALUES ('home_page_off_action', 'console');
