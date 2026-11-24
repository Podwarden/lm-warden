-- Model aliases (owner decisions 2026-11-23/24, see the bundled-vLLM plan):
-- a named, first-class object clients call. An alias holds an ORDERED list of
-- target model ids (loaded or not) plus optional proxy-side settings; the
-- proxy resolves a request's `model` against alias names FIRST (an alias may
-- share the name of a model and then wins), then served names, and picks the
-- first target that is loaded.
--
-- targets holds model ids WITHOUT a foreign key, like router_rules: a deleted
-- target must not cascade the alias away. The model delete route calls
-- ModelAliasRepo.drop_target, which removes the id from every alias's target
-- list -- the alias itself survives, with an empty/shrunken list.
--
-- params is a JSON object mapping a request parameter to {"value": ...,
-- "mode": "default" | "force" | "cap"}; the proxy never trusts the shape it
-- stores, it re-validates on write (app/aliases/schemas.py).
CREATE TABLE model_aliases (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    targets TEXT NOT NULL DEFAULT '[]',
    guard TEXT NOT NULL DEFAULT 'inherit',
    thinking_policy TEXT NOT NULL DEFAULT 'inherit',
    params TEXT NOT NULL DEFAULT '{}',
    enabled INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

