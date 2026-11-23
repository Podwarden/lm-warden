# LM Warden KV store protocol, version 1, minor 1

The KV store daemon (`python -m lmw_kv.daemon`) keeps KV-cache blocks in a RAM tier
(a shared-memory file) and an encrypted NVMe tier. Any inference engine can talk to
it. The protocol knows only **namespaces**, **block keys**, **groups** and **byte
slots**; it has no notion of any engine, request or process role.

## Versions

`hello` returns `protocol` (the major version, 1) and, from minor 1 on, `minor`. A
daemon of minor 0 sends no `minor`; read that as 0. Inside protocol 1 every change is
backward compatible:

- Every field added in a minor is optional and has a stated default. A request without
  it means exactly what it meant before.
- **The daemon ignores fields it does not know.** An adapter written for a later minor
  still works against this daemon, and its extra fields change nothing.
- An op a daemon does not know is answered `unknown_op`. An adapter that needs a minor 1
  op checks `minor >= 1` first, or treats `unknown_op` like any failed call (fail open).

Minor 1 adds: `owner`, `model`, `role` and `rank` on `register_ns`; `tenant`,
`scope_tag` and `pos` on `begin_save`; `tenant` on `begin_load` and `record_hits`; the
ops `release_owner`, `retire_scopes` and `set_policy`; and the accounting maps in
`stats`.

## Transport

- A Unix stream socket, mode `0600`, path given by `--socket`. Put it on a
  host-local filesystem next to the RAM file (LM Warden uses `/dev/shm/lmw-kv.sock`),
  never on a volume another host or container also mounts: an engine must only ever
  reach the daemon whose RAM file it maps.
- The NVMe directory has one writer. The daemon holds an exclusive `flock` on
  `<nvme dir>/.lock` while it runs; a second daemon started on the same directory
  logs one warning and runs RAM-only. It tries the lock again every 30 seconds and
  turns NVMe on when it gets it.
- One JSON object per line, UTF-8, terminated by `\n`, in each direction. A line is
  at most 8 MiB (the daemon closes the connection on a longer one).
- A connection is a strictly ordered conversation: send a request, read its reply.
  The daemon answers the requests of one connection in order and may serve many
  connections at once. Do not pipeline more than you can match by `id`.
- Request: `{"op": "<op>", "id": <int>, ...fields}`. Reply: `{"id": <same>, "ok": true, ...fields}`
  or `{"id": <same>, "ok": false, "error": "<code>: <message>"}`. `id` is chosen by
  the client and echoed back; if the request was not valid JSON, `id` is `null`.
- A bad request never closes the connection.
- Byte values (namespace ids, block keys, scopes) are lower-case hex strings.
  Block keys, parent keys and namespace ids are exactly 32 bytes (64 hex chars); a
  scope is 0 to 255 bytes. Tenants, scope tags and models are exactly 16 bytes (32 hex
  chars), or empty: a missing field, `null` and `""` all mean empty. A position key is
  32 bytes, or empty. A value of the wrong length is `bad_request`.

Error codes: `bad_request` (missing or malformed field, rejected namespace or group
sizes), `unknown_op`, `internal` (the daemon failed; logged on its side).

## Memory model

`hello` returns `ram_path`, the shared-memory file that holds block data. The engine
maps it (read/write) itself. Offsets returned by `begin_save` and `begin_load` are
**byte offsets into that file**, one per group, each page-aligned. The daemon never
copies block data over the socket.

One size, one inode per daemon. The file keeps one size for the life of the daemon,
so a mapping stays valid. `ram_bytes` can be smaller than what was configured
(clamped to free shared memory) and is zero when there is no RAM tier. A daemon
that starts unlinks any file left at `ram_path` before it sizes and creates its
own, so each daemon run has a new inode. It never resizes or reuses a file that an
engine from an earlier run may still map. An engine checks, after every transfer,
that `ram_path` is still the inode it mapped (compare device and inode numbers). If
it is not, the daemon restarted: the engine stops using the store until it restarts
itself, and treats that transfer as failed.

## Concepts

- **Namespace**: everything that makes blocks interchangeable between engines: model
  repo and revision, tokenizer, KV dtype, block size, attention layout, KV groups,
  tensor/pipeline parallel size, `engine_family`, `layout_version`. Its id is
  `SHA-256("ns\0" + canonical JSON)` computed by the daemon; blocks of different
  namespaces never see each other.
- **Block key**: 32 bytes chained over the parent key, so a key commits to the whole
  prefix. See "Key derivation" below; the daemon never derives keys, adapters do.
- **Group**: one KV-cache group (for example full attention, or linear attention
  state). A block stores one payload per group; the sizes are fixed per namespace.
- **Scope**: opaque bytes stored with a block (for example a tenant or cache salt).
  The daemon stores it; it does not interpret it.
- **Tag**: an opaque string chosen by the client (for example an engine's request id)
  that ties `record_hits` and `pop_hits` together.
- **Owner** (minor 1): a string of at most 255 bytes naming who holds a namespace (for
  LM Warden, one loaded model row). A namespace keeps every owner that registered it.
  The empty owner `""` is the legacy owner of minor 0 adapters.
- **Model** (minor 1): 16 bytes that group the namespaces of one model (for example one
  per KV group and parallel rank). Accounting and fair shares work per model. Empty
  means the first 16 bytes of the namespace id: each namespace is its own model.
- **Tenant** (minor 1): 16 opaque bytes naming who pays for a block, for accounting and
  fair shares only. It is never part of a block key and never decides a hit. A block is
  charged to the tenant that saved it, then to the last tenant that loaded it ("last
  user pays"). Empty is the unattributed tenant.
- **Scope tag** (minor 1): 16 bytes, the public handle of a block's isolation scope,
  empty for a block anyone may share. The daemon only compares tags, so a client can
  retire a scope by its tag without ever sending the scope (see `retire_scopes`).
  "Tenants and isolation scopes" below gives LM Warden's encoding.
  An adapter must send `scope_tag` for every block saved under an isolation scope. The
  daemon cannot tell an isolated block from a sharing one, so an untagged block can be
  neither retired nor refused.
- **Position key** (minor 1): blocks of one prompt position across groups and ranks
  share `pos`, which defaults to `key`; the daemon evicts them together. An adapter
  that stores one prompt position in several namespaces (one per KV group or rank)
  sends the same `pos` for each, for example the key of that position computed with a
  namespace common to all groups and ranks. A state block then never leaves before its
  attention block.

## Key derivation

An adapter computes block keys itself (the daemon only stores them). Everything below
is exact; `lmw_kv/keys.py` is the reference implementation and the known-answer test
at the end lets you check yours. All hashes are SHA-256, HMAC is HMAC-SHA-256.

**Namespace id** (`ns`, 32 bytes; `register_ns` returns the same value, compute it to
check). Take the namespace object with the 11 fields of `register_ns`, serialise it as
JSON with keys sorted, separators `,` and `:` (no spaces), non-ASCII escaped as
`\uXXXX` (the default of Python's `json.dumps`), `kv_groups` as a list in order,
integers without a fraction. Then `ns = SHA256(b"ns\x00" + that_json_utf8)`.
`block_size` is **tokens per block**.

**Secret and salt.** The secret is 32 bytes. LM Warden's engine adapters read it from
the file named by the environment variable `LMW_KV_KEY_FILE`: text, first line
`v1:<64 hex chars>` followed by a newline (the LM Warden network key file). Missing or
malformed means no caching. The salt is 0 to 255 bytes of opaque data (for example a
tenant or cache salt; empty if the engine has none). Different salts give disjoint
keys. Encode a salt as `salt_field = bytes([len(salt)]) + salt`.

LM Warden's vLLM adapter builds the salt from the request's `lmw_scope` (hex, empty
by default) and vLLM's `cache_salt`. With no `cache_salt` (missing or empty) the salt
is the scope bytes. With one, the salt is
`SHA256(b"lmw-cache-salt\x00" + bytes([len(scope)]) + scope + utf8(cache_salt))`, so
two cache salts on one prefix never share a block.

**Root key** (the `parent` of the first block):
`root = HMAC(secret, b"root\x00" + salt_field + ns)`.

**Block key**:
`key = HMAC(secret, b"blk\x00" + salt_field + ns + parent + tokens)` where `parent`
is the previous block's key (32 bytes, the root key for block 0) and `tokens` is the
block's token ids, each as an unsigned 32-bit little-endian integer, concatenated.

**Which tokens.** Cut the prompt's token ids into consecutive blocks of exactly
`block_size` tokens starting at token 0. Only whole blocks have keys; a partial
trailing block is never stored or looked up. Block `i` covers tokens
`[i*block_size, (i+1)*block_size)` and chains on block `i-1`'s key.

**Lookup order.** Pass `keys` to `lookup` and `begin_load` ordered from the first
block (`i = 0`) onward; the reply counts or loads the longest stored **prefix**, and
stops at the first missing block. Multiply the count by `block_size` for tokens.

Known answer (namespace JSON above with `model_repo` `org/model`, `revision_sha` `r1`,
`tokenizer_sha` `tok`, `kv_dtype` `fp8`, `block_size` 16, `attention_layout` `hnd`,
`kv_groups` `["full","gdn"]`, `tp` 1, `pp` 1, `engine_family` `test`, `layout_version` 1;
secret = bytes 0 to 31; salt empty; block 0 = token ids 0..15, block 1 = 16..31):

```
ns:     1040228e1c4af1f3270767fb15964b9f3e83c48b22525ea45659c674c90221d4
secret: 000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f
root:   ffd19c51b3df702ea5c0e1d49ccbb1514127ed54fc13d77b3e1a631c680435e8
block0: d51d2dfcc1c7d9cfe31a2c1d11a05b24bf749635fad595ebf4247b0939b8ebfa
block1: 1598d74a71e78da4932e9d20b94f2395f03a22cba058ed7be646d9684a2771df
```

## Tenants and isolation scopes

LM Warden names each API key to the cache with values derived from the network key
(the 32-byte secret above, called `nk` here). The daemon never derives them; it only
stores and compares them. `lmw_kv/scopes.py` is the reference implementation. All
hashes are SHA-256, HMAC is HMAC-SHA-256 keyed with `nk`. `utf8(x)` is the UTF-8 bytes
of a string, `u8(n)` one byte, `u32le(n)` an unsigned 32-bit little-endian integer.
A network key that is not 32 bytes, or a negative epoch, is an error.

- `kv_tenant_id` is a key's stable cache identity. Traffic with no API key uses the
  tenant id `internal`, which is always sharing.
- **Tenant tag** (16 bytes), for accounting only, never part of a block key:
  `tenant = HMAC(nk, b"lmw-tenant\x00v1\x00" + utf8(kv_tenant_id))[:16]`.
- **Isolation scope** (32 bytes) of an isolated key at epoch `epoch` (an integer, 0 or
  more, written in decimal ASCII):
  `scope = HMAC(nk, b"lmw-scope\x00v1\x00" + utf8(kv_tenant_id) + b"\x00" + ascii(epoch))`.
  A sharing key's scope is empty. The scope is the `lmw_scope` of "Key derivation"
  (sent as hex).
- **Scope tag** (16 bytes), the public handle of a scope:
  `scope_tag = SHA256(b"lmw-scope-tag\x00" + scope)[:16]`, and empty for an empty scope.
- **GPU cache salt**, the engine `cache_salt` LM Warden sends: with `cs =
  utf8(client_salt)` (the client's salt, maybe empty),
  `"lmw1:" + hex(HMAC(nk, b"lmw-gpu\x00" + u8(len(scope)) + scope + u32le(len(cs)) + cs))`,
  hex in lower case. The scope length byte keeps the sharing form (empty scope) and the
  isolated form apart.
- **Model tag** (16 bytes), names a model for per-model accounting:
  `model_tag = SHA256(b"lmw-model\x00" + utf8(owner) + b"\x00" + utf8(revision))[:16]`.

Known answer (`nk` = bytes 0 to 31, `kv_tenant_id` `t1`, epoch 0; the GPU salt of that
scope with client salt `abc`; the sharing GPU salt of client salt `lmw1:x`; the model
tag of owner `m1`, revision `851bf6e8`):

```
tenant:       cc1426b65e852de2c716fa6e209b70e5
scope:        16077d3e8febec67e6006b804025ebe8bd8dc505c7187edfd737362d9e06f3ce
scope_tag:    e59b05605013afe4e2b61c139735fd4d
gpu_salt:     lmw1:e0438dafb1f71eefab002a6908359e0484c4d494676d89650f025b1b8ed1cf13
gpu_sharing:  lmw1:a49883a97abbb59ce52fbbb41b8afb5ea9e2576b3c35cf5233aab7efc474191f
model_tag:    6f466aae377f940c46bc6ad6566d9b46
```

These values are secrets of the key holder (except the model tag). They never appear in
a log line, a history row, a capture or an engine's argv.

On the wire (minor 1) the tenant tag goes in `tenant`, the scope tag in `scope_tag`,
the model tag in `model`. The daemon's error messages never repeat them.

## Ops

### `hello`
Request: `{"op":"hello","id":1}`
Reply: `protocol` (int, 1), `minor` (int, 1; missing on a minor 0 daemon), `ram_path`
(string), `ram_bytes` (int). Check `protocol == 1` before anything else. The daemon
creates the RAM file (and commits all of it) before it listens, so once `hello` answers
the file at `ram_path` exists with `ram_bytes` bytes (0: no RAM tier).

### `register_ns`
Request: `namespace` (object with exactly the fields `model_repo`, `revision_sha`,
`tokenizer_sha`, `kv_dtype`, `block_size`, `attention_layout`, `kv_groups` (list of
strings), `tp`, `pp`, `engine_family`, `layout_version`) and `groups` (object: group
name to the byte size of one block's payload for that group, positive integers), and
optionally `nvme` (boolean, default `true`). With `nvme` false the namespace's blocks
stay in RAM only: `commit_save` never queues an NVMe write for them, and `lookup`,
`begin_save` and `begin_load` ignore any copy already on NVMe.

Minor 1 fields, all optional:

- `owner` (string, at most 255 UTF-8 bytes, default `""`). The namespace remembers each
  owner with its `nvme` flag. NVMe is on for the namespace only while every owner
  wants it. The legacy owner `""` registering again sets its own flag again (the last
  registration wins, as in minor 0). `release_owner` drops an owner.
- `model` (hex, 16 bytes or empty, default empty: the first 16 bytes of the namespace
  id).
- `role` (string, at most 32 bytes, default `"attention"`): which part of the model the
  namespace holds, for example `attention`, `gdn_state` or `mtp`. Reported in `stats`.
- `rank` (int, 0 or more, default 0): the parallel rank, reported in `stats`.

`model`, `role` and `rank` are fixed by the first registration of the namespace; a
later registration with other values only adds its owner.
Reply: `ns` (hex id). Idempotent; registering again with different group sizes is
`bad_request`. A namespace whose blocks need larger slots than the RAM pool has may
wait for the pool to go idle, then run without RAM (lookups return 0) until it
registers again.

### `lookup`
Request: `ns`, `keys` (list of hex). Reply: `n`, the number of leading keys stored
(RAM or NVMe). Blocks still being saved do not count. An unknown namespace gives 0.

### `begin_save`
Request: `ns`, `key`, `parent` (the previous block's key, or the root key), `scope`,
and optionally (minor 1) `tenant` (16 bytes, default empty), `scope_tag` (16 bytes,
default empty: a sharing block) and `pos` (32 bytes, default `key`).
Reply: `offsets`: object group to byte offset, or `null` when the block already
exists or is being saved, or there is no free slot, or the namespace is unusable
(not registered, or released by its last owner). It is also `null` when `scope_tag`
is retired, and for any non-empty `scope_tag` while the retired scope file cannot be
read (see `retire_scopes`). An adapter must send `scope_tag` for every block saved under an isolation scope. The
daemon cannot tell an isolated block from a sharing one, so an untagged block can be
neither retired nor refused.
On an object, write each group's payload (exactly its registered size) at its
offset, then `commit_save`.

### `commit_save`
Request: `ns`, `key`. Reply: `{}`. The block becomes visible to lookups and is
queued for its NVMe write (only if its namespace has `nvme` true). Unknown or already committed blocks are ignored.

### `abort_save`
Request: `ns`, `key`. Reply: `{}`. Gives back a slot from `begin_save` that will not
be committed. A block that is not being saved is ignored.

### `begin_load`
Request: `ns`, `keys`, and optionally (minor 1) `tenant` (16 bytes, default empty).
A non-empty `tenant` takes over the charge of every loaded block, in RAM and on NVMe.
Reply: `handle` (int), `slots`: a list, one object
(group to byte offset) per key, for a prefix of `keys` (it may be shorter than
`lookup` reported, or empty), and `nvme` (int): how many of those blocks this call
read from NVMe (the rest were already in RAM). Blocks that live only on NVMe are
first read into RAM, so this call can take a while. The listed slots are pinned: they stay valid until
`end_load`. Read them from the file, then end the load.

### `end_load`
Request: `handle`. Reply: `{}`. Releases exactly the pins that `begin_load` took.
An unknown or ended handle is ignored.

### `record_hits`
Request: `tag`, `ram_tokens` (int), `nvme_tokens` (int), `load_s` (number), and
optionally (minor 1) `tenant` (16 bytes, default empty).
Reply: `{}`. Adds to the running totals kept for `tag`. The daemon keeps the most
recent 10 000 tags. The tokens are also added to the tenant's totals in `stats`.

### `pop_hits`
Request: `tag`. Reply: `hits`: `{"ram_tokens", "nvme_tokens", "load_s"}`, or `{}` (an
empty object) when nothing was recorded for `tag`. The tag is forgotten. A client tells
"nothing recorded" (`{}`) from a failed call (no reply, `ok: false`); daemons before this
revision replied `null` for nothing recorded, which a client reads as `{}`.

### `stats`
Reply: `stats`: `ram_bytes`, `ram_used`, `nvme_used`, `nvme_enabled`, `blocks`,
`evictions`, `nvme_evictions`. `nvme_used` is the bytes of the segment files on disk
(what the NVMe cap limits), not the live payload.

Minor 1 adds (byte counts are RAM slot bytes and NVMe record bytes on disk; tenants,
models and namespaces are hex):

- `policy` (`"lru"` or `"fair"`) and `nvme_cap_bytes`.
- `started_at` (float, Unix seconds, wall time): when the store started. Evictions,
  `evicted_by` and the hit token counts below are counted since then; a daemon that
  does not send it is read as "unknown".
- `retired_scopes` (int, how many scope tags are retired), `isolation_refused`
  (boolean: the retired scope file cannot be read, so isolated blocks are refused) and
  `retired_persist_degraded` (boolean: some retired tags are in memory only).
- `dead_bytes`: `{"ram", "nvme"}`, bytes of blocks no one can use again (retired scope,
  or a namespace with no owner). They are not counted below.
- `by_tenant`: tenant to `ram_bytes`, `nvme_bytes`, `ram_share`, `nvme_share` (its fair
  share in bytes, summed over models), `ram_evictions`, `nvme_evictions` (its blocks
  evicted), `evicted_by` (tenant to how many of its blocks that tenant's saves evicted),
  `ram_tokens`, `nvme_tokens` (from `record_hits`).
- `by_model`: model to `owners` (list), `last_owner` (the owner that registered the
  model last, `""` if none: it names a model whose owners are all released), `weight`, `ram_part`, `nvme_part` (its part of
  each tier in bytes), `ram_bytes`, `nvme_bytes`, and `tenants`: tenant to `ram_bytes`,
  `nvme_bytes`, `ram_share`, `nvme_share`, `ram_evictions`, `nvme_evictions` and
  `evicted_by` inside that model (the eviction fields default to 0 and `{}` when a
  daemon does not send them). A tenant whose blocks were all evicted keeps its row. The
  per-model eviction counts are kept for the 4096 most recently counted pairs; `by_tenant`
  keeps every count.
- `by_namespace`: namespace id to `model`, `role`, `rank`, `owners` (owner to its
  `nvme` flag), `live` (false once the last owner is released), `ram_bytes`,
  `nvme_bytes`.

### `release_owner` (minor 1)
Request: `owner` (string, at most 255 bytes). Reply: `released` (int): how many
namespaces were left with no owner. The owner is dropped from every namespace that
has it; an unknown owner is not an error (`released` 0). A namespace left with no owner
is unregistered until it registers again: `lookup` gives 0, `begin_load` gives no
slots, `begin_save` gives `null`, and its blocks are the first to be evicted under
`fair`. Until they are evicted its blocks stay, and registering again makes them
visible again.

### `retire_scopes` (minor 1)
Request: `scope_tags` (list of hex, each 16 bytes). Reply: `added` (int): how many of
them were not retired before. A retired scope tag is retired for good. Its blocks are
never found, loaded or saved again, in any tier, and are the first to be evicted under
`fair`. The op is idempotent: sending the same tags again adds 0, so a client can send
its whole list again after every reconnect (LM Warden does). A tag of the wrong length
anywhere in the list is `bad_request` and retires none of them.

The set takes effect in memory at once. While the NVMe tier is on it is also kept in
`<nvme_dir>/retired.db`, an SQLite file with one table `retired` (`tag` BLOB primary
key, `at` REAL seconds since the epoch), so it survives a restart. A daemon without
NVMe keeps it in memory only and relies on the client sending it again.

Failure is closed for isolation. If the file cannot be opened, read or written when
the daemon attaches it (at start, or when the NVMe tier comes on), the daemon logs an
error, keeps every tag it knows retired and reports `isolation_refused: true` in
`stats`. Every block with a non-empty scope tag is then refused (`lookup` stops at it,
`begin_load` stops before it, `begin_save` gives `null`) until a later attach works, in
practice a restart. Sharing blocks are not affected, and the daemon keeps serving. A
later write that fails keeps the new tags retired in memory, retries them on the next
write and reports `retired_persist_degraded: true`.

### `set_policy` (minor 1)
Request: `policy`, an object. Reply: `{}`. Every key is optional (missing means the
default), and keys the daemon does not know are ignored:

- `mode`: `"lru"` (the default) or `"fair"`.
- `idle_ram_s`, `idle_nvme_s`: seconds unused after which a block is idle and loses its
  protection; numbers above 0, default 7200 (2 hours) and 604800 (7 days).
- `owners`: owner to weight (integer 1 to 100). A model's weight is the largest weight
  of its owners, default 1.
- `tenants`: tenant (hex, 16 bytes) to `{"weight", "max_ram_pct", "max_nvme_pct"}`.
  `weight` is an integer 1 to 100, default 6 (LM Warden sends a key's scheduler
  priority plus 1). `max_ram_pct` and `max_nvme_pct` are hard caps, an integer 1 to 100
  (percent of that tier) or `null` for no cap (the default). A tenant not listed gets
  the defaults.

A policy with any bad value is `bad_request` and changes nothing: the previous policy
stays in force. A good one replaces the whole policy at once; the next eviction uses
it. The daemon starts with `{"mode": "lru"}`.

## Eviction

Under `lru` (the default) the daemon evicts exactly as minor 0: the oldest evictable
RAM block, and on NVMe the oldest segment file that can go. Tenants and models are
only counted.

Under `fair`, a **RAM** victim is a whole position: every block that shares its
`model` and `pos`, all evicted in one step, and only when each of them can go (not
pinned by a load, not being saved or written, and either on NVMe already or with no
child). The daemon takes, in order: dead blocks (retired scope or a namespace with no
owner), idle blocks (unused longer than `idle_ram_s`), then the least recently used
position of whoever is most over their share, and only then the oldest position
overall. A taker that is itself over its share, when nobody over has a position that can
go, never gets the oldest position overall: its save is refused, and a load (a block
promoted from NVMe by `begin_load`) takes only the oldest position whose every block is
also on NVMe, so its owner moves to the slower tier and loses nothing. The loader is
charged for the block and is evicted first afterwards. Each tier is split between models by
weight, and each model's part between the tenants that hold blocks in it, by tenant
weight. A tenant at its `max_ram_pct` cap evicts its own oldest position first, or its
save is refused.

On **NVMe** the unit is a whole segment file, because only deleting a file frees disk.
Under `fair` the daemon reclaims the segment with the most dead bytes, then the most
idle bytes (unused longer than `idle_nvme_s`), then the most bytes of tenants over
their share, then the oldest. Every block in that segment leaves NVMe, together with
its position's members in other namespaces, and with any descendant left without a
parent. A tenant at its `max_nvme_pct` cap reclaims the segment where its share of the
live bytes is largest; if others hold more live bytes there than it does, its write
stays in RAM only. A namespace that has not registered since the
daemon started counts as dead only after 30 minutes, so a model that loads slowly keeps
its blocks.

## Connection loss

A connection owns the saves it began and the loads it began, but any connection may
commit, abort or end them by key or handle (the client library uses one connection per
thread).

For each connection the daemon tracks the saves begun and not committed or aborted,
and the loads begun with at least one slot and not ended. When the connection drops,
it aborts those saves and ends those loads. A crashed engine therefore leaks neither
slots nor pins. Committed blocks stay.

## Failing open

The daemon is an optimisation. A client must treat a missing socket, a connection
error, a reply that does not arrive within its deadline, and `ok: false` as "no
cache": lookups return 0, saves are dropped, the request proceeds as if the daemon
did not exist. Keep the deadline short (tens of milliseconds for `lookup` and
`begin_save`). `begin_load` may read every key from NVMe, so give it a deadline that
grows with the number of keys (the reference client allows 2 s plus 50 ms per key, at
most 30 s). `register_ns` runs once per namespace when an engine starts; the
reference client gives it 30 s and, after a timeout or connection error (not after
`ok: false`), tries it once more, since registering is idempotent. Drop the
connection after any failure (a late reply cannot be matched
to a new request), and back off before reconnecting. The reference client is
`lmw_kv/client.py`.

## Example

```
-> {"op":"hello","id":1}
<- {"id":1,"ok":true,"protocol":1,"ram_path":"/dev/shm/lmw-kv.pool","ram_bytes":8589934592}
-> {"op":"begin_save","id":2,"ns":"<64 hex>","key":"<64 hex>","parent":"<64 hex>","scope":""}
<- {"id":2,"ok":true,"offsets":{"full":0,"gdn":8192}}
-> {"op":"commit_save","id":3,"ns":"<64 hex>","key":"<64 hex>"}
<- {"id":3,"ok":true}
```
