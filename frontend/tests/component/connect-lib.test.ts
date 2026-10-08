import { describe, it, expect, vi, afterEach } from 'vitest';
import { existsSync, readFileSync } from 'node:fs';
import path from 'node:path';
import { envSnippet, settingsJsonSnippet, localOnlySnippet } from '@/lib/router';
import {
  CONTEXT_FALLBACK,
  MODEL_PLACEHOLDER,
  buildVerifyRequest,
  deriveConnectState,
  explainVerify,
  filesFor,
  parseConnectParams,
  renderTemplate,
  requirementChecks,
  verifyModelFor,
} from '@/lib/connect';
import {
  CLAUDE_CODE,
  CODEX,
  CONTINUE,
  MODEL_BIG,
  MODEL_SMALL,
  MODEL_UNKNOWN_CTX,
  MODELS,
  OPENCODE,
  ROUTER_OFF,
  ROUTER_ON,
  TOKEN_PLAIN,
  TOKEN_RELAY,
  TOKENS_DEAD,
  TOKENS_EMPTY,
  TOKENS_LIVE,
  connectOut,
} from './connect-test-utils';

const VARS = {
  origin: 'https://w.example',
  key: 'vw_secret123',
  model: 'qwen',
  header: 'X-LMWarden-Key',
  context: 65536,
};

describe('renderTemplate', () => {
  it('substitutes all five placeholders', () => {
    const t = '{{origin}}/v1 {{key}} {{model}} {{header}} {{context}}';
    expect(renderTemplate(t, VARS)).toBe('https://w.example/v1 vw_secret123 qwen X-LMWarden-Key 65536');
  });

  it('substitutes every occurrence, and verify_model', () => {
    expect(renderTemplate('{{model}}/{{model}} {{verify_model}}', { ...VARS, verify_model: 'claude-haiku-4-5' })).toBe(
      'qwen/qwen claude-haiku-4-5',
    );
  });

  it('verify_model defaults to the model', () => {
    expect(renderTemplate('{{verify_model}}', VARS)).toBe('qwen');
  });

  it('leaves unknown text and unknown placeholders alone', () => {
    expect(renderTemplate('a {{nope}} {b} {{ origin }} $x', VARS)).toBe('a {{nope}} {b} {{ origin }} $x');
  });

  it('never throws on missing vars: uses the placeholder defaults', () => {
    const out = renderTemplate('{{key}} {{model}} {{header}} {{context}}', { origin: 'https://w.example' });
    expect(out).toBe(`vw_YOUR_KEY ${MODEL_PLACEHOLDER} X-LMWarden-Key ${CONTEXT_FALLBACK}`);
    expect(renderTemplate('{{key}}', { origin: 'o', key: null, model: null, context: null })).toBe('vw_YOUR_KEY');
    expect(renderTemplate('{{context}}', { origin: 'o', context: null })).toBe('32768');
    expect(renderTemplate('{{origin}}', {})).not.toContain('{{');
  });
});

// The TS half of the no-drift guarantee (plan §1.1): the router panel's helpers
// must produce byte for byte what the server's catalogue renders. The fixture
// is written by the backend task; it is shared with pytest.
const GOLDEN = path.resolve(__dirname, '../../../tests/fixtures/connect/claude-code.golden.json');

describe.skipIf(!existsSync(GOLDEN))('golden parity with the server catalogue', () => {
  const g = existsSync(GOLDEN)
    ? (JSON.parse(readFileSync(GOLDEN, 'utf8')) as Record<string, string>)
    : ({} as Record<string, string>);
  it('envSnippet', () => {
    expect(envSnippet(g.origin, g.header, g.key)).toBe(g.env);
  });
  it('settingsJsonSnippet', () => {
    expect(settingsJsonSnippet(g.origin, g.header, g.key)).toBe(g.settings_json);
  });
  it('localOnlySnippet', () => {
    expect(localOnlySnippet(g.origin, g.model, g.key)).toBe(g.local_only);
  });
});

describe('filesFor', () => {
  it('picks the requires_tools variant matching the model, keeps neutral files', () => {
    const mode = CONTINUE.modes[0];
    expect(filesFor(mode, MODEL_BIG).map((f) => f.id)).toEqual(['config_tools']);
    expect(filesFor(mode, MODEL_SMALL).map((f) => f.id)).toEqual(['config']);
    expect(filesFor(OPENCODE.modes[0], MODEL_SMALL).map((f) => f.id)).toEqual(['config']);
  });
  it('with no model, the no-tools variant (never claims a capability)', () => {
    expect(filesFor(CONTINUE.modes[0], null).map((f) => f.id)).toEqual(['config']);
  });
});

describe('requirementChecks', () => {
  const local = CLAUDE_CODE.modes[1];
  const routerMode = CLAUDE_CODE.modes[0];
  const byId = (rows: ReturnType<typeof requirementChecks>) => Object.fromEntries(rows.map((r) => [r.id, r]));

  it('min context: ok when the model is big enough, with the numbers', () => {
    const r = byId(requirementChecks(CLAUDE_CODE, local, { model: MODEL_BIG, key: null, router: ROUTER_ON, models: MODELS }));
    expect(r.min_context.state).toBe('ok');
    expect(r.min_context.detail).toBe('qwen3-4b-dp4: 65,536 tokens (needs ≥ 49,152)');
    // recommended 131,072 is soft: a 64k model is a warning, not a failure.
    expect(r.recommended_context.state).toBe('warn');
  });

  it('min context: warn when too small, unknown when the window is unknown', () => {
    const small = byId(requirementChecks(CLAUDE_CODE, local, { model: MODEL_SMALL, key: null, router: ROUTER_ON, models: MODELS }));
    expect(small.min_context.state).toBe('warn');
    expect(small.min_context.detail).toBe('tiny-8k: 8,192 tokens (needs ≥ 49,152)');
    const unk = byId(requirementChecks(CLAUDE_CODE, local, { model: MODEL_UNKNOWN_CTX, key: null, router: ROUTER_ON, models: MODELS }));
    expect(unk.min_context.state).toBe('unknown');
    expect(unk.min_context.detail).toBe('mystery: context unknown');
    const none = byId(requirementChecks(CLAUDE_CODE, local, { model: null, key: null, router: ROUTER_ON, models: MODELS }));
    expect(none.min_context.state).toBe('unknown');
  });

  it('router mode: min context is evaluated per rule target and names the smallest', () => {
    const r = byId(requirementChecks(CLAUDE_CODE, routerMode, { model: MODEL_BIG, key: null, router: ROUTER_ON, models: MODELS }));
    expect(r.min_context.state).toBe('warn');
    expect(r.min_context.detail).toBe('tiny-8k: 8,192 tokens (needs ≥ 49,152)');
  });

  it('tools: fail when the model has no tool calling, ok when it has', () => {
    expect(byId(requirementChecks(OPENCODE, OPENCODE.modes[0], { model: MODEL_SMALL, key: null, router: ROUTER_ON })).tools.state).toBe('fail');
    expect(byId(requirementChecks(OPENCODE, OPENCODE.modes[0], { model: MODEL_BIG, key: null, router: ROUTER_ON })).tools.state).toBe('ok');
    expect(byId(requirementChecks(OPENCODE, OPENCODE.modes[0], { model: null, key: null, router: ROUTER_ON })).tools.state).toBe('unknown');
  });

  it('router_on: fail when routing is disabled', () => {
    expect(byId(requirementChecks(CLAUDE_CODE, routerMode, { model: null, key: null, router: ROUTER_OFF })).router_on.state).toBe('fail');
    expect(byId(requirementChecks(CLAUDE_CODE, routerMode, { model: null, key: null, router: ROUTER_ON })).router_on.state).toBe('ok');
  });

  it('relay_key: fail for a non-relay key, ok for a relay key, unknown with none picked', () => {
    const rk = (key: typeof TOKEN_PLAIN | null) =>
      byId(requirementChecks(CLAUDE_CODE, routerMode, { model: null, key, router: ROUTER_ON })).relay_key.state;
    expect(rk(TOKEN_PLAIN)).toBe('fail');
    expect(rk(TOKEN_RELAY)).toBe('ok');
    expect(rk(null)).toBe('unknown');
  });

  it('router-only requirements are dropped in local mode; text-only rows are unknown', () => {
    const r = byId(requirementChecks(CLAUDE_CODE, local, { model: MODEL_BIG, key: TOKEN_PLAIN, router: ROUTER_OFF }));
    expect(r.router_on).toBeUndefined();
    expect(r.relay_key).toBeUndefined();
    expect(r.client_version.state).toBe('unknown');
    expect(r.client_version.text).toMatch(/2\.1\.227/);
  });

  it('honours a requirement\'s modes list (null = every mode)', () => {
    const scoped = {
      ...CLAUDE_CODE,
      requirements: [{ ...CLAUDE_CODE.requirements[0], modes: ['router'] }],
    };
    expect(requirementChecks(scoped, local, { model: MODEL_BIG, key: null, router: ROUTER_ON })).toHaveLength(0);
    expect(requirementChecks(scoped, routerMode, { model: MODEL_BIG, key: null, router: ROUTER_ON })).toHaveLength(1);
  });

  it('tools: a model whose tool support is unknown is unknown, not a failure', () => {
    const r = byId(requirementChecks(OPENCODE, OPENCODE.modes[0], {
      model: { ...MODEL_BIG, supports_tools: null }, key: null, router: ROUTER_ON,
    }));
    expect(r.tools.state).toBe('unknown');
    expect(r.tools.detail).toBe('qwen3-4b-dp4: tool calling unknown');
  });
});

describe('verifyModelFor', () => {
  it("router mode: the first enabled rule's example_model", () => {
    expect(verifyModelFor(CLAUDE_CODE, CLAUDE_CODE.modes[0], ROUTER_ON, 'qwen3-4b-dp4')).toBe('claude-haiku-4-5');
  });
  it('router mode skips rules without an example', () => {
    const router = { ...ROUTER_ON, rules: [{ ...ROUTER_ON.rules[0], example_model: null }, ROUTER_ON.rules[1]] };
    expect(verifyModelFor(CLAUDE_CODE, CLAUDE_CODE.modes[0], router, 'qwen')).toBe('claude-sonnet-4-5');
  });
  it('every other mode: the selected served name, or the placeholder', () => {
    expect(verifyModelFor(CLAUDE_CODE, CLAUDE_CODE.modes[1], ROUTER_ON, 'qwen')).toBe('qwen');
    expect(verifyModelFor(OPENCODE, OPENCODE.modes[0], ROUTER_ON, null)).toBe(MODEL_PLACEHOLDER);
  });
});

describe('buildVerifyRequest', () => {
  afterEach(() => vi.useRealTimers());

  it('anthropic router verify: same-origin URL, header rendered, no Authorization, credentials omitted', () => {
    const v = CLAUDE_CODE.modes[0].verify!;
    const { url, init, done } = buildVerifyRequest(v, { ...VARS, verify_model: 'claude-haiku-4-5' }, 'http://tab.local:3000');
    done();
    expect(url).toBe('http://tab.local:3000/v1/messages');
    expect(url).not.toContain('vw_secret123');
    expect(init.method).toBe('POST');
    expect(init.credentials).toBe('omit');
    const h = init.headers as Record<string, string>;
    expect(h['X-LMWarden-Key']).toBe('vw_secret123');
    expect(h['anthropic-version']).toBe('2023-06-01');
    expect(Object.keys(h).map((k) => k.toLowerCase())).not.toContain('authorization');
    expect(JSON.parse(init.body as string)).toEqual({
      model: 'claude-haiku-4-5',
      max_tokens: 16,
      messages: [{ role: 'user', content: 'Reply with the single word OK.' }],
    });
  });

  it('openai verify: Bearer only because the template says so', () => {
    const { init, done } = buildVerifyRequest(OPENCODE.modes[0].verify!, VARS, 'http://tab.local');
    done();
    expect((init.headers as Record<string, string>).Authorization).toBe('Bearer vw_secret123');
    expect(JSON.parse(init.body as string).model).toBe('qwen');
  });

  it('defaults to this tab’s origin, never the public URL', () => {
    const { url, done } = buildVerifyRequest(OPENCODE.modes[0].verify!, VARS);
    done();
    expect(url).toBe(`${window.location.origin}/v1/chat/completions`);
  });

  it('aborts after 15 s', () => {
    vi.useFakeTimers();
    const { init } = buildVerifyRequest(OPENCODE.modes[0].verify!, VARS, 'http://tab.local');
    expect(init.signal!.aborted).toBe(false);
    vi.advanceTimersByTime(15_000);
    expect(init.signal!.aborted).toBe(true);
  });
});

describe('explainVerify (plan §5.4)', () => {
  const ctx = { latencyMs: 812, model: 'qwen3-4b-dp4', origin: 'https://w.example' };

  it('200 anthropic: status, latency, model, answer', () => {
    const body = { model: 'qwen3-4b-dp4', content: [{ type: 'text', text: 'OK' }] };
    const r = explainVerify(200, body, 'anthropic', ctx);
    expect(r.tone).toBe('ok');
    expect(r.text).toBe("200 in 812 ms · answered by qwen3-4b-dp4 · 'OK'");
  });

  it('200 openai: answer from choices, trimmed to 80 characters', () => {
    const long = 'x'.repeat(200);
    const r = explainVerify(200, { model: 'qwen', choices: [{ message: { content: long } }] }, 'openai', ctx);
    expect(r.text).toBe(`200 in 812 ms · answered by qwen · '${'x'.repeat(80)}…'`);
  });

  it('200 responses: answer from the message output_text parts', () => {
    const body = {
      model: 'qwen',
      output: [
        { type: 'reasoning', summary: [{ type: 'summary_text', text: 'hmm' }] },
        { type: 'message', content: [{ type: 'output_text', text: 'O' }, { type: 'output_text', text: 'K' }] },
      ],
    };
    const r = explainVerify(200, body, 'openai', { ...ctx, expect: 'openai_responses' });
    expect(r).toEqual({ tone: 'ok', text: "200 in 812 ms · answered by qwen · 'OK'" });
  });

  it('200 responses without message text is not a pass', () => {
    const r = explainVerify(200, { model: 'qwen', output: [] }, 'openai', { ...ctx, expect: 'openai_responses' });
    expect(r.tone).toBe('danger');
    expect(r.text).toMatch(/no message text/);
  });

  it('openai error envelope from /v1/responses surfaces its message', () => {
    const body = { error: { message: 'previous_response_id is not supported', type: 'invalid_request_error' } };
    expect(explainVerify(400, body, 'openai', ctx).text).toBe('previous_response_id is not supported');
  });

  it('401', () => {
    expect(explainVerify(401, {}, 'openai', ctx)).toEqual({
      tone: 'danger',
      text: 'Key not accepted: unknown, expired or revoked.',
    });
  });

  it('403 relay_not_allowed', () => {
    const body = { type: 'error', error: { type: 'permission_error', message: 'router: this API key may not relay to Anthropic (relay_not_allowed)' } };
    expect(explainVerify(403, body, 'anthropic', ctx).text).toBe(
      'This key may not relay to Anthropic — tick May relay on the key.',
    );
  });

  it('403 token_not_allowed', () => {
    const body = { detail: 'router: token not allowed for this model (token_not_allowed)' };
    expect(explainVerify(403, body, 'anthropic', ctx).text).toBe("This key's model allow list excludes this model.");
  });

  it('404', () => {
    expect(explainVerify(404, {}, 'openai', ctx).text).toBe(
      'Model qwen3-4b-dp4 is not served here (not loaded, or not a served name).',
    );
  });

  it("413 / 400: the server's message", () => {
    expect(explainVerify(413, { error: { message: 'body too large' } }, 'openai', ctx).text).toBe('body too large');
    expect(explainVerify(400, { detail: [{ msg: 'bad' }] }, 'openai', ctx).text).toMatch(/400/);
    expect(explainVerify(400, { detail: 'max_tokens too big' }, 'openai', ctx).text).toBe('max_tokens too big');
  });

  it('502 / 529: the local model failed', () => {
    expect(explainVerify(502, { error: { message: 'engine down' } }, 'openai', ctx).text).toBe(
      'The local model failed: engine down',
    );
    expect(explainVerify(529, { type: 'error', error: { type: 'overloaded_error', message: 'busy' } }, 'anthropic', ctx).text).toBe(
      'The local model failed: busy',
    );
  });

  it('abort and network', () => {
    expect(explainVerify('abort', null, 'openai', ctx).text).toBe('No answer in 15 s.');
    expect(explainVerify('network', null, 'openai', ctx).text).toBe("Couldn't reach https://w.example from this tab.");
  });

  it('never echoes the key', () => {
    const body = { error: { message: 'bad key vw_secret123' } };
    expect(explainVerify(400, body, 'openai', { ...ctx, key: 'vw_secret123' }).text).not.toContain('vw_secret123');
  });
});

describe('deriveConnectState (plan §5.5)', () => {
  const out = connectOut();
  it('loading', () => {
    expect(deriveConnectState({ clients: undefined, tokens: undefined, errors: {} }).kind).toBe('loading');
  });
  it('clients error wins', () => {
    const s = deriveConnectState({ clients: undefined, tokens: TOKENS_LIVE, errors: { clients: true } });
    expect(s.kind).toBe('error');
    expect(s.strip).toMatchObject({ tone: 'danger', role: 'alert', action: 'retry' });
    expect(s.strip!.text).toBe("Couldn't load the client catalogue. Nothing was changed.");
  });
  it('no loaded model', () => {
    const s = deriveConnectState({ clients: { ...out, models: [MODELS[3]] }, tokens: TOKENS_LIVE, errors: {} });
    expect(s.kind).toBe('no_model');
    expect(s.strip).toMatchObject({ tone: 'amber', role: 'alert', action: 'models' });
  });
  it('no live key (none, or all dead)', () => {
    for (const tokens of [TOKENS_EMPTY, TOKENS_DEAD]) {
      const s = deriveConnectState({ clients: out, tokens, errors: {} });
      expect(s.kind).toBe('no_key');
      expect(s.strip).toMatchObject({ tone: 'neutral', role: 'status', action: 'create_key' });
    }
  });
  it('tokens error', () => {
    const s = deriveConnectState({ clients: out, tokens: undefined, errors: { tokens: true } });
    expect(s.kind).toBe('tokens_error');
    expect(s.strip!.text).toBe("Couldn't load your keys; snippets keep the placeholder.");
  });
  it('ok: no strip (tokens still loading is not a problem)', () => {
    expect(deriveConnectState({ clients: out, tokens: TOKENS_LIVE, errors: {} })).toEqual({ kind: 'ok', strip: null });
    expect(deriveConnectState({ clients: out, tokens: undefined, errors: {} }).kind).toBe('ok');
  });
});

describe('parseConnectParams', () => {
  const clients = [CLAUDE_CODE, OPENCODE, CONTINUE, CODEX];
  const p = (q: string) => parseConnectParams(new URLSearchParams(q), clients);

  it('reads tool and mode', () => {
    expect(p('tool=claude-code&mode=local')).toEqual({ tool: 'claude-code', mode: 'local', corrected: false });
    expect(p('tool=opencode')).toEqual({ tool: 'opencode', mode: 'default', corrected: false });
  });
  it('unknown tool falls back to the first and asks for the URL to be corrected', () => {
    expect(p('tool=nope')).toEqual({ tool: 'claude-code', mode: 'router', corrected: true });
  });
  it('no tool: first tool, nothing to correct', () => {
    expect(p('')).toEqual({ tool: 'claude-code', mode: 'router', corrected: false });
  });
  it('unknown mode falls back to the first mode; a client without modes has none', () => {
    expect(p('tool=claude-code&mode=zzz')).toEqual({ tool: 'claude-code', mode: 'router', corrected: true });
    expect(p('tool=codex-cli')).toEqual({ tool: 'codex-cli', mode: null, corrected: false });
  });
  it('an empty catalogue', () => {
    expect(parseConnectParams(new URLSearchParams('tool=x'), [])).toEqual({ tool: null, mode: null, corrected: false });
  });
});

describe('describeRoute', () => {
  it('names the local model from the log, and the fallback reason', async () => {
    const { describeRoute } = await import('@/lib/connect');
    expect(describeRoute({ route: 'local', reason: null, model_out: 'qwen' })).toBe('route: local on qwen');
    expect(describeRoute({ route: 'fallback', reason: 'status_400' })).toBe('fell back to Anthropic (status_400)');
    expect(describeRoute({ route: 'passthrough', reason: null })).toBe('passed through to Anthropic');
    expect(describeRoute(undefined)).toBeNull();
  });
});

describe('presentation helpers', () => {
  it('groups tools by section in first-appearance order and labels support honestly', async () => {
    const { groupTools, SUPPORT, keyComment, hardMinContext, verifyBlocker } = await import('@/lib/connect');
    expect(groupTools([CLAUDE_CODE, OPENCODE, CONTINUE, CODEX]).map((g) => g.label)).toEqual([
      'Anthropic protocol', 'OpenAI protocol', 'Editors', 'Not supported',
    ]);
    expect(SUPPORT.unsupported.label).toBe('Not supported');
    expect(keyComment('bash', 'laptop', 'vw_ab…')).toBe('# key: laptop (vw_ab…)');
    expect(keyComment('typescript', 'laptop', 'vw_ab…')).toBe('// key: laptop (vw_ab…)');
    expect(keyComment('json', 'laptop', 'vw_ab…')).toBeNull();
    expect(hardMinContext(CLAUDE_CODE, CLAUDE_CODE.modes[1])).toBe(49152);
    expect(hardMinContext(OPENCODE, OPENCODE.modes[0])).toBeNull();
    expect(verifyBlocker(CLAUDE_CODE.modes[0], ROUTER_OFF, true)).toMatch(/Routing is off/);
    expect(verifyBlocker(OPENCODE.modes[0], ROUTER_OFF, false)).toMatch(/Paste or create a key/);
    expect(verifyBlocker(OPENCODE.modes[0], ROUTER_OFF, true)).toBeNull();
  });
});
