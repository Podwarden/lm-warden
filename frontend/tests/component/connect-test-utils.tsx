// Fixtures for the Connect page (plan 2026-10-04-connect-clients §2, §9): a full
// `GET /api/connect/clients` response with one client per support level, the
// models it lists, and token pages for the key picker.

import type { TokenListPage } from '@/lib/router';
import type { ConnectClient, ConnectClientsOut, ConnectModel, ConnectRequirement, ConnectRouter } from '@/lib/connect';
import { token } from './router-test-utils';

export { token };

type TokenItem = TokenListPage['items'][number];

export const CONNECT_URL = '/api/connect/clients';
export const ORIGIN = 'https://lmwarden.example';

export const MODEL_BIG: ConnectModel = {
  id: 'm1', served_name: 'qwen3-4b-dp4', status: 'loaded', backend: 'vllm',
  context_window: 65536, supports_tools: true, supports_vision: false, supports_reasoning: true,
};
export const MODEL_SMALL: ConnectModel = {
  id: 'm2', served_name: 'tiny-8k', status: 'loaded', backend: 'llamacpp',
  context_window: 8192, supports_tools: false, supports_vision: false, supports_reasoning: false,
};
export const MODEL_UNKNOWN_CTX: ConnectModel = {
  id: 'm3', served_name: 'mystery', status: 'loaded', backend: 'vllm',
  context_window: null, supports_tools: true, supports_vision: false, supports_reasoning: false,
};
export const MODEL_IDLE: ConnectModel = {
  id: 'm4', served_name: 'llama-70b', status: 'registered', backend: 'vllm',
  context_window: 131072, supports_tools: true, supports_vision: false, supports_reasoning: false,
};
export const MODELS: ConnectModel[] = [MODEL_BIG, MODEL_UNKNOWN_CTX, MODEL_SMALL, MODEL_IDLE];

export const ROUTER_ON: ConnectRouter = {
  enabled: true,
  passthrough_unmatched: true,
  rules: [
    { pattern: 'claude-haiku*', target_served_name: 'qwen3-4b-dp4', target_status: 'loaded', fallback: true, example_model: 'claude-haiku-4-5' },
    { pattern: 'claude-sonnet-4-5', target_served_name: 'tiny-8k', target_status: 'loaded', fallback: false, example_model: 'claude-sonnet-4-5' },
  ],
};
export const ROUTER_OFF: ConnectRouter = { ...ROUTER_ON, enabled: false };

/** A requirement with the API's defaults (every flag present, all modes). */
export function req(over: Partial<ConnectRequirement> & Pick<ConnectRequirement, 'id' | 'text'>): ConnectRequirement {
  return { min_context_tokens: null, soft: false, needs_tools: false, router_on: false, relay_key: false, modes: null, ...over };
}

const ANTHROPIC_VERIFY_HEADERS = { 'anthropic-version': '2023-06-01', 'content-type': 'application/json' };
const PROMPT = [{ role: 'user', content: 'Reply with the single word OK.' }];

export const CLAUDE_CODE: ConnectClient = {
  id: 'claude-code',
  name: 'Claude Code',
  group: 'anthropic',
  protocol: 'anthropic',
  support: 'full',
  summary: 'Some Claude models run here; the rest go to Anthropic on your own login.',
  capability: 'The Claude models your rules name run here.',
  requirements: [
    req({ id: 'client_version', text: 'Claude Code 2.1.227 or newer (ANTHROPIC_CUSTOM_HEADERS).' }),
    req({ id: 'router_on', text: 'Routing must be on.', router_on: true, modes: ['router'] }),
    req({ id: 'relay_key', text: 'The key must be allowed to relay to Anthropic.', relay_key: true, modes: ['router'] }),
    req({ id: 'min_context', text: 'At least 49,152 tokens of context.', min_context_tokens: 49152 }),
    req({ id: 'recommended_context', text: '131,072 tokens recommended.', min_context_tokens: 131072, soft: true }),
  ],
  docs: { url: 'https://code.claude.com/docs/en/llm-gateway', accessed: '2026-10-04' },
  verified: null,
  modes: [
    {
      id: 'router',
      title: 'Router: keep your Claude login',
      description: 'Rules decide which Claude models run here.',
      needs_relay_key: true,
      files: [
        {
          id: 'shell', label: 'Shell', path: null, language: 'bash', requires_tools: null,
          template: 'export ANTHROPIC_BASE_URL={{origin}}\nexport ANTHROPIC_CUSTOM_HEADERS="{{header}}: {{key}}"\nclaude',
          rendered: `export ANTHROPIC_BASE_URL=${ORIGIN}\nexport ANTHROPIC_CUSTOM_HEADERS="X-LMWarden-Key: vw_YOUR_KEY"\nclaude`,
        },
      ],
      verify: {
        method: 'POST',
        path: '/v1/messages',
        headers: { '{{header}}': '{{key}}', ...ANTHROPIC_VERIFY_HEADERS },
        body: { model: '{{verify_model}}', max_tokens: 16, messages: PROMPT },
        expect: 'anthropic_message',
        covers: 'A rule-matched Claude model served locally with the key in {{header}}.',
        not_covered: 'Pass-through to Anthropic needs your Claude login.',
      },
    },
    {
      id: 'local',
      title: 'Local only (no Anthropic account)',
      description: 'Everything runs on one model here.',
      needs_relay_key: false,
      files: [
        {
          id: 'shell', label: 'Shell', path: null, language: 'bash', requires_tools: null,
          template: 'export ANTHROPIC_BASE_URL={{origin}}\nexport ANTHROPIC_AUTH_TOKEN={{key}}\nexport ANTHROPIC_MODEL={{model}}\nexport CLAUDE_CODE_MAX_CONTEXT_TOKENS={{context}}',
          rendered: '',
        },
      ],
      verify: {
        method: 'POST',
        path: '/v1/messages',
        headers: { 'x-api-key': '{{key}}', ...ANTHROPIC_VERIFY_HEADERS },
        body: { model: '{{verify_model}}', max_tokens: 16, messages: PROMPT },
        expect: 'anthropic_message',
        covers: 'The local model answers the Anthropic protocol.',
        not_covered: null,
      },
    },
  ],
};

const OPENAI_VERIFY = {
  method: 'POST' as const,
  path: '/v1/chat/completions',
  headers: { Authorization: 'Bearer {{key}}', 'content-type': 'application/json' },
  body: { model: '{{verify_model}}', max_tokens: 16, stream: false, messages: PROMPT },
  expect: 'openai_chat' as const,
  covers: 'The local model answers Chat Completions with this key.',
  not_covered: null,
};

export const OPENCODE: ConnectClient = {
  id: 'opencode',
  name: 'OpenCode',
  group: 'openai',
  protocol: 'openai',
  support: 'local_only',
  summary: 'LM Warden as an OpenAI-compatible provider for local models.',
  capability: 'Local models only.',
  requirements: [
    req({ id: 'tools', text: 'Needs a model with tool calling.', needs_tools: true }),
    req({ id: 'min_context', text: '32,768 tokens or more.', min_context_tokens: 32768, soft: true }),
  ],
  docs: { url: 'https://opencode.ai/docs/providers/', accessed: '2026-10-04' },
  verified: { status: 'run', date: '2026-10-04', client_version: 'opencode 1.2.3', note: null },
  modes: [
    {
      id: 'default',
      title: 'Local models',
      description: '',
      needs_relay_key: false,
      files: [
        {
          id: 'config', label: 'opencode.json', path: 'opencode.json', language: 'json', requires_tools: null,
          template: '{"provider":{"lmwarden":{"options":{"baseURL":"{{origin}}/v1","apiKey":"{{key}}"},"models":{"{{model}}":{"limit":{"context":{{context}}}}}}}}',
          rendered: '',
        },
      ],
      verify: OPENAI_VERIFY,
    },
  ],
};

export const CONTINUE: ConnectClient = {
  id: 'continue',
  name: 'Continue',
  group: 'editors',
  protocol: 'openai',
  support: 'documented',
  summary: 'Continue in VS Code or JetBrains, local models.',
  capability: 'Chat, edit and apply on local models.',
  requirements: [],
  docs: { url: 'https://docs.continue.dev/reference', accessed: '2026-10-04' },
  verified: { status: 'documented', date: '2026-10-04', client_version: null, note: null },
  modes: [
    {
      id: 'default',
      title: 'Local models',
      description: '',
      needs_relay_key: false,
      files: [
        { id: 'config_tools', label: 'config.yaml', path: '~/.continue/config.yaml', language: 'yaml', requires_tools: true,
          template: 'model: {{model}}\ncapabilities: [tool_use]', rendered: '' },
        { id: 'config', label: 'config.yaml', path: '~/.continue/config.yaml', language: 'yaml', requires_tools: false,
          template: 'model: {{model}}', rendered: '' },
      ],
      verify: OPENAI_VERIFY,
    },
  ],
};

export const CODEX: ConnectClient = {
  id: 'codex-cli',
  name: 'Codex CLI',
  group: 'openai',
  protocol: 'openai',
  support: 'unsupported',
  summary: 'Codex speaks only the Responses API, which this warden does not serve.',
  capability: 'Codex CLI cannot use LM Warden today, in any mode.',
  requirements: [],
  docs: { url: 'https://learn.chatgpt.com/docs/config-file/config-reference', accessed: '2026-10-04' },
  verified: null,
  modes: [],
};

export function connectOut(over: Partial<ConnectClientsOut> = {}): ConnectClientsOut {
  return {
    origin: ORIGIN,
    origin_source: 'public_url',
    header_name: 'X-LMWarden-Key',
    key_placeholder: 'vw_YOUR_KEY',
    router: ROUTER_ON,
    models: MODELS,
    selected_model: 'qwen3-4b-dp4',
    clients: [CLAUDE_CODE, OPENCODE, CONTINUE, CODEX],
    ...over,
  };
}

export function tokenPage(items: TokenItem[], total = items.length): TokenListPage {
  return { items, total, limit: 500, offset: 0, near_expiry: 0 };
}

export const TOKEN_PLAIN = token();
export const TOKEN_RELAY = token({ id: 't5', name: 'claude-code', prefix: 'vw_rl12', anthropic_relay: true });
export const TOKENS_LIVE = tokenPage([TOKEN_PLAIN, TOKEN_RELAY]);
/** Every key exists but none authenticates. */
export const TOKENS_DEAD = tokenPage([
  token({ id: 't2', name: 'old', is_revoked: true }),
  token({ id: 't3', name: 'stale', is_expired: true }),
  token({ id: 't4', name: 'held', is_paused: true }),
]);
export const TOKENS_EMPTY = tokenPage([]);
