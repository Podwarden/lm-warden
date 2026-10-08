"use client";

// One tool's panel (plan §3, §5.6): what the tool gets, what it needs
// (evaluated against the picks), the mode and file tabs, the snippet, Verify,
// and the verification record with the docs it was built from. An
// unsupported tool shows its reason and nothing to copy or test.

import { cn } from "@/lib/utils";
import {
  SUPPORT,
  type ConnectClient,
  type ConnectFile,
  type ConnectMode,
  type ConnectRouter,
  type ConnectVerified,
  type RequirementCheck,
  type TemplateVars,
} from "@/lib/connect";
import { Requirements } from "./requirements";
import { SnippetBlock } from "./snippet-block";
import { Tabs, panelId, tabId } from "./tabs";
import { TOOLS_ID } from "./tool-list";
import { VerifyRow } from "./verify-row";
import { LINK, META, SEC, SEC_BODY, pill } from "./styles";

const MODES_ID = "connect-mode";
const FILES_ID = "connect-file";

function verifiedLine(v: ConnectVerified | null): string {
  if (!v) return "Not yet run against a live warden.";
  const by = v.client_version ? ` with ${v.client_version}` : "";
  if (v.status === "run") return `Verified ${v.date}${by}${v.note ? `. ${v.note}` : ""}`;
  if (v.status === "documented") return `Documented ${v.date}, not run`;
  return `Tried ${v.date}${by}: it failed${v.note ? `. ${v.note}` : "."}`;
}

function ExternalIcon() {
  return (
    <svg viewBox="0 0 12 12" className="ml-0.5 inline h-3 w-3 align-baseline" fill="none" stroke="currentColor" strokeWidth="1.4" aria-hidden="true">
      <path d="M4.5 2.5h5v5M9.5 2.5 3 9" />
    </svg>
  );
}

interface Props {
  client: ConnectClient;
  mode: ConnectMode | null;
  onModeChange: (id: string) => void;
  files: ConnectFile[];
  renderFile: (f: ConnectFile) => string;
  checks: RequirementCheck[];
  router: ConnectRouter | null;
  vars: TemplateVars;
  createdKey: string | null;
  pasted: string;
  onPaste: (v: string) => void;
  resetKey: string;
  /** "Using this tab's address…" when public_url is unset. */
  originNote: string | null;
}

export function ClientPanel({
  client,
  mode,
  onModeChange,
  files,
  renderFile,
  checks,
  router,
  vars,
  createdKey,
  pasted,
  onPaste,
  resetKey,
  originNote,
}: Props) {
  const s = SUPPORT[client.support];
  const unsupported = client.support === "unsupported";

  return (
    <section
      role="tabpanel"
      id={panelId(TOOLS_ID)}
      aria-labelledby={tabId(TOOLS_ID, client.id)}
      data-testid="connect-panel"
      className={cn(SEC, "min-w-0")}
    >
      <div className="flex flex-wrap items-center gap-x-2.5 gap-y-1 px-4 pb-2 pt-4">
        <h2 className="m-0 text-[18px] font-semibold leading-tight">{client.name}</h2>
        <span className={pill(s.tone)}>{s.label}</span>
      </div>
      <div className={SEC_BODY}>
        <p className="m-0 max-w-[78ch] text-[14px]">{client.summary}</p>
        <p className={cn("m-0 mt-1.5 max-w-[78ch] text-[13px] text-chat-muted", unsupported && "text-chat-fg")}>
          {client.capability}
        </p>

        {!unsupported && checks.length > 0 && (
          <div className="mt-4">
            <Requirements checks={checks} />
          </div>
        )}

        {!unsupported && mode && (
          <div className="mt-5">
            {client.modes.length > 1 && (
              <Tabs
                items={client.modes.map((m) => ({ id: m.id, label: m.title }))}
                selected={mode.id}
                onSelect={onModeChange}
                label="Setup mode"
                idBase={MODES_ID}
                className="mb-3"
              />
            )}
            <div
              {...(client.modes.length > 1
                ? { role: "tabpanel", id: panelId(MODES_ID), "aria-labelledby": tabId(MODES_ID, mode.id) }
                : {})}
              className="min-w-0"
            >
              {client.modes.length === 1 && <h3 className="m-0 mb-1 text-[15px] font-semibold">{mode.title}</h3>}
              {mode.description && <p className={cn(META, "m-0 mb-3 max-w-[78ch]")}>{mode.description}</p>}
              <SnippetBlock key={`${client.id}/${mode.id}`} files={files} render={renderFile} idBase={FILES_ID} />
              {originNote && <p className={cn(META, "m-0 mt-2")}>{originNote}</p>}
              {mode.verify && (
                <VerifyRow
                  verify={mode.verify}
                  mode={mode}
                  protocol={client.protocol}
                  router={router}
                  vars={vars}
                  createdKey={createdKey}
                  pasted={pasted}
                  onPaste={onPaste}
                  resetKey={resetKey}
                />
              )}
            </div>
          </div>
        )}

        <div className={cn(META, "mt-5 flex flex-wrap gap-x-5 gap-y-1 border-t border-vw-rule-soft/45 pt-3")}>
          <span>{verifiedLine(client.verified)}</span>
          <a href={client.docs.url} target="_blank" rel="noopener noreferrer" className={LINK}>
            {client.name.replace(/\s*\(.*\)$/, "")} docs
            <ExternalIcon />
            <span className="sr-only"> (opens in a new tab)</span>
          </a>
          <span>accessed {client.docs.accessed}</span>
        </div>
      </div>
    </section>
  );
}
