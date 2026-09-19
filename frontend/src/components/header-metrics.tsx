'use client';
// Header metrics widget — compact live VRAM% + GPU% + active-model badge
// mounted inside <NavBar /> (right cluster, between ThemeSwitcher and
// the menu button). One <EventSource> per browser tab, ref-counted by
// frontend/src/lib/header-metrics-stream.ts.
//
// Visual language:
//   - server-rack instrument cluster — three readouts as a single
//     horizontal pill, divided by hairline slate dividers, mono digits.
//   - chrome stays subdued (slate-900/40 + slate-700/60 border) until
//     a probe error or terminal-error promotes the cluster to warning
//     (amber-400) / fault (red-400) state.
//   - the model slot is the cluster's identity slot — ONE CHIP PER LOADED
//     MODEL, each with its own dot, so the operator can tell at a glance
//     whether the rack is running anything and whether all of it is healthy.
//     Two chips render inline; beyond that the remainder folds into a "+N"
//     counter so the pill's width stays bounded at any fleet size, and the
//     title + aria-label enumerate every model so nothing is actually lost.
//   - hidden on /login and /setup matches NavBar's own gate.
//
// Restraint notes: this widget is glanceable and SHOULD NOT compete
// with the brand block or the Stats page charts. We deliberately don't
// animate the digits — flicker on every 2s tick is fatiguing.
import { Cpu, MemoryStick } from 'lucide-react';
import { useHeaderMetrics } from '@/lib/header-metrics-stream';
import {
  activeModelsOf,
  worstModelStatus,
  HEADER_MODELS_INLINE,
  type HeaderActiveModel,
  type HeaderModelStatus,
} from '@/lib/header-models';

// Format a percentage 0–100 (or null) into a fixed-width readout. We use
// figure-tab-numerals via Tailwind's `tabular-nums` so the digit grid
// doesn't jitter as values move between 1 and 100.
function pct(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) return '--';
  const clamped = Math.max(0, Math.min(100, Math.round(value)));
  return String(clamped).padStart(2, ' ');
}

// Compact MiB → GiB rendering for the VRAM tooltip — keeps the badge
// itself percentage-only.
function gib(mib: number | null): string {
  if (mib === null) return '--';
  if (!mib) return '0';
  return (mib / 1024).toFixed(1);
}

export function HeaderMetrics() {
  const { status, frame, errorCode } = useHeaderMetrics();

  // Terminal states render a quiet fault chip rather than the cluster.
  // We don't want the widget to nag — if /api/header/metrics/stream is
  // 401 (session expired) or 404 (endpoint missing on an old build),
  // collapse to a hyphenated placeholder so the chrome stays calm.
  const terminal = status === 'terminal-error';
  const reconnecting = status === 'reconnecting';
  const probeError = frame?.probe_error ?? null;

  // EVERY loaded model, from one shape regardless of the API's age.
  const models = activeModelsOf(frame);

  // The cluster's own state summarises N engines into one accent colour, so
  // it takes the WORST status rather than the first model's: three healthy
  // engines must not paint over the fourth that crashed. Each chip below
  // still carries its own dot, so the summary never hides which one.
  const modelStatus = worstModelStatus(models);
  const failed = !terminal && modelStatus === 'failed';
  const loading = !terminal && modelStatus === 'loading';
  const loaded = !terminal && modelStatus === 'loaded';

  // Accent: red (terminal / failed), amber (reconnecting / probe-error),
  // sky (loading), emerald (loaded), slate (idle). We deliberately
  // desaturate the idle state so the widget doesn't demand attention when
  // nothing is happening.
  //
  // Order matters: a *failed* engine outranks the amber "instruments are
  // unreliable" hint. A degraded nvidia-smi probe must never mask a dead
  // engine — that inversion is how a crash stays invisible.
  const accent =
    terminal || failed ? 'text-red-400' :
    reconnecting || probeError ? 'text-amber-400' :
    loading ? 'text-sky-400' :
    loaded ? 'text-emerald-400' :
    'text-slate-400';

  // The cluster-wide dot, shown only when the model slot has nothing of its
  // own to say (offline / idle). With models present each chip draws its own.
  const dot =
    terminal || failed ? 'bg-red-400/80' :
    reconnecting || probeError ? 'bg-amber-400/80' :
    loading ? 'bg-sky-400 animate-pulse' :
    loaded ? 'bg-emerald-400 shadow-[0_0_6px_rgba(52,211,153,0.6)]' :
    'bg-slate-500/70';

  // A probe error means nothing was measured: "--", never a number. The
  // backend sends null then (#255); an older API sent 0, which must not
  // reach the gauges either.
  const vramPct = terminal || probeError ? null : frame?.vram_pct ?? null;
  const gpuPct = terminal || probeError ? null : frame?.gpu_util_pct ?? null;

  // What the slot shows when there is no per-model list to show: 'offline' =
  // the stream is gone (we know nothing); 'idle' = the stream is fine and the
  // box is serving nothing. Two different situations that must not share a
  // word. A model's OWN 'loading'/'failed' state now rides on its chip's dot
  // instead of replacing the fleet's names with a bare status word — with
  // several models, collapsing all of them to "error" would say less than the
  // shortest useful thing.
  const emptyLabel = terminal ? 'offline' : 'idle';

  // Visible chips + the folded remainder. Bounded width at any N.
  const inline = models.slice(0, HEADER_MODELS_INLINE);
  const overflow = models.length - inline.length;

  // Build a multi-line tooltip that surfaces the data the badge omits:
  // per-GPU breakdown, every model by name, probe error, status hint.
  const tooltipLines: string[] = [];
  // Lost GPU telemetry leads the tooltip (#255): with the probe down the
  // gauges read "--", and the reason is the first thing to see. A model can
  // still be serving here -- it opened the GPUs before they became
  // unreadable -- but the next load will fail.
  if (probeError) tooltipLines.push(`GPU telemetry unavailable: ${probeError}`);
  if (frame) {
    if (!probeError) {
      tooltipLines.push(
        `VRAM ${gib(frame.vram_used_mib)} / ${gib(frame.vram_total_mib)} GiB (all cards)`,
      );
      // The GPU readout is a max, so with several cards it names one of them.
      // Saying which, and listing the rest, is what stops "GPU 90%" from
      // reading as a statement about the box.
      tooltipLines.push(`GPU ${pct(frame.gpu_util_pct).trim()}% on the busiest card`);
      for (const g of frame.gpus) {
        const name = g.name ?? `GPU ${g.index}`;
        tooltipLines.push(
          `  ${name}: ${gib(g.memory_used_mib)}/${gib(g.memory_total_mib)} GiB · util ${g.utilization_pct}%`,
        );
      }
    }
    // EVERY model, including any the "+N" counter folded away. The chip is
    // width-bounded; the tooltip is not, so this is where nothing is lost.
    for (const m of models) {
      tooltipLines.push(`${STATUS_VERB[m.status]}: ${m.served_model_name}`);
    }
  }
  if (reconnecting) tooltipLines.push('Reconnecting…');
  if (terminal) {
    tooltipLines.push(
      errorCode === 401
        ? 'Session expired — refresh the page'
        : `Stream unavailable (HTTP ${errorCode ?? '?'})`,
    );
  }
  const tooltip = tooltipLines.join('\n') || 'header metrics';

  return (
    <div
      // role=status so screen readers announce the cluster but don't
      // promote it to a live region (we don't want every 2s tick
      // narrated). aria-live=off forces that.
      role="status"
      aria-live="off"
      // The accessible name enumerates EVERY model, including any the "+N"
      // counter folded away — a screen-reader user has no tooltip to hover.
      aria-label={
        'Header metrics — ' +
        (probeError
          ? 'GPU telemetry unavailable, '
          : `VRAM ${pct(vramPct).trim()} percent, ` +
            `GPU ${pct(gpuPct).trim()} percent on the busiest card, `) +
        (models.length === 0
          ? emptyLabel
          : models
              .map((m) => `${m.served_model_name} ${m.status}`)
              .join(', '))
      }
      title={tooltip}
      data-testid="header-metrics"
      data-status={status}
      data-model-status={modelStatus ?? 'idle'}
      className={[
        'hidden md:inline-flex items-center gap-2',
        'h-8 px-2.5 rounded-md',
        'border border-slate-700/60 bg-slate-900/40',
        'font-mono text-xs leading-none',
        'transition-colors duration-300',
        accent,
      ].join(' ')}
    >
      {/* VRAM readout */}
      <span className="inline-flex items-center gap-1.5 tabular-nums">
        <MemoryStick className="h-3.5 w-3.5 opacity-70" aria-hidden="true" />
        <span className="text-slate-400">VRAM</span>
        <span data-testid="header-metrics-vram-pct" className="text-slate-100">
          {pct(vramPct)}
        </span>
        <span className="text-slate-500">%</span>
      </span>

      <span className="h-3 w-px bg-slate-700/80" aria-hidden="true" />

      {/* GPU util readout */}
      <span className="inline-flex items-center gap-1.5 tabular-nums">
        <Cpu className="h-3.5 w-3.5 opacity-70" aria-hidden="true" />
        <span className="text-slate-400">GPU</span>
        <span data-testid="header-metrics-gpu-pct" className="text-slate-100">
          {pct(gpuPct)}
        </span>
        <span className="text-slate-500">%</span>
      </span>

      <span className="h-3 w-px bg-slate-700/80" aria-hidden="true" />

      {/* Model slot — identity. One chip per loaded model, each with its own
          dot, so a crashed engine beside a healthy one is visible as such
          rather than averaged into one word. Two inline, the rest folded into
          a counter; the title and aria-label above still name every one. */}
      <span
        data-testid="header-metrics-models"
        className="inline-flex items-center gap-2 max-w-[18rem]"
      >
        {models.length === 0 ? (
          <span className="inline-flex items-center gap-1.5">
            <span
              aria-hidden="true"
              className={['h-1.5 w-1.5 rounded-full transition-colors', dot].join(' ')}
            />
            <span
              data-testid="header-metrics-model"
              className="truncate text-slate-200"
            >
              {emptyLabel}
            </span>
          </span>
        ) : (
          <>
            {inline.map((m) => (
              <ModelChip key={m.id} model={m} muted={terminal} />
            ))}
            {overflow > 0 && (
              <span
                data-testid="header-metrics-model-overflow"
                className="shrink-0 rounded bg-slate-800/80 px-1 text-[10px] text-slate-300"
              >
                +{overflow}
              </span>
            )}
          </>
        )}
      </span>
    </div>
  );
}

// Per-model dot colours. Deliberately the SAME vocabulary as the cluster
// accent — emerald serving, sky starting, red dead — so a chip and the pill
// around it never mean different things by the same colour.
const MODEL_DOT: Record<HeaderModelStatus, string> = {
  loaded: 'bg-emerald-400 shadow-[0_0_6px_rgba(52,211,153,0.6)]',
  loading: 'bg-sky-400 animate-pulse',
  failed: 'bg-red-400/80',
};

const STATUS_VERB: Record<HeaderModelStatus, string> = {
  loaded: 'Loaded',
  loading: 'Loading',
  failed: 'Failed',
};

// Shown next to the NAME, not instead of it. The old single-model badge
// replaced the name with a bare "loading"/"error" and left the name reachable
// only by hovering — its own comment said so. With several models that trade
// gets worse, not better: two chips both reading "error" name neither engine.
// A short suffix keeps the word AND the name; 'loaded' needs no suffix,
// because a serving model's dot already says it and the common case should be
// the quietest.
const STATUS_SUFFIX: Partial<Record<HeaderModelStatus, string>> = {
  loading: 'loading',
  // 'error', not 'offline': the stream is fine and is telling us the engine
  // died. The two failures must not share a word.
  failed: 'error',
};

/** One model's dot + name. `muted` when the stream is gone and we know
 *  nothing current — the last frame's names stay readable but stop claiming
 *  to be live. */
function ModelChip({
  model,
  muted,
}: {
  model: HeaderActiveModel;
  muted: boolean;
}) {
  return (
    <span
      data-testid="header-metrics-model-chip"
      data-model-status={model.status}
      className="inline-flex min-w-0 items-center gap-1.5"
      title={`${STATUS_VERB[model.status]}: ${model.served_model_name}`}
    >
      <span
        aria-hidden="true"
        className={[
          'h-1.5 w-1.5 shrink-0 rounded-full transition-colors',
          muted ? 'bg-slate-500/70' : MODEL_DOT[model.status],
        ].join(' ')}
      />
      <span className="truncate text-slate-200">{model.served_model_name}</span>
      {STATUS_SUFFIX[model.status] && (
        <span className="shrink-0 text-[10px] text-slate-400">
          {STATUS_SUFFIX[model.status]}
        </span>
      )}
    </span>
  );
}
