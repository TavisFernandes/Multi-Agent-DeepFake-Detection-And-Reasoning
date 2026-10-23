import { useState, useRef, useEffect, useMemo } from 'react';
import { motion, AnimatePresence, MotionConfig } from 'framer-motion';
import { ArrowLeft, UploadCloud, Download, RefreshCcw } from 'lucide-react';
import { usePageTransition } from '@/context/TransitionContext';
import { readNdjson } from '@/lib/ndjson';

// ----------------------------------------------------------------------
// Types
// ----------------------------------------------------------------------
export type AgentResult = {
  agent: string;
  confidence: number;
  finding: string;
  status: 'pending' | 'streaming' | 'done';
  gradcam_b64?: string | null;
  backend_status?: string;
  is_fake?: boolean | null;
};

export type AppState = 'idle' | 'analysing' | 'done' | 'error';

type MediaKind = 'image' | 'video' | 'audio';

type Verdict = {
  verdict: string,
  confidence: number,
  contributing: string[],
  unavailable: string[],
  fusion_method?: string,
  predicted_category?: string,
  category_probabilities?: Record<string, number>,
  dynamic_weights?: Record<string, number>,
  decision_threshold?: number,
  decision_thresholds?: { tau_low?: number; tau_high?: number },
  fake_probability?: number,
  decision_confidence?: number,
  verdict_class?: string,
  is_abstention?: boolean,
  leading_category_is_hypothesis?: boolean,
};

// ----------------------------------------------------------------------
// Static data
// ----------------------------------------------------------------------
const INITIAL_AGENTS: AgentResult[] = [
  { agent: 'Spatial Analyst',   confidence: 0, finding: '', status: 'pending', gradcam_b64: null },
  { agent: 'Temporal Analyst',  confidence: 0, finding: '', status: 'pending' },
  { agent: 'Audio Analyst',     confidence: 0, finding: '', status: 'pending' },
  { agent: 'Context Analyst',   confidence: 0, finding: '', status: 'pending' },
];

// Same order as INITIAL_AGENTS.
const AGENT_INFO = [
  { name: 'Spatial', model: 'EfficientNet-B0', checks: 'Face boundaries, blending seams and pixel artifacts in a frame.' },
  { name: 'Temporal', model: 'R3D-18', checks: 'Flicker and landmark jitter across a 16-frame window.' },
  { name: 'Audio', model: 'Mel-spectrogram CNN', checks: 'Vocoder artifacts across 128 mel bands of the soundtrack.' },
  { name: 'Context', model: 'Explanation only', checks: 'Describes whether the visual and audio scores agree. It does not vote.' },
];

// Mirrors the backend, which routes files by extension (main.py _detect_media_type).
const EXTS: Record<MediaKind, string[]> = {
  image: ['jpg', 'jpeg', 'png', 'webp', 'bmp', 'tiff', 'tif'],
  video: ['mp4', 'mov', 'avi', 'mkv', 'webm', 'flv'],
  audio: ['wav', 'mp3', 'm4a', 'aac', 'ogg', 'flac'],
};
const ACCEPT = Object.values(EXTS).flat().map((e) => `.${e}`).join(',');
const MAX_BYTES = 50 * 1024 * 1024;
const BACKEND_UNREACHABLE = "Can't reach the analysis service. Check that FastAPI is running on port 8000.";

const SKIPPED_FOR: Record<MediaKind, string> = {
  image: 'Temporal and Audio are skipped for still images.',
  video: 'All four models run. Audio is skipped if the video has no soundtrack.',
  audio: 'Spatial and Temporal are skipped for audio-only files.',
};

// Fallback when the backend omits thresholds: corrected_fixed uses 0.50 ± 0.05.
const DEFAULT_LOW = 0.45;
const DEFAULT_HIGH = 0.55;

function mediaKind(file: File): MediaKind | null {
  const ext = file.name.toLowerCase().split('.').pop() ?? '';
  return (Object.keys(EXTS) as MediaKind[]).find((k) => EXTS[k].includes(ext)) ?? null;
}

const pct = (v: number) => `${(v * 100).toFixed(1)}%`;
const prettify = (s: string) => s.replace(/_/g, ' ').toLowerCase();

// ----------------------------------------------------------------------
// Page
// ----------------------------------------------------------------------
export default function Console() {
  const { navigateTo } = usePageTransition();
  const [appState, setAppState] = useState<AppState>('idle');
  const [errorMessage, setErrorMessage] = useState('');
  const [fileError, setFileError] = useState('');
  const [dragging, setDragging] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [agents, setAgents] = useState<AgentResult[]>(INITIAL_AGENTS);
  const [verdict, setVerdict] = useState<Verdict | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const abortRef = useRef<AbortController | null>(null);
  useEffect(() => () => abortRef.current?.abort(), []); // stop the stream if the user leaves the page
  const resultsRef = useRef<HTMLDivElement>(null);

  const kind = selectedFile ? mediaKind(selectedFile) : null;
  const previewUrl = useMemo(() => (selectedFile ? URL.createObjectURL(selectedFile) : null), [selectedFile]);
  useEffect(() => () => { if (previewUrl) URL.revokeObjectURL(previewUrl); }, [previewUrl]);

  const reported = agents.filter((a) => a.status !== 'pending').length;
  const analysing = appState === 'analysing';

  // ----------------------------------------------------------------------
  // Actions
  // ----------------------------------------------------------------------
  const reset = () => {
    setAppState('idle');
    setAgents(INITIAL_AGENTS);
    setVerdict(null);
    setErrorMessage('');
  };

  const pickFile = (file: File) => {
    if (analysing) return;
    if (!mediaKind(file)) {
      setFileError(
        `VERITAS can't read .${file.name.split('.').pop()} files. Use an image (JPG, PNG, WEBP, BMP, TIFF), video (MP4, MOV, AVI, MKV, WEBM, FLV) or audio file (WAV, MP3, M4A, AAC, OGG, FLAC).`
      );
      return;
    }
    if (file.size > MAX_BYTES) {
      setFileError(`This file is ${(file.size / 1048576).toFixed(1)} MB. The limit is 50 MB.`);
      return;
    }
    setFileError('');
    setSelectedFile(file);
    reset();
  };

  const dropProps = {
    onDragOver: (e: React.DragEvent) => { e.preventDefault(); if (!analysing) setDragging(true); },
    onDragLeave: () => setDragging(false),
    onDrop: (e: React.DragEvent) => {
      e.preventDefault();
      setDragging(false);
      const f = e.dataTransfer.files[0];
      if (f) pickFile(f);
    },
  };

  const downloadReport = () => {
    if (!selectedFile || !verdict) return;
    const report = {
      file: { name: selectedFile.name, size_bytes: selectedFile.size, media_type: kind },
      analysed_at: new Date().toISOString(),
      verdict,
      agents: agents.map(({ gradcam_b64, ...rest }) => rest),
    };
    const url = URL.createObjectURL(new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' }));
    const a = document.createElement('a');
    a.href = url;
    a.download = `veritas-report-${selectedFile.name}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const startAnalysis = async () => {
    if (!selectedFile || appState === 'analysing') return;

    setAppState('analysing');
    setAgents(INITIAL_AGENTS);
    setVerdict(null);
    setErrorMessage('');
    // On stacked (mobile) layout the results sit below the file card.
    if (window.matchMedia('(max-width: 1023px)').matches) {
      resultsRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }

    let hasError = false;
    let verdictReceived = false;

    try {
      const formData = new FormData();
      formData.append('file', selectedFile);

      abortRef.current = new AbortController();
      const response = await fetch('/api/analyse/stream', {
        method: 'POST',
        body: formData,
        signal: abortRef.current.signal,
      });

      if (!response.ok || !response.body) {
        const text = await response.text().catch(() => '');
        let detail: unknown;
        try { detail = JSON.parse(text).detail; } catch { /* not a FastAPI JSON error */ }
        throw new Error(
          typeof detail === 'string' ? detail
          // Vite's dev proxy answers an empty 500 when FastAPI isn't listening.
          : response.status >= 500 && !text ? BACKEND_UNREACHABLE
          : `Server returned ${response.status}: ${response.statusText || 'Unable to connect'}`
        );
      }

      // FastAPI streams NDJSON: one JSON object per line, each with a "type".
      await readNdjson(response.body, (data) => {
        // FastAPI calls agent updates "agent_result"; the handlers below use the older name.
        const eventName = data.type === 'agent_result' ? 'agent_update' : data.type;

        if (eventName === 'agent_update') {
          if (!data.finding) return;
          const agentName = ((data.agent || data.agent_name) as string) ?? '';
          let idx = 0;
          if (agentName.toLowerCase().includes('spatial'))  idx = 0;
          else if (agentName.toLowerCase().includes('temporal')) idx = 1;
          else if (agentName.toLowerCase().includes('audio'))    idx = 2;
          else if (agentName.toLowerCase().includes('context'))  idx = 3;

          setAgents(prev => {
            const copy = [...prev];
            copy[idx] = {
              agent: INITIAL_AGENTS[idx].agent,
              confidence: data.confidence as number,
              finding: data.finding as string,
              status: 'streaming',
              backend_status: data.status as string | undefined,
              is_fake: (data.is_fake !== undefined) ? (data.is_fake as boolean | null) : undefined,
              // Carry gradcam_b64 only for the spatial slot (idx 0)
              ...(idx === 0 && { gradcam_b64: (data.gradcam_b64 as string | null) ?? null }),
            };
            return copy;
          });

          // Transition to 'done' after the confidence bar has animated (600 ms)
          const capturedIdx = idx;
          setTimeout(() => {
            setAgents(prev => {
              const copy = [...prev];
              copy[capturedIdx] = { ...copy[capturedIdx], status: 'done' };
              return copy;
            });
          }, 650);

        } else if (eventName === 'final_verdict') {
          verdictReceived = true;
          const rawVerdict = (data.verdict as string).toUpperCase();
          let normalised = rawVerdict;
          if (rawVerdict === 'FAKE' || rawVerdict === 'DEEPFAKE') normalised = 'DEEPFAKE';
          else if (rawVerdict === 'REAL' || rawVerdict === 'AUTHENTIC') normalised = 'AUTHENTIC';
          else if (rawVerdict.includes('UNCERTAIN') || rawVerdict.includes('INSUFFICIENT')) {
            normalised = 'UNCERTAIN';
          } else if (rawVerdict.includes('UNVERIFIED')) {
            normalised = 'UNVERIFIED';
          }

          const verdictClass = (data.verdict_class as string | undefined)
            ?? (normalised === 'DEEPFAKE' || normalised === 'AUTHENTIC'
              || normalised === 'UNCERTAIN' || normalised === 'UNVERIFIED'
              ? normalised
              : 'UNCERTAIN');
          const isAbstention = Boolean(
            data.is_abstention
            ?? (verdictClass === 'UNCERTAIN' || verdictClass === 'UNVERIFIED')
          );

          setVerdict({
            verdict: normalised,
            confidence: (data.overall_confidence ?? data.confidence) as number,
            contributing: (data.contributing_agents as string[]) || [],
            unavailable: (data.unavailable_agents as string[]) || [],
            fusion_method: data.fusion_method as string | undefined,
            predicted_category: data.predicted_category as string | undefined,
            category_probabilities: data.category_probabilities as Record<string, number> | undefined,
            dynamic_weights: data.dynamic_weights as Record<string, number> | undefined,
            decision_threshold: data.decision_threshold as number | undefined,
            decision_thresholds: data.decision_thresholds as Verdict['decision_thresholds'],
            fake_probability: data.fake_probability as number | undefined,
            decision_confidence: data.decision_confidence as number | undefined,
            verdict_class: verdictClass,
            is_abstention: isAbstention,
            leading_category_is_hypothesis: Boolean(
              data.leading_category_is_hypothesis ?? isAbstention
            ),
          });

        } else if (eventName === 'done') {
          return true;

        } else if (eventName === 'error') {
          hasError = true;
          setErrorMessage(
            (data.message ?? data.error ?? 'An unexpected error occurred during inference') as string
          );
          setAppState('error');
          return true;
        }
      });

      if (!hasError) {
        if (verdictReceived) {
          setAppState('done');
        } else {
          setErrorMessage('Connection closed before analysis verdict was received.');
          setAppState('error');
        }
      }

    } catch (err: any) {
      if (err.name === 'AbortError') return; // the page was left mid-analysis
      console.error(err);
      // fetch rejects with a TypeError when the dev server itself can't be reached.
      setErrorMessage(err instanceof TypeError ? BACKEND_UNREACHABLE : err.message || 'Failed to connect to analysis stream');
      setAppState('error');
    }
  };

  const status =
    analysing ? `Analysing, ${reported} of 4 reported`
    : appState === 'done' ? 'Analysis complete'
    : appState === 'error' ? 'Analysis failed'
    : selectedFile ? 'Ready to analyse'
    : 'Add a file to begin';

  // ----------------------------------------------------------------------
  // Render
  // ----------------------------------------------------------------------
  return (
    <MotionConfig reducedMotion="user">
      <div className="min-h-screen w-full bg-paper text-ink selection:bg-[#a5d8ff]">
        {/* Same glass bar the landing nav docks into */}
        <header className="island-nav liquid-glass" data-docked="true">
          <div className="grid grid-cols-[1fr_auto_1fr] items-center gap-2">
            <div className="flex justify-start">
              <button
                onClick={() => navigateTo('/')}
                className="flex items-center gap-2 rounded-full px-3 py-2 text-sm text-ink/70 transition-colors hover:bg-white/60 hover:text-ink focus-visible:outline-2 focus-visible:outline-signal cursor-pointer"
              >
                <ArrowLeft className="h-4 w-4" />
                <span className="hidden sm:inline">Home</span>
              </button>
            </div>
            <span className="wordmark flex h-11 items-center px-2">VERITAS</span>
            <p className="flex items-center justify-end gap-2 text-right text-sm text-ink/70" aria-live="polite">
              {analysing && <span className="h-2 w-2 shrink-0 animate-pulse rounded-full bg-signal" />}
              <span className="truncate">{status}</span>
            </p>
          </div>
        </header>

        <main className="mx-auto grid max-w-6xl gap-6 px-4 pt-24 pb-16 sm:px-6 lg:grid-cols-[minmax(0,380px)_minmax(0,1fr)] lg:gap-8 lg:pt-28">
          {/* ── Left: the file ─────────────────────────────────────────── */}
          <section className="lg:sticky lg:top-28 lg:self-start">
            <div className="rounded-[1.75rem] bg-white p-5 shadow-[0_24px_60px_-34px_rgba(14,22,48,0.35)] sm:p-6">
              <h1 className="text-3xl font-medium tracking-[-0.03em]">Check a file</h1>
              <p className="mt-2 text-sm leading-relaxed text-ink/60">
                Images, videos or audio up to 50 MB. VERITAS picks the right models for the file type.
              </p>

              <input
                ref={fileInputRef}
                id="console-file"
                type="file"
                accept={ACCEPT}
                disabled={analysing}
                className="sr-only"
                onChange={(e) => {
                  const f = e.target.files?.[0];
                  if (f) pickFile(f);
                  e.target.value = '';
                }}
              />

              {selectedFile && previewUrl && kind ? (
                <div
                  {...dropProps}
                  className={`mt-5 overflow-hidden rounded-2xl border transition-colors ${dragging ? 'border-signal bg-[#eef3ff]' : 'border-ink/10 bg-paper'}`}
                >
                  <div className="flex items-center justify-center bg-ink/[0.04]">
                    {kind === 'image' && <img src={previewUrl} alt="Preview of the selected image" className="max-h-64 w-full object-contain" />}
                    {kind === 'video' && <video src={previewUrl} controls muted className="max-h-64 w-full" />}
                    {kind === 'audio' && <audio src={previewUrl} controls className="w-full p-4" />}
                  </div>
                  <div className="flex items-start justify-between gap-3 p-4">
                    <div className="min-w-0">
                      <p className="truncate text-sm font-medium" title={selectedFile.name}>{selectedFile.name}</p>
                      <p className="mt-0.5 text-xs text-ink/55">
                        {kind[0].toUpperCase() + kind.slice(1)}, {(selectedFile.size / 1048576).toFixed(1)} MB
                      </p>
                    </div>
                    <label
                      htmlFor="console-file"
                      className={`shrink-0 rounded-full px-3 py-1.5 text-xs font-medium text-signal transition-colors hover:bg-signal/10 ${analysing ? 'pointer-events-none opacity-40' : 'cursor-pointer'}`}
                    >
                      Change file
                    </label>
                  </div>
                  <p className="border-t border-ink/10 px-4 py-3 text-xs leading-relaxed text-ink/60">{SKIPPED_FOR[kind]}</p>
                </div>
              ) : (
                <label
                  htmlFor="console-file"
                  {...dropProps}
                  className={`mt-5 flex min-h-[200px] cursor-pointer flex-col items-center justify-center gap-3 rounded-2xl border-2 border-dashed p-6 text-center transition-colors focus-within:outline-2 focus-within:outline-signal ${
                    dragging ? 'border-signal bg-[#eef3ff]' : 'border-ink/15 bg-paper hover:border-signal/50'
                  }`}
                >
                  <UploadCloud className={`pointer-events-none h-9 w-9 ${dragging ? 'text-signal' : 'text-ink/40'}`} />
                  <span className="pointer-events-none text-sm font-medium">
                    {dragging ? 'Drop to add it' : 'Drop a file here, or click to choose one'}
                  </span>
                  <span className="pointer-events-none text-xs text-ink/50">JPG, PNG, MP4, MOV, WAV, MP3 and more</span>
                </label>
              )}

              {fileError && (
                <p role="alert" className="mt-3 rounded-xl bg-rose-50 px-4 py-3 text-sm leading-relaxed text-rose-800">
                  {fileError}
                </p>
              )}

              <button
                disabled={!selectedFile || analysing}
                onClick={appState === 'done' ? () => fileInputRef.current?.click() : startAnalysis}
                className="try-it mt-5 flex w-full items-center justify-center gap-2 rounded-full px-5 py-3.5 text-sm font-medium text-white focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-signal enabled:cursor-pointer disabled:cursor-not-allowed disabled:opacity-45"
              >
                {analysing && <span className="h-4 w-4 animate-spin rounded-full border-2 border-white/50 border-t-white" />}
                {analysing ? 'Analysing…' : appState === 'done' ? 'Analyse another file' : 'Analyse file'}
              </button>

              <p className="mt-4 text-xs leading-relaxed text-ink/50">
                Results are an informational assessment, not legal evidence.
              </p>
            </div>
          </section>

          {/* ── Right: verdict, then the evidence behind it ─────────────── */}
          <section ref={resultsRef} className="flex min-w-0 flex-col gap-6 scroll-mt-24" aria-label="Results">
            <VerdictPanel
              appState={appState}
              hasFile={Boolean(selectedFile)}
              reported={reported}
              verdict={verdict}
              errorMessage={errorMessage}
              onRetry={startAnalysis}
              onDownload={downloadReport}
            />

            <p className="text-sm text-ink/60">
              Each card is one model’s own reading. Only the fused score above decides the verdict, so a card can lean
              the other way.
            </p>
            <div className="grid gap-4 md:grid-cols-2">
              {agents.map((ag, idx) => (
                <AgentCard key={ag.agent} agent={ag} index={idx} analysing={analysing} weight={verdict?.dynamic_weights} />
              ))}
            </div>
          </section>
        </main>
      </div>
    </MotionConfig>
  );
}

// ----------------------------------------------------------------------
// Verdict
// ----------------------------------------------------------------------
function VerdictPanel({
  appState, hasFile, reported, verdict, errorMessage, onRetry, onDownload,
}: {
  appState: AppState;
  hasFile: boolean;
  reported: number;
  verdict: Verdict | null;
  errorMessage: string;
  onRetry: () => void;
  onDownload: () => void;
}) {
  const low = verdict?.decision_thresholds?.tau_low ?? DEFAULT_LOW;
  const high = verdict?.decision_thresholds?.tau_high ?? DEFAULT_HIGH;
  const card = 'rounded-[1.75rem] bg-white p-6 shadow-[0_24px_60px_-34px_rgba(14,22,48,0.35)] sm:p-8';

  if (appState === 'error') {
    return (
      <div className={card} role="alert">
        <h2 className="text-3xl font-medium tracking-[-0.03em] text-rose-700">Analysis failed</h2>
        <p className="mt-3 max-w-[60ch] text-sm leading-relaxed text-ink/70">{errorMessage}</p>
        <p className="mt-1 text-sm text-ink/50">Check that the backend is running, then try again.</p>
        <button
          onClick={onRetry}
          className="mt-5 inline-flex items-center gap-2 rounded-full bg-paper px-4 py-2 text-sm font-medium transition-colors hover:bg-ink/10 cursor-pointer"
        >
          <RefreshCcw className="h-4 w-4" />
          Try again
        </button>
      </div>
    );
  }

  if (!verdict) {
    const running = appState === 'analysing';
    return (
      <div className={card}>
        <h2 className="text-3xl font-medium tracking-[-0.03em]">
          {running ? 'Analysing your file' : hasFile ? 'Ready when you are' : 'Your verdict will appear here'}
        </h2>
        <p className="mt-3 max-w-[60ch] text-sm leading-relaxed text-ink/60">
          {running
            ? `${reported} of 4 models have reported. The verdict follows once fusion weighs their scores.`
            : 'Each model scores how likely the file is fake. VERITAS weighs those scores into one number and places it on this scale.'}
        </p>
        {running && (
          <div className="mt-5 h-1 overflow-hidden rounded-full bg-ink/10">
            <motion.div className="h-full bg-signal" animate={{ width: `${(reported / 4) * 100}%` }} transition={{ duration: 0.4 }} />
          </div>
        )}
        <ScoreScale low={low} high={high} />
      </div>
    );
  }

  const isAuthentic = verdict.verdict === 'AUTHENTIC';
  const isDeepfake = verdict.verdict === 'DEEPFAKE';
  const isUnverified = verdict.verdict === 'UNVERIFIED' || verdict.verdict_class === 'UNVERIFIED';
  const isUncertain = !isUnverified && (verdict.verdict === 'UNCERTAIN'
    || verdict.verdict_class === 'UNCERTAIN'
    || Boolean(verdict.is_abstention && !isDeepfake && !isAuthentic));
  const showHypothesis = Boolean(verdict.leading_category_is_hypothesis || isUncertain || isUnverified);
  const fakeScore = verdict.fake_probability ?? verdict.confidence;
  const shortNames = (list: string[]) => list.map((a) => a.split(' ')[0]).join(', ');

  const headline = isDeepfake ? 'Likely deepfake' : isAuthentic ? 'Likely authentic' : isUnverified ? "Can't verify" : 'Inconclusive';
  const color = isDeepfake ? 'text-rose-700' : isAuthentic ? 'text-emerald-700' : isUnverified ? 'text-ink/60' : 'text-amber-700';

  let subtitle: string;
  if (isUnverified) {
    subtitle = 'None of the models could produce usable evidence, so authenticity cannot be claimed.';
  } else if (isUncertain) {
    subtitle = 'The fused score sits inside the uncertainty band, so VERITAS will not call it either way. Inconclusive does not mean authentic.';
  } else if (verdict.unavailable.length === 0) {
    subtitle = 'All models contributed to this verdict.';
  } else {
    subtitle = `Based on ${shortNames(verdict.contributing)}. ${shortNames(verdict.unavailable)} could not run on this file.`;
  }

  const weights = Object.entries(verdict.dynamic_weights ?? {}).sort((a, b) => b[1] - a[1]);
  const categories = Object.entries(verdict.category_probabilities ?? {}).sort((a, b) => b[1] - a[1]);

  return (
    <motion.div
      className={card}
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, ease: [0.22, 1, 0.36, 1] }}
    >
      <div className="flex flex-wrap items-end justify-between gap-4">
        <h2 className={`text-4xl font-medium tracking-[-0.035em] sm:text-5xl ${color}`}>{headline}</h2>
        <div className="text-right">
          <p className={`font-mono text-4xl ${color}`}>{pct(fakeScore)}</p>
          <p className="text-sm text-ink/55">fused fake score</p>
        </div>
      </div>
      <p className="mt-3 max-w-[62ch] text-sm leading-relaxed text-ink/70">{subtitle}</p>

      <ScoreScale score={fakeScore} low={low} high={high} />

      {(weights.length > 0 || categories.length > 0) && (
        <div className="mt-8 grid gap-8 border-t border-ink/10 pt-6 sm:grid-cols-2">
          {weights.length > 0 && (
            <div>
              <h3 className="text-sm font-medium">How much each model counted</h3>
              <p className="mt-1 text-xs text-ink/55">Weights are rescaled over the models that ran.</p>
              <Bars rows={weights} />
            </div>
          )}
          {categories.length > 0 && (
            <div>
              <h3 className="text-sm font-medium">Likely manipulation type</h3>
              <p className="mt-1 text-xs text-ink/55">
                {showHypothesis ? 'A hypothesis only: the evidence is not strong enough to name one.' : 'How the fused evidence splits across manipulation types.'}
              </p>
              <Bars rows={categories} />
            </div>
          )}
        </div>
      )}

      <div className="mt-8 flex flex-wrap items-center justify-between gap-3 border-t border-ink/10 pt-5 text-xs text-ink/55">
        <span>
          Fusion: <span className="font-mono text-ink/80">{verdict.fusion_method ?? 'corrected_fixed'}</span>
          {verdict.decision_confidence != null && <>, decision confidence <span className="font-mono text-ink/80">{pct(verdict.decision_confidence)}</span></>}
        </span>
        <button
          onClick={onDownload}
          className="inline-flex items-center gap-2 rounded-full bg-paper px-4 py-2 text-sm font-medium text-ink transition-colors hover:bg-ink/10 cursor-pointer"
        >
          <Download className="h-4 w-4" />
          Download report
        </button>
      </div>
    </motion.div>
  );
}

function ScoreScale({ score, low, high }: { score?: number; low: number; high: number }) {
  return (
    <div className="mt-7">
      <div className="relative h-3">
        <div className="absolute inset-0 flex overflow-hidden rounded-full">
          <div className="bg-emerald-500/80" style={{ width: `${low * 100}%` }} />
          <div className="bg-amber-400" style={{ width: `${(high - low) * 100}%` }} />
          <div className="flex-1 bg-rose-500/80" />
        </div>
        {score !== undefined && (
          <motion.div
            className="absolute top-1/2 h-6 w-6 -translate-x-1/2 -translate-y-1/2 rounded-full border-[3px] border-white bg-ink shadow-[0_2px_8px_rgba(14,22,48,0.35)]"
            initial={{ left: '0%' }}
            animate={{ left: `${Math.min(Math.max(score, 0), 1) * 100}%` }}
            transition={{ duration: 1.1, ease: [0.22, 1, 0.36, 1] }}
            aria-hidden="true"
          />
        )}
      </div>
      <div className="relative mt-2 h-4 font-mono text-[11px] text-ink/45">
        <span className="absolute left-0">0</span>
        <span className="absolute -translate-x-full pr-1" style={{ left: `${low * 100}%` }}>{low.toFixed(2)}</span>
        <span className="absolute pl-1" style={{ left: `${high * 100}%` }}>{high.toFixed(2)}</span>
        <span className="absolute right-0">1</span>
      </div>
      <ul className="mt-4 flex flex-wrap gap-x-5 gap-y-1 text-xs text-ink/60">
        <li className="flex items-center gap-1.5"><span className="h-2 w-2 rounded-full bg-emerald-500" />Below {low.toFixed(2)}: authentic</li>
        <li className="flex items-center gap-1.5"><span className="h-2 w-2 rounded-full bg-amber-400" />In between: inconclusive</li>
        <li className="flex items-center gap-1.5"><span className="h-2 w-2 rounded-full bg-rose-500" />Above {high.toFixed(2)}: deepfake</li>
      </ul>
    </div>
  );
}

function Bars({ rows }: { rows: [string, number][] }) {
  return (
    <ul className="mt-4 space-y-3">
      {rows.map(([label, value]) => (
        <li key={label}>
          <div className="flex justify-between text-xs">
            <span className="capitalize text-ink/75">{prettify(label)}</span>
            <span className="font-mono text-ink/60">{pct(value)}</span>
          </div>
          <div className="mt-1 h-1.5 overflow-hidden rounded-full bg-ink/[0.07]">
            <motion.div
              className="h-full rounded-full bg-signal"
              initial={{ width: 0 }}
              animate={{ width: `${value * 100}%` }}
              transition={{ duration: 0.8, ease: [0.22, 1, 0.36, 1] }}
            />
          </div>
        </li>
      ))}
    </ul>
  );
}

// ----------------------------------------------------------------------
// Agent card
// ----------------------------------------------------------------------
const UNAVAILABLE_STATUSES = ['untrained', 'mocked', 'no_audio', 'not_applicable', 'unavailable', 'error', 'failed'];

function AgentCard({
  agent, index, analysing, weight,
}: {
  agent: AgentResult;
  index: number;
  analysing: boolean;
  weight?: Record<string, number>;
}) {
  const [showHeatmap, setShowHeatmap] = useState(false);
  const info = AGENT_INFO[index];
  const isPending = agent.status === 'pending';
  const isExplanation = agent.backend_status === 'explanation';
  const isUnavailable = !isPending && !isExplanation && (agent.is_fake === null || agent.is_fake === undefined || UNAVAILABLE_STATUSES.includes(agent.backend_status ?? ''));
  // Neutral on purpose: a single model's score is a signal, not a decision, so it must not be coloured as one.
  const tone = { text: 'text-ink/70', bar: 'bg-ink/40' };
  const counted = Object.entries(weight ?? {}).find(([k]) => k.toLowerCase().includes(info.name.toLowerCase()))?.[1];

  const label = isPending ? (analysing ? 'Analysing' : 'Waiting') : isUnavailable ? 'Skipped' : 'Reported';

  return (
    <article
      className={`flex flex-col gap-4 rounded-[1.25rem] border p-5 transition-colors ${
        isUnavailable ? 'border-ink/[0.06] bg-white/50' : 'border-ink/10 bg-white'
      }`}
    >
      <header className="flex items-start justify-between gap-3">
        <div>
          <h3 className="text-base font-medium">{info.name}</h3>
          <p className="font-mono text-xs text-ink/50">{info.model}</p>
        </div>
        <span
          className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs ${
            isPending && analysing ? 'bg-signal/10 text-signal' : isUnavailable ? 'bg-ink/[0.06] text-ink/55' : isPending ? 'bg-ink/[0.04] text-ink/50' : 'bg-paper text-ink/70'
          }`}
        >
          {isPending && analysing && <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-signal" />}
          {label}
        </span>
      </header>

      <div className="flex-1">
        {isPending ? (
          <>
            <p className="text-sm leading-relaxed text-ink/60">{info.checks}</p>
            {analysing && (
              <div className="mt-3 space-y-2" aria-hidden="true">
                <div className="h-2.5 w-full animate-pulse rounded bg-ink/[0.06]" />
                <div className="h-2.5 w-4/6 animate-pulse rounded bg-ink/[0.06]" />
              </div>
            )}
          </>
        ) : (
          <motion.p initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="text-sm leading-relaxed text-ink/75">
            {agent.finding}
          </motion.p>
        )}
      </div>

      {!isPending && !isUnavailable && (
        <div>
          <div className="mb-1.5 flex justify-between text-xs">
            <span className="text-ink/55">
              {isExplanation ? 'Visual–audio score gap' : 'This model’s fake probability'}
              {counted !== undefined && `, counted at ${pct(counted)}`}
            </span>
            <span className={`font-mono ${tone.text}`}>{pct(agent.confidence)}</span>
          </div>
          <div className="h-1.5 w-full overflow-hidden rounded-full bg-ink/[0.07]">
            <motion.div
              initial={{ width: 0 }}
              animate={{ width: `${agent.confidence * 100}%` }}
              transition={{ duration: 0.6, ease: 'easeOut' }}
              className={`h-full rounded-full ${tone.bar}`}
            />
          </div>
        </div>
      )}

      {agent.gradcam_b64 && (
        <div>
          <button
            onClick={() => setShowHeatmap((v) => !v)}
            aria-expanded={showHeatmap}
            className="text-sm font-medium text-signal hover:underline cursor-pointer"
          >
            {showHeatmap ? 'Hide heatmap' : 'Show where it looked'}
          </button>
          <AnimatePresence initial={false}>
            {showHeatmap && (
              <motion.div
                initial={{ height: 0, opacity: 0 }}
                animate={{ height: 'auto', opacity: 1 }}
                exit={{ height: 0, opacity: 0 }}
                transition={{ duration: 0.3, ease: [0.22, 1, 0.36, 1] }}
                className="overflow-hidden"
              >
                <img
                  src={`data:image/png;base64,${agent.gradcam_b64}`}
                  alt="GradCAM heatmap of the regions that most influenced the spatial model"
                  className="mt-3 max-h-72 w-full rounded-xl object-contain"
                />
                <p className="mt-2 text-xs leading-relaxed text-ink/55">
                  Warmer colours mark the regions that pushed the model hardest toward a fake classification.
                </p>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      )}
    </article>
  );
}
