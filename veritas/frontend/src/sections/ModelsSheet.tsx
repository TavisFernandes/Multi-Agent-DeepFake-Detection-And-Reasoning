import { useState } from 'react';
import { AnimatePresence, motion } from 'framer-motion';
import { ArrowRight, ChevronRight } from 'lucide-react';
import { usePageTransition } from '@/context/TransitionContext';

const MODELS = [
  { id: 'spatial', label: 'Spatial', Content: VisualEvidenceContent, Graphic: VisualEvidenceGraphic },
  { id: 'temporal', label: 'Temporal', Content: TemporalContent, Graphic: TemporalGraphic },
  { id: 'audio', label: 'Audio', Content: AudioContent, Graphic: AudioGraphic },
  { id: 'fusion', label: 'Fusion', Content: FusionContent, Graphic: FusionGraphic },
];

// Bottom sheet that slides up over the sticky hero; one panel, switched by the segmented control.
export function ModelsSheet() {
  const [active, setActive] = useState(MODELS[0].id);
  const model = MODELS.find((m) => m.id === active)!;

  return (
    <section
      id="models"
      className="relative z-10 min-h-[100svh] rounded-t-[2.5rem] bg-white px-5 pt-5 pb-24 shadow-[0_-30px_80px_-30px_rgba(14,22,48,0.35)] sm:px-10"
    >
      <div className="mx-auto h-1.5 w-12 rounded-full bg-ink/15" aria-hidden="true" />

      <div className="mx-auto max-w-6xl pt-16">
        <div className="flex flex-col gap-6 md:flex-row md:items-end md:justify-between">
          <h2 className="max-w-[16ch] text-4xl font-medium tracking-[-0.03em] text-ink sm:text-5xl">
            How each model reads your media
          </h2>
          <div role="tablist" aria-label="Models" className="flex self-start rounded-full bg-paper p-1 md:self-auto">
            {MODELS.map((m) => (
              <button
                key={m.id}
                role="tab"
                aria-selected={active === m.id}
                aria-controls="model-panel"
                onClick={() => setActive(m.id)}
                className={`relative rounded-full px-3 py-2 text-sm transition-colors focus-visible:outline-2 focus-visible:outline-signal sm:px-5 cursor-pointer ${
                  active === m.id ? 'text-ink' : 'text-ink/55 hover:text-ink'
                }`}
              >
                {active === m.id && (
                  <motion.span
                    layoutId="model-pill"
                    className="absolute inset-0 rounded-full bg-white shadow-[0_1px_3px_rgba(14,22,48,0.15)]"
                    transition={{ type: 'spring', bounce: 0.15, duration: 0.45 }}
                  />
                )}
                <span className="relative">{m.label}</span>
              </button>
            ))}
          </div>
        </div>

        <AnimatePresence mode="wait" initial={false}>
          <motion.div
            key={model.id}
            id="model-panel"
            role="tabpanel"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            transition={{ duration: 0.25, ease: [0.22, 1, 0.36, 1] }}
            className="mt-14 grid items-center gap-10 lg:min-h-[480px] lg:grid-cols-2 lg:gap-16"
          >
            <model.Content />
            <div className="flex justify-center">
              <model.Graphic />
            </div>
          </motion.div>
        </AnimatePresence>
      </div>
    </section>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Content Components
// ─────────────────────────────────────────────────────────────────────────────

function VisualEvidenceContent() {
  return (
    <div className="space-y-6">
      <h2 className="text-3xl sm:text-4xl lg:text-5xl font-bold tracking-tight text-ink leading-tight">
        Sub-pixel artifact & boundary inspection.
      </h2>
      <p className="text-zinc-600 text-sm sm:text-base leading-relaxed max-w-xl">
        The Spatial Agent extracts high-entropy face crops at 224×224 resolution. Evaluated through an EfficientNet-B0
        backbone, it detects sub-pixel warping, GAN blending seams, and generative noise patterns.
      </p>
      <ul className="space-y-2.5 font-mono text-xs text-zinc-600 border-l border-blue-500/40 pl-4">
        <li className="flex items-center gap-2">
          <ChevronRight className="w-3.5 h-3.5 text-blue-700" />
          <span>Isolates face-boundary boundary blending seams</span>
        </li>
        <li className="flex items-center gap-2">
          <ChevronRight className="w-3.5 h-3.5 text-blue-700" />
          <span>Computes gradient-weighted class activation heatmaps (GradCAM)</span>
        </li>
        <li className="flex items-center gap-2">
          <ChevronRight className="w-3.5 h-3.5 text-blue-700" />
          <span>Frozen production weight: <strong className="text-zinc-800">0.40</strong></span>
        </li>
      </ul>
    </div>
  );
}

function TemporalContent() {
  return (
    <div className="space-y-6">
      <h2 className="text-3xl sm:text-4xl lg:text-5xl font-bold tracking-tight text-ink leading-tight">
        Spatio-temporal motion across 16 frames.
      </h2>
      <p className="text-zinc-600 text-sm sm:text-base leading-relaxed max-w-xl">
        Synthetic face swaps that deceive static inspections frequently collapse in motion. The Temporal Agent samples a
        16-frame tensor sequence through 3D ResNet convolutions (R3D-18), exposing inter-frame micro-flickering and facial landmark jitter.
      </p>
      <ul className="space-y-2.5 font-mono text-xs text-zinc-600 border-l border-blue-500/40 pl-4">
        <li className="flex items-center gap-2">
          <ChevronRight className="w-3.5 h-3.5 text-blue-700" />
          <span>Evaluates 3D spatio-temporal video tensors</span>
        </li>
        <li className="flex items-center gap-2">
          <ChevronRight className="w-3.5 h-3.5 text-blue-700" />
          <span>Catches expression desynchronization and unnatural blinking</span>
        </li>
        <li className="flex items-center gap-2">
          <ChevronRight className="w-3.5 h-3.5 text-blue-700" />
          <span>Frozen production weight: <strong className="text-zinc-800">0.30</strong></span>
        </li>
      </ul>
    </div>
  );
}

function AudioContent() {
  return (
    <div className="space-y-6">
      <h2 className="text-3xl sm:text-4xl lg:text-5xl font-bold tracking-tight text-ink leading-tight">
        Mel-spectrogram synthesis & vocoder detection.
      </h2>
      <p className="text-zinc-600 text-sm sm:text-base leading-relaxed max-w-xl">
        Modern deception marries modified visuals with AI-cloned audio. The Audio Agent converts 16 kHz audio streams into
        128 log-mel spectrogram bands to identify neural vocoder synthesis artifacts. Media lacking audio is cleanly bypassed.
      </p>
      <ul className="space-y-2.5 font-mono text-xs text-zinc-600 border-l border-blue-500/40 pl-4">
        <li className="flex items-center gap-2">
          <ChevronRight className="w-3.5 h-3.5 text-blue-700" />
          <span>128-band log-mel spectrogram frequency decomposition</span>
        </li>
        <li className="flex items-center gap-2">
          <ChevronRight className="w-3.5 h-3.5 text-blue-700" />
          <span>Detects synthetic phase dispersion in high-frequency formants</span>
        </li>
        <li className="flex items-center gap-2">
          <ChevronRight className="w-3.5 h-3.5 text-blue-700" />
          <span>Joins the visual score by noisy-OR; skipped when there is no soundtrack</span>
        </li>
      </ul>
    </div>
  );
}

function FusionContent() {
  return (
    <div className="space-y-6">
      <h2 className="text-3xl sm:text-4xl lg:text-5xl font-bold tracking-tight text-ink leading-tight">
        Visual evidence first, then audio, then a three-way verdict.
      </h2>
      <p className="text-zinc-600 text-sm sm:text-base leading-relaxed max-w-xl">
        The production fusion engine (<code>corrected_fixed</code>) blends the two visual models, then joins that with
        audio by noisy-OR, so a fake in either stream raises the score. Context checks that the streams agree but does
        not move the score.
      </p>
      <div className="p-4 rounded-xl bg-white/80 border border-ink/10 font-mono text-xs space-y-2 text-zinc-700">
        <div className="text-blue-700 font-semibold">P = 1 − (1 − P_visual) · (1 − P_audio)</div>
        <p className="text-zinc-600 text-[11px] leading-relaxed">
          P_visual = (0.40 · spatial + 0.30 · temporal) / 0.70. A missing stream drops out instead of diluting the score.
        </p>
        <div className="pt-2 flex flex-wrap gap-2 text-[11px]">
          <span className="px-2 py-0.5 rounded bg-emerald-500/10 text-emerald-700 border border-emerald-500/20">AUTHENTIC &lt; 0.45</span>
          <span className="px-2 py-0.5 rounded bg-amber-500/10 text-amber-700 border border-amber-500/20">UNCERTAIN [0.45 - 0.55]</span>
          <span className="px-2 py-0.5 rounded bg-rose-500/10 text-rose-700 border border-rose-500/20">DEEPFAKE &gt; 0.55</span>
        </div>
      </div>
    </div>
  );
}

export function VerifyCta() {
  const { navigateTo } = usePageTransition();
  const onOpenConsole = () => navigateTo('/console');
  return (
    <section id="console" className="relative w-full py-28 px-4 sm:px-6 lg:px-12 bg-paper border-t border-ink/10">
    <div className="max-w-4xl mx-auto text-center space-y-8">

      <h2 className="text-4xl sm:text-5xl lg:text-6xl font-bold tracking-tight text-ink leading-tight">
        Ready to verify a video?
      </h2>

      <p className="text-zinc-600 text-base sm:text-lg max-w-2xl mx-auto leading-relaxed font-sans">
        Upload suspect video containers, static portraits, or voice streams for real multi-agent analysis.
        VERITAS executes live inference across Spatial, Temporal, Audio, and Context neural networks with noisy-OR evidence fusion (<code className="text-blue-700 bg-blue-500/10 px-1 rounded">corrected_fixed</code>).
      </p>

      <div className="pt-2 flex flex-col sm:flex-row items-center justify-center gap-4">
        <button
          onClick={onOpenConsole}
          data-testid="button-open-live-console"
          className="inline-flex items-center gap-2.5 px-8 py-4 rounded-xl bg-blue-600 hover:bg-blue-500 text-white font-semibold text-base transition-all duration-200 shadow-lg shadow-blue-500/25 hover:shadow-blue-500/40 hover:-translate-y-0.5 cursor-pointer"
        >
          <span>Open Live Console</span>
          <ArrowRight className="w-4 h-4" />
        </button>
      </div>

      <div className="pt-8 grid grid-cols-2 sm:grid-cols-4 gap-4 border-t border-ink/10 font-mono text-xs text-zinc-500 text-center">
        <div>
          <span className="text-zinc-700 font-medium block">DRAG & DROP</span>
          <span>MP4, AVI, MKV, WAV</span>
        </div>
        <div>
          <span className="text-zinc-700 font-medium block">STREAMING TELEMETRY</span>
          <span>Live SSE Agent Log</span>
        </div>
        <div>
          <span className="text-zinc-700 font-medium block">EXPLAINABLE GRADCAM</span>
          <span>Visual Heatmap Loci</span>
        </div>
        <div>
          <span className="text-zinc-700 font-medium block">FROZEN RUNTIME</span>
          <span>corrected_fixed</span>
        </div>
      </div>
    </div>
    </section>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Custom Architectural Graphics (Zero external 3D, Pure SVG/CSS Art Direction)
// ─────────────────────────────────────────────────────────────────────────────

function VisualEvidenceGraphic() {
  return (
    <div className="relative w-full max-w-md aspect-square rounded-2xl border border-ink/10 bg-paper p-6 flex flex-col justify-between overflow-hidden shadow-[0_24px_60px_-30px_rgba(14,22,48,0.35)]">
      <div className="absolute inset-0 bg-[linear-gradient(to_right,#0e16300c_1px,transparent_1px),linear-gradient(to_bottom,#0e16300c_1px,transparent_1px)] bg-[size:20px_20px] pointer-events-none" />

      <div className="relative flex items-center justify-between font-mono text-[11px] text-zinc-500 z-10 border-b border-ink/10 pb-3">
        <span className="text-zinc-700 font-semibold">TENSOR: 224×224 RGB</span>
        <span className="text-emerald-700">GRADCAM_ACTIVE</span>
      </div>

      <div className="relative my-auto flex items-center justify-center z-10">
        <div className="relative w-48 h-56 rounded-xl border border-ink/15 bg-white overflow-hidden shadow-inner flex items-center justify-center">
          <svg className="w-32 h-44 text-zinc-400" viewBox="0 0 100 120" fill="none" stroke="currentColor" strokeWidth="1.2">
            <ellipse cx="50" cy="55" rx="32" ry="42" />
            <path d="M35 50 Q40 46 45 50" />
            <path d="M55 50 Q60 46 65 50" />
            <path d="M50 56 L50 68 L46 72 L54 72" />
            <path d="M40 84 Q50 90 60 84" />
          </svg>

          <div className="absolute top-12 left-8 w-16 h-16 rounded-full bg-rose-500/30 blur-md pointer-events-none" />
          <div className="absolute top-24 right-10 w-20 h-14 rounded-full bg-amber-500/25 blur-lg pointer-events-none" />
          <div className="absolute bottom-6 left-12 w-24 h-12 rounded-full bg-blue-500/20 blur-md pointer-events-none" />

          <div className="absolute inset-4 border border-rose-500/40 rounded-lg pointer-events-none">
            <span className="absolute -top-2 left-2 px-1 text-[9px] font-mono bg-rose-500 text-white rounded">
              SEAM: 0.884
            </span>
          </div>
        </div>
      </div>

      <div className="relative flex justify-between font-mono text-[10px] text-zinc-600 z-10 border-t border-ink/10 pt-3">
        <span>LOCI: PERI-ORBITAL BLENDING</span>
        <span className="text-blue-700">EfficientNet-B0</span>
      </div>
    </div>
  );
}

function TemporalGraphic() {
  return (
    <div className="relative w-full max-w-md aspect-square rounded-2xl border border-ink/10 bg-paper p-6 flex flex-col justify-between overflow-hidden shadow-[0_24px_60px_-30px_rgba(14,22,48,0.35)]">
      <div className="absolute inset-0 bg-[linear-gradient(to_right,#0e16300c_1px,transparent_1px),linear-gradient(to_bottom,#0e16300c_1px,transparent_1px)] bg-[size:20px_20px] pointer-events-none" />

      <div className="relative flex items-center justify-between font-mono text-[11px] text-zinc-500 z-10 border-b border-ink/10 pb-3">
        <span className="text-zinc-700 font-semibold">WINDOW: 16 FRAMES</span>
        <span className="text-blue-700">R3D-18 CONV3D</span>
      </div>

      <div className="relative my-auto flex flex-col gap-3 z-10 w-full">
        <div className="grid grid-cols-4 gap-2">
          {['T_01', 'T_04', 'T_08', 'T_12'].map((t, idx) => (
            <div
              key={t}
              className={`h-24 rounded-lg border p-2 flex flex-col justify-between ${
                idx === 2 ? 'border-amber-500/60 bg-amber-500/10' : 'border-ink/10 bg-white/80'
              }`}
            >
              <div className="flex justify-between font-mono text-[9px] text-zinc-600">
                <span>{t}</span>
                {idx === 2 && <span className="text-amber-700 font-bold">JITTER</span>}
              </div>
              <div className="w-full flex justify-center py-1">
                <div
                  className={`w-7 h-9 rounded border ${
                    idx === 2 ? 'border-amber-500 bg-amber-400/20' : 'border-zinc-300 bg-zinc-200'
                  }`}
                />
              </div>
              <div className="w-full bg-zinc-200 h-1 rounded-full overflow-hidden">
                <div
                  className={`h-full ${idx === 2 ? 'bg-amber-400 w-4/5' : 'bg-blue-600 w-1/3'}`}
                />
              </div>
            </div>
          ))}
        </div>

        <div className="relative h-6 w-full flex items-center px-2">
          <div className="h-[1px] w-full bg-zinc-300" />
          <div className="absolute left-[62%] -top-1 w-2 h-2 rounded-full bg-amber-400 animate-ping" />
          <div className="absolute left-[62%] top-0 w-2 h-2 rounded-full bg-amber-400" />
        </div>
      </div>

      <div className="relative flex justify-between font-mono text-[10px] text-zinc-600 z-10 border-t border-ink/10 pt-3">
        <span>INTER-FRAME JITTER DETECTED</span>
        <span className="text-amber-700">Δt: 0.16s DISCONTINUITY</span>
      </div>
    </div>
  );
}

function AudioGraphic() {
  return (
    <div className="relative w-full max-w-md aspect-square rounded-2xl border border-ink/10 bg-paper p-6 flex flex-col justify-between overflow-hidden shadow-[0_24px_60px_-30px_rgba(14,22,48,0.35)]">
      <div className="absolute inset-0 bg-[linear-gradient(to_right,#0e16300c_1px,transparent_1px),linear-gradient(to_bottom,#0e16300c_1px,transparent_1px)] bg-[size:20px_20px] pointer-events-none" />

      <div className="relative flex items-center justify-between font-mono text-[11px] text-zinc-500 z-10 border-b border-ink/10 pb-3">
        <span className="text-zinc-700 font-semibold">SPECTRUM: 128 MEL-BANDS</span>
        <span className="text-blue-700">16 kHz MONO</span>
      </div>

      <div className="relative my-auto flex flex-col gap-1.5 z-10 w-full p-2">
        {[
          { label: '8.0 kHz', bars: [20, 35, 60, 85, 45, 90, 75, 30, 95, 60] },
          { label: '4.0 kHz', bars: [40, 70, 90, 50, 80, 60, 85, 70, 40, 80] },
          { label: '2.0 kHz', bars: [60, 85, 40, 75, 90, 65, 50, 85, 70, 90] },
          { label: '1.0 kHz', bars: [80, 50, 70, 90, 60, 85, 95, 60, 80, 50] },
          { label: '250 Hz',  bars: [95, 90, 85, 80, 75, 70, 80, 85, 90, 95] },
        ].map((band, idx) => (
          <div key={band.label} className="flex items-center gap-2">
            <span className="font-mono text-[9px] text-zinc-500 w-12 text-right">{band.label}</span>
            <div className="flex-1 flex gap-1 h-3 items-end">
              {band.bars.map((h, i) => (
                <div
                  key={i}
                  style={{ height: `${h}%` }}
                  className={`flex-1 rounded-sm ${
                    idx === 0 && i >= 6 ? 'bg-rose-500/80' : 'bg-blue-500/60'
                  }`}
                />
              ))}
            </div>
          </div>
        ))}
      </div>

      <div className="relative flex justify-between font-mono text-[10px] text-zinc-600 z-10 border-t border-ink/10 pt-3">
        <span>VOCODER DISPERSION AT 8.0 kHz</span>
        <span className="text-rose-700">PHASE ANOMALY</span>
      </div>
    </div>
  );
}

function FusionGraphic() {
  return (
    <div className="relative w-full max-w-md aspect-square rounded-2xl border border-ink/10 bg-paper p-6 flex flex-col justify-between overflow-hidden shadow-[0_24px_60px_-30px_rgba(14,22,48,0.35)]">
      <div className="absolute inset-0 bg-[linear-gradient(to_right,#0e16300c_1px,transparent_1px),linear-gradient(to_bottom,#0e16300c_1px,transparent_1px)] bg-[size:20px_20px] pointer-events-none" />

      <div className="relative flex items-center justify-between font-mono text-[11px] text-zinc-500 z-10 border-b border-ink/10 pb-3">
        <span className="text-zinc-700 font-semibold">SYNTHESIS: corrected_fixed</span>
        <span className="text-emerald-700">FROZEN_PROD</span>
      </div>

      <div className="relative my-auto flex flex-col items-center justify-center gap-3 z-10 w-full py-2">
        <div className="grid grid-cols-4 gap-2 w-full text-center font-mono text-[10px]">
          <div className="p-2 rounded bg-white border border-ink/10">
            <span className="text-zinc-600 block">SPATIAL</span>
            <span className="text-blue-700 font-bold">0.40</span>
          </div>
          <div className="p-2 rounded bg-white border border-ink/10">
            <span className="text-zinc-600 block">TEMPORAL</span>
            <span className="text-blue-700 font-bold">0.30</span>
          </div>
          <div className="p-2 rounded bg-white border border-ink/10">
            <span className="text-zinc-600 block">AUDIO</span>
            <span className="text-blue-700 font-bold">OR</span>
          </div>
          <div className="p-2 rounded bg-white border border-ink/10">
            <span className="text-zinc-600 block">CONTEXT</span>
            <span className="text-blue-700 font-bold">check</span>
          </div>
        </div>

        <div className="w-full p-3 rounded-xl border border-blue-500/40 bg-blue-500/10 text-center font-mono space-y-1">
          <div className="text-xs text-ink font-bold tracking-wider">VISUAL MEAN, THEN NOISY-OR</div>
          <div className="text-[11px] text-blue-800">P = 1 − (1 − P_visual) · (1 − P_audio)</div>
        </div>

        <div className="grid grid-cols-3 gap-2 w-full text-center font-mono text-[9px]">
          <div className="p-1.5 rounded bg-emerald-500/10 border border-emerald-500/30 text-emerald-700">
            AUTHENTIC
          </div>
          <div className="p-1.5 rounded bg-amber-500/10 border border-amber-500/30 text-amber-700">
            UNCERTAIN
          </div>
          <div className="p-1.5 rounded bg-rose-500/10 border border-rose-500/30 text-rose-700">
            DEEPFAKE
          </div>
        </div>
      </div>

      <div className="relative flex justify-between font-mono text-[10px] text-zinc-600 z-10 border-t border-ink/10 pt-3">
        <span>CONTRACT: TRIPARTITE DEADBAND</span>
        <span className="text-emerald-700">PPO v3: DISABLED</span>
      </div>
    </div>
  );
}
