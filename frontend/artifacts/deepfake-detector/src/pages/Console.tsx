import { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Shield, ArrowLeft, UploadCloud, Image as ImageIcon, Video, Mic,
  Link as LinkIcon, CheckCircle2, AlertTriangle, Copy, User,
  ArrowUp, CheckCircle
} from 'lucide-react';
import { usePageTransition } from '@/context/TransitionContext';
import { useTypingEffect } from '@/hooks/useTypingEffect';

const MOCK_FLOW = [
  { role: 'user', text: "I've uploaded a video. Can you check whether it's authentic?" },
  { role: 'system', text: "I've received your content. I'm now examining the visuals, movement, audio, and available context. This usually takes a few moments." },
  { role: 'agent', agent: 'Visual Analysis', color: 'cyan', text: 'Several visual inconsistencies were detected around the face — areas that appear edited or artificially generated.' },
  { role: 'agent', agent: 'Motion Analysis', color: 'blue', text: 'Unusual movement patterns were identified between frames. The motion does not behave as naturally recorded video normally would.' },
  { role: 'agent', agent: 'Audio Analysis', color: 'emerald', text: 'The audio shows characteristics commonly found in synthesised speech. Some frequencies appear inconsistent with natural recording conditions.' },
  { role: 'agent', agent: 'Context Verification', color: 'purple', text: 'Limited online context was found for this content. No matching source could be verified.' },
  { role: 'summary', text: 'Based on the available evidence, this content shows several indicators commonly associated with AI-generated media. Visual inconsistencies, unusual movement, and audio irregularities collectively suggest the content is likely manipulated. VERITAS has high confidence that this media has been artificially generated.' }
];

const PROGRESS_STEPS = [
  { label: 'Examining visuals...', color: 'cyan' },
  { label: 'Checking movement...', color: 'blue' },
  { label: 'Listening to audio...', color: 'emerald' },
  { label: 'Verifying context...', color: 'purple' },
];

// Stagger delays (in seconds) timed to sync with the overlay uncovering.
// The overlay fades away ~350ms after route change. Elements start appearing ~200ms
// into that fade, creating a smooth sequential reveal.
const STAGGER = {
  header: 0.18,
  sidebar: 0.28,
  chat: 0.38,
};

export default function Console() {
  const { navigateTo } = usePageTransition();
  const [activeTab, setActiveTab] = useState('video');
  const [hasFile, setHasFile] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [chatSteps, setChatSteps] = useState(0);
  const [progressSteps, setProgressSteps] = useState(0);
  const chatScrollRef = useRef<HTMLDivElement>(null);

  const visibleMessages = MOCK_FLOW.slice(0, chatSteps);

  const startAnalysis = () => {
    if (!hasFile || chatSteps > 0) return;

    let pStep = 0;
    const pInterval = setInterval(() => {
      pStep++;
      setProgressSteps(pStep);
      if (pStep >= 4) clearInterval(pInterval);
    }, 1500);

    setChatSteps(1);
    let step = 1;
    const interval = setInterval(() => {
      step++;
      setChatSteps(step);
      if (step >= MOCK_FLOW.length) clearInterval(interval);
    }, 2200);
  };

  useEffect(() => {
    if (chatScrollRef.current) {
      chatScrollRef.current.scrollTop = chatScrollRef.current.scrollHeight;
    }
  }, [chatSteps]);

  const tabs = [
    { id: 'image', icon: ImageIcon, label: 'Image' },
    { id: 'video', icon: Video, label: 'Video' },
    { id: 'audio', icon: Mic, label: 'Audio' },
    { id: 'url', icon: LinkIcon, label: 'URL' },
  ];

  return (
    // Page-level entrance: fades in as the overlay uncovers
    <motion.div
      className="h-screen w-full bg-background text-foreground flex flex-col overflow-hidden"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.35, delay: 0.15, ease: 'easeOut' }}
    >
      {/* ── Top Header ─────────────────────────────────────────────────────── */}
      <motion.div
        className="shrink-0 h-[60px] glass-card border-b border-white/10 flex items-center justify-between px-6 z-20"
        initial={{ opacity: 0, y: -12 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4, delay: STAGGER.header, ease: 'easeOut' }}
      >
        <div className="flex items-center gap-3">
          <Shield className="w-6 h-6 text-cyan-400" />
          <span className="font-heading font-bold text-white tracking-wide">VERITAS</span>
          <span className="text-xs text-slate-500 hidden sm:block">AI Content Verification Console</span>
        </div>
        <button
          onClick={() => navigateTo('/')}
          data-testid="button-back-home"
          className="flex items-center gap-2 text-slate-400 hover:text-white transition-colors text-sm"
        >
          <ArrowLeft className="w-4 h-4" />
          Back to Home
        </button>
      </motion.div>

      {/* ── Two-column body ──────────────────────────────────────────────── */}
      <div className="flex flex-1 overflow-hidden">

        {/* LEFT SIDEBAR */}
        <motion.div
          className="w-full max-w-[320px] shrink-0 border-r border-white/10 flex flex-col overflow-y-auto bg-background/50"
          initial={{ opacity: 0, x: -18 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.45, delay: STAGGER.sidebar, ease: 'easeOut' }}
        >
          <div className="p-6 flex flex-col gap-5 flex-1">

            {/* Title */}
            <div>
              <h2 className="text-lg font-heading font-bold text-white mb-1">Analyse Content</h2>
              <p className="text-xs text-slate-500">Upload a file or paste a link below.</p>
            </div>

            {/* Type tabs */}
            <div className="flex bg-white/5 rounded-lg p-1 gap-1">
              {tabs.map(tab => (
                <button
                  key={tab.id}
                  onClick={() => { setActiveTab(tab.id); setHasFile(false); setChatSteps(0); setProgressSteps(0); }}
                  data-testid={`console-tab-${tab.id}`}
                  className={`flex-1 flex flex-col items-center gap-1 py-2 px-1 text-xs font-medium rounded-md transition-colors ${
                    activeTab === tab.id ? 'bg-white/10 text-white' : 'text-slate-400 hover:text-white hover:bg-white/5'
                  }`}
                >
                  <tab.icon className="w-4 h-4" />
                  {tab.label}
                </button>
              ))}
            </div>

            {/* Drop zone */}
            <div
              data-testid="console-dropzone"
              className={`rounded-2xl border-2 border-dashed flex flex-col items-center justify-center p-6 cursor-pointer transition-all duration-300 min-h-[160px] ${
                hasFile
                  ? 'border-cyan-500/50 bg-cyan-500/5'
                  : 'border-white/10 hover:border-cyan-500/40 bg-white/5 hover:bg-white/8'
              }`}
              style={{
                borderImage: hasFile ? undefined : 'none',
                animation: hasFile ? undefined : 'borderPulse 3s ease-in-out infinite',
              }}
              onDragOver={(e) => e.preventDefault()}
              onDrop={(e) => {
                e.preventDefault();
                setIsUploading(true);
                setTimeout(() => { setIsUploading(false); setHasFile(true); }, 1000);
              }}
              onClick={() => {
                if (!hasFile) {
                  setIsUploading(true);
                  setTimeout(() => { setIsUploading(false); setHasFile(true); }, 1000);
                }
              }}
            >
              {isUploading ? (
                <div className="flex flex-col items-center gap-3">
                  <div className="w-10 h-10 rounded-full border-t-2 border-cyan-500 animate-spin" />
                  <p className="text-slate-400 text-sm">Loading...</p>
                </div>
              ) : hasFile ? (
                <div className="flex flex-col items-center gap-3 text-center">
                  <div className="w-12 h-12 rounded-xl bg-cyan-500/20 flex items-center justify-center text-cyan-400">
                    <CheckCircle2 className="w-6 h-6" />
                  </div>
                  <p className="text-white text-sm font-medium">Content loaded</p>
                  <p className="text-xs text-slate-400">sample_video_042.mp4<br />(14.2 MB)</p>
                </div>
              ) : (
                <div className="flex flex-col items-center gap-3 text-center text-slate-400">
                  <UploadCloud className="w-10 h-10" />
                  <div>
                    <p className="text-sm font-medium">Drop your file here, or click to browse</p>
                    <p className="text-xs mt-1 opacity-60">MP4, MOV, WAV, JPG, PNG — Max 50MB</p>
                  </div>
                </div>
              )}
            </div>

            {/* Analyse button */}
            <button
              disabled={!hasFile || chatSteps > 0}
              onClick={startAnalysis}
              data-testid="console-button-analyse"
              className={`w-full py-3.5 rounded-xl font-heading font-bold transition-all text-sm ${
                hasFile && chatSteps === 0
                  ? 'bg-gradient-to-r from-cyan-500 to-blue-600 text-white shadow-[0_0_20px_rgba(6,182,212,0.3)] hover:shadow-[0_0_30px_rgba(6,182,212,0.5)]'
                  : 'bg-white/5 text-slate-500 cursor-not-allowed'
              }`}
            >
              {chatSteps > 0 ? 'Analysing...' : 'Analyse with VERITAS'}
            </button>

            {/* Progress steps */}
            {chatSteps > 0 && (
              <motion.div
                className="flex flex-col gap-2"
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
              >
                {PROGRESS_STEPS.map((step, i) => {
                  const done = progressSteps > i + 1;
                  const active = progressSteps === i + 1;
                  return (
                    <motion.div
                      key={i}
                      initial={{ opacity: 0, x: -8 }}
                      animate={{ opacity: progressSteps > i ? 1 : 0.3, x: 0 }}
                      transition={{ delay: i * 0.1 }}
                      className="flex items-center gap-3 text-xs"
                    >
                      <div className={`w-4 h-4 rounded-full flex items-center justify-center shrink-0 ${
                        done ? `bg-${step.color}-500/20` :
                        active ? `bg-${step.color}-500/10` : 'bg-white/5'
                      }`}>
                        {done ? (
                          <CheckCircle className={`w-3 h-3 text-${step.color}-400`} />
                        ) : active ? (
                          <div className={`w-2 h-2 rounded-full bg-${step.color}-500 animate-pulse`} />
                        ) : (
                          <div className="w-2 h-2 rounded-full bg-white/20" />
                        )}
                      </div>
                      <span className={active ? 'text-white' : done ? 'text-slate-400' : 'text-slate-600'}>
                        {step.label}
                      </span>
                    </motion.div>
                  );
                })}
              </motion.div>
            )}

            {/* Spacer */}
            <div className="flex-1" />

            {/* About blurb */}
            <div className="flex items-start gap-3 pt-4 border-t border-white/5">
              <Shield className="w-4 h-4 text-cyan-500/50 shrink-0 mt-0.5" />
              <p className="text-xs text-slate-500 leading-relaxed">
                VERITAS combines insights from four specialised systems to determine whether content is authentic or AI-generated.
              </p>
            </div>
          </div>
        </motion.div>

        {/* RIGHT CHAT AREA */}
        <motion.div
          className="flex-1 flex flex-col overflow-hidden"
          initial={{ opacity: 0, y: 14 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.45, delay: STAGGER.chat, ease: 'easeOut' }}
        >
          {/* Messages */}
          <div className="flex-1 overflow-y-auto p-6" ref={chatScrollRef}>
            {chatSteps === 0 ? (
              <div className="h-full flex flex-col items-center justify-center gap-6 text-center max-w-md mx-auto">
                <motion.div
                  initial={{ scale: 0.8, opacity: 0 }}
                  animate={{ scale: 1, opacity: 1 }}
                  transition={{ duration: 0.6 }}
                  className="w-20 h-20 rounded-2xl bg-gradient-to-br from-cyan-500/20 to-blue-600/20 border border-cyan-500/30 flex items-center justify-center shadow-[0_0_40px_rgba(6,182,212,0.15)]"
                >
                  <Shield className="w-10 h-10 text-cyan-400" />
                </motion.div>

                <motion.div
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.2 }}
                >
                  <h3 className="text-2xl font-heading font-bold text-white mb-2">What would you like to verify?</h3>
                  <p className="text-slate-400 text-sm leading-relaxed">
                    Upload a file using the panel on the left, then click Analyse.
                  </p>
                </motion.div>

                <motion.div
                  className="flex flex-col gap-2 w-full"
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.4 }}
                >
                  {[
                    "Is this Instagram video real?",
                    "Check if this image is AI-generated",
                    "Verify this audio clip",
                  ].map((prompt) => (
                    <div
                      key={prompt}
                      className="glass-card rounded-xl px-4 py-3 text-sm text-slate-400 border border-white/5 text-left cursor-default select-none"
                    >
                      {prompt}
                    </div>
                  ))}
                </motion.div>
              </div>
            ) : (
              <div className="max-w-3xl mx-auto flex flex-col gap-6">
                <AnimatePresence>
                  {visibleMessages.map((msg, i) => (
                    <motion.div
                      key={i}
                      initial={{ opacity: 0, y: 12 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ duration: 0.3 }}
                      className={`flex gap-4 ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
                    >
                      {msg.role !== 'user' && (
                        <div className="w-9 h-9 rounded-full bg-gradient-to-br from-cyan-500 to-blue-600 flex items-center justify-center shrink-0 shadow-[0_0_12px_rgba(6,182,212,0.4)]">
                          <Shield className="w-4 h-4 text-white" />
                        </div>
                      )}

                      <div className={`max-w-[80%] rounded-2xl px-5 py-4 ${
                        msg.role === 'user'
                          ? 'bg-white/10 text-white rounded-br-sm'
                          : msg.role === 'agent'
                          ? 'bg-background/80 border border-white/10 rounded-tl-sm'
                          : msg.role === 'summary'
                          ? 'bg-gradient-to-br from-red-500/10 to-orange-500/5 border border-red-500/20 rounded-tl-sm'
                          : 'bg-transparent text-slate-400 text-sm rounded-tl-sm'
                      }`}>
                        {msg.role === 'agent' && (
                          <div className="mb-2">
                            <span className={`inline-block px-2 py-0.5 rounded text-xs font-bold uppercase tracking-wider bg-${msg.color}-500/20 text-${msg.color}-400`}>
                              {msg.agent}
                            </span>
                          </div>
                        )}

                        {msg.role === 'summary' ? (
                          <div>
                            <div className="flex items-center gap-2 mb-3 text-red-400 font-heading font-bold">
                              <AlertTriangle className="w-4 h-4" />
                              Likely AI Generated — High Confidence
                            </div>
                            <ConsoleTypewriterText text={msg.text} />
                          </div>
                        ) : (
                          <p className="text-sm md:text-base leading-relaxed text-slate-200">{msg.text}</p>
                        )}

                        {msg.role !== 'user' && msg.role !== 'system' && (
                          <div className="mt-3">
                            <button
                              data-testid={`button-copy-${i}`}
                              className="text-slate-600 hover:text-slate-400 transition-colors"
                            >
                              <Copy className="w-3.5 h-3.5" />
                            </button>
                          </div>
                        )}
                      </div>

                      {msg.role === 'user' && (
                        <div className="w-9 h-9 rounded-full bg-white/15 flex items-center justify-center shrink-0">
                          <User className="w-4 h-4 text-white" />
                        </div>
                      )}
                    </motion.div>
                  ))}

                  {/* Typing indicator */}
                  {chatSteps > 0 && chatSteps < MOCK_FLOW.length && (
                    <motion.div
                      initial={{ opacity: 0, y: 8 }}
                      animate={{ opacity: 1, y: 0 }}
                      className="flex gap-4"
                    >
                      <div className="w-9 h-9 rounded-full bg-gradient-to-br from-cyan-500 to-blue-600 flex items-center justify-center shrink-0">
                        <Shield className="w-4 h-4 text-white" />
                      </div>
                      <div className="bg-background/80 border border-white/10 rounded-2xl rounded-tl-sm px-5 py-4 flex gap-1.5 items-center">
                        <div className="w-2 h-2 rounded-full bg-cyan-500 animate-bounce" style={{ animationDelay: '0ms' }} />
                        <div className="w-2 h-2 rounded-full bg-cyan-500 animate-bounce" style={{ animationDelay: '150ms' }} />
                        <div className="w-2 h-2 rounded-full bg-cyan-500 animate-bounce" style={{ animationDelay: '300ms' }} />
                      </div>
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>
            )}
          </div>

          {/* Bottom input bar */}
          <div className="shrink-0 border-t border-white/10 p-4 bg-background/30 backdrop-blur-sm">
            <div className="max-w-3xl mx-auto">
              <div className="flex items-center gap-3 glass-card rounded-xl border border-white/10 px-4 py-3">
                <input
                  type="text"
                  placeholder="Ask VERITAS anything about your content..."
                  readOnly
                  data-testid="console-chat-input"
                  className="flex-1 bg-transparent text-sm text-slate-400 placeholder-slate-600 outline-none cursor-default"
                />
                <button
                  data-testid="console-send-button"
                  className="w-8 h-8 rounded-lg bg-cyan-500/10 border border-cyan-500/30 flex items-center justify-center text-cyan-400 hover:bg-cyan-500/20 transition-colors"
                >
                  <ArrowUp className="w-4 h-4" />
                </button>
              </div>
              <p className="text-center text-xs text-slate-600 mt-2">
                VERITAS analyses content only. Results are informational and not legal evidence.
              </p>
            </div>
          </div>
        </motion.div>
      </div>
    </motion.div>
  );
}

function ConsoleTypewriterText({ text }: { text: string }) {
  const { displayedText } = useTypingEffect(text, 18);
  return <p className="text-sm md:text-base text-slate-200 leading-relaxed">{displayedText}</p>;
}
