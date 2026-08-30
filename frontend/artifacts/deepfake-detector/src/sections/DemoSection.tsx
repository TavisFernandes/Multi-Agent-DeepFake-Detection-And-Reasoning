import { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { UploadCloud, Image as ImageIcon, Video, Mic, Link as LinkIcon, AlertTriangle, CheckCircle2, Copy, User, Shield } from 'lucide-react';
import { useTypingEffect } from '@/hooks/useTypingEffect';
import { useLocation } from 'wouter';

const MOCK_FLOW = [
  { role: 'user', text: "I've uploaded a video. Can you check whether it's authentic?" },
  { role: 'system', text: "I've received your content. I'm now examining the visuals, movement, audio, and available context. This usually takes a few moments." },
  { role: 'agent', agent: 'Visual Analysis', color: 'cyan', text: 'Several visual inconsistencies were detected around the face — areas that appear edited or artificially generated.' },
  { role: 'agent', agent: 'Motion Analysis', color: 'blue', text: 'Unusual movement patterns were identified between frames. The motion does not behave as naturally recorded video normally would.' },
  { role: 'agent', agent: 'Audio Analysis', color: 'emerald', text: 'The audio shows characteristics commonly found in synthesised speech. Some frequencies appear inconsistent with natural recording conditions.' },
  { role: 'agent', agent: 'Context Verification', color: 'purple', text: 'Limited online context was found for this content. No matching source could be verified.' },
  { role: 'summary', text: 'Based on the available evidence, this content shows several indicators commonly associated with AI-generated media. Visual inconsistencies, unusual movement, and audio irregularities collectively suggest the content is likely manipulated. VERITAS has high confidence that this media has been artificially generated.' }
];

export function DemoSection() {
  const [activeTab, setActiveTab] = useState('video');
  const [isUploading, setIsUploading] = useState(false);
  const [hasFile, setHasFile] = useState(false);
  const [chatSteps, setChatSteps] = useState<number>(0);
  const scrollRef = useRef<HTMLDivElement>(null);
  const [, setLocation] = useLocation();

  const startAnalysis = () => {
    if (!hasFile) return;
    setChatSteps(1);
    let step = 1;
    const interval = setInterval(() => {
      step++;
      setChatSteps(step);
      if (step >= MOCK_FLOW.length) clearInterval(interval);
    }, 2000);
  };

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [chatSteps]);

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsUploading(true);
    setTimeout(() => { setIsUploading(false); setHasFile(true); }, 1000);
  };

  const visibleMessages = MOCK_FLOW.slice(0, chatSteps);

  return (
    <section id="demo" className="py-32 relative">
      <div className="max-w-7xl mx-auto px-6">
        <motion.div
          className="text-center mb-16"
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
        >
          <h2 className="text-3xl md:text-5xl font-heading font-bold mb-4">See VERITAS in Action</h2>
          <p className="text-slate-400 max-w-2xl mx-auto">
            Watch how VERITAS analyses a sample piece of content. Click a content type, then hit Analyse.
          </p>
        </motion.div>

        <div className="flex flex-col lg:flex-row gap-8 h-[700px]">
          {/* Left Panel */}
          <motion.div
            className="w-full lg:w-1/3 flex flex-col gap-6"
            initial={{ opacity: 0, x: -20 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true }}
          >
            <div className="flex bg-white/5 rounded-lg p-1 gap-1">
              {[
                { id: 'image', icon: ImageIcon, label: 'Image' },
                { id: 'video', icon: Video, label: 'Video' },
                { id: 'audio', icon: Mic, label: 'Audio' },
                { id: 'url', icon: LinkIcon, label: 'URL' }
              ].map(tab => (
                <button
                  key={tab.id}
                  onClick={() => { setActiveTab(tab.id); setHasFile(false); setChatSteps(0); }}
                  data-testid={`tab-${tab.id}`}
                  className={`flex-1 flex items-center justify-center gap-2 py-2 px-3 text-sm font-medium rounded-md transition-colors ${
                    activeTab === tab.id ? 'bg-white/10 text-white' : 'text-slate-400 hover:text-white hover:bg-white/5'
                  }`}
                >
                  <tab.icon className="w-4 h-4" />
                  <span className="hidden sm:inline">{tab.label}</span>
                </button>
              ))}
            </div>

            <div
              onDragOver={(e) => { e.preventDefault(); e.stopPropagation(); }}
              onDrop={handleDrop}
              data-testid="dropzone"
              className={`flex-1 border-2 border-dashed rounded-2xl flex flex-col items-center justify-center p-8 transition-colors group cursor-pointer ${
                hasFile ? 'border-cyan-500/50 bg-cyan-500/5' : 'border-white/10 hover:border-cyan-500/50 bg-white/5 hover:bg-white/10'
              }`}
              onClick={() => {
                if (!hasFile) {
                  setIsUploading(true);
                  setTimeout(() => { setIsUploading(false); setHasFile(true); }, 1000);
                }
              }}
            >
              {isUploading ? (
                <div className="flex flex-col items-center gap-4">
                  <div className="w-12 h-12 rounded-full border-t-2 border-cyan-500 animate-spin" />
                  <p className="text-slate-400">Loading content...</p>
                </div>
              ) : hasFile ? (
                <div className="flex flex-col items-center gap-4">
                  <div className="w-16 h-16 rounded-xl bg-cyan-500/20 flex items-center justify-center text-cyan-400">
                    <CheckCircle2 className="w-8 h-8" />
                  </div>
                  <p className="text-white font-medium">Content loaded successfully.</p>
                  <p className="text-sm text-slate-400 text-center">sample_video_042.mp4<br />(14.2 MB)</p>
                </div>
              ) : (
                <div className="flex flex-col items-center gap-4 text-slate-400 group-hover:text-white transition-colors">
                  <UploadCloud className="w-12 h-12" />
                  <div className="text-center">
                    <p className="font-medium">Drag &amp; drop or click to upload</p>
                    <p className="text-sm mt-1 opacity-70">Supports MP4, MOV, WAV, JPG, PNG (Max 50MB)</p>
                  </div>
                </div>
              )}
            </div>

            <button
              disabled={!hasFile || chatSteps > 0}
              onClick={startAnalysis}
              data-testid="button-analyse"
              className={`w-full py-4 rounded-xl font-heading font-bold text-lg transition-all ${
                hasFile && chatSteps === 0
                  ? 'bg-gradient-to-r from-cyan-500 to-blue-600 text-white shadow-[0_0_20px_rgba(6,182,212,0.3)] hover:shadow-[0_0_30px_rgba(6,182,212,0.5)]'
                  : 'bg-white/5 text-slate-500 cursor-not-allowed'
              }`}
            >
              {chatSteps > 0 ? 'Analysing...' : 'Analyse with VERITAS'}
            </button>
          </motion.div>

          {/* Right Panel */}
          <motion.div
            className="w-full lg:w-2/3 glass-card rounded-2xl flex flex-col overflow-hidden border-white/10"
            initial={{ opacity: 0, x: 20 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true }}
          >
            <div className="border-b border-white/10 px-6 py-4 flex items-center gap-3 bg-background/50 backdrop-blur-md z-10">
              <div className="w-3 h-3 rounded-full bg-cyan-500 shadow-[0_0_10px_rgba(6,182,212,0.8)] animate-pulse" />
              <span className="font-heading font-semibold text-white">VERITAS</span>
            </div>

            <div className="flex-1 overflow-y-auto p-6 flex flex-col gap-6" ref={scrollRef}>
              {chatSteps === 0 && (
                <div className="flex-1 flex items-center justify-center text-slate-500 text-sm">
                  VERITAS is ready. Select a content type and click Analyse.
                </div>
              )}

              <AnimatePresence>
                {visibleMessages.map((msg, i) => (
                  <motion.div
                    key={i}
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    className={`flex gap-4 ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
                  >
                    {msg.role !== 'user' && (
                      <div className="w-8 h-8 rounded-full bg-gradient-to-br from-cyan-500 to-blue-600 flex items-center justify-center shrink-0">
                        <Shield className="w-4 h-4 text-white" />
                      </div>
                    )}

                    <div className={`max-w-[85%] rounded-2xl p-4 ${
                      msg.role === 'user' ? 'bg-white/10 text-white' :
                      msg.role === 'agent' ? 'bg-background border border-white/10' :
                      msg.role === 'summary' ? 'bg-red-500/10 border border-red-500/20' :
                      'bg-transparent text-slate-400 text-sm'
                    }`}>
                      {msg.role === 'agent' && (
                        <div className="flex items-center gap-2 mb-2">
                          <span className={`px-2 py-0.5 rounded text-xs font-bold uppercase tracking-wider bg-${msg.color}-500/20 text-${msg.color}-400`}>
                            {msg.agent}
                          </span>
                        </div>
                      )}

                      {msg.role === 'summary' ? (
                        <div>
                          <div className="flex items-center gap-2 mb-3 text-red-400 font-heading font-bold">
                            <AlertTriangle className="w-5 h-5" />
                            Likely AI Generated — High Confidence
                          </div>
                          <TypewriterText text={msg.text} />
                        </div>
                      ) : (
                        <div className="text-sm md:text-base text-slate-200 leading-relaxed">{msg.text}</div>
                      )}

                      {msg.role !== 'user' && msg.role !== 'system' && (
                        <div className="mt-3 flex items-center gap-2">
                          <button className="text-slate-500 hover:text-white transition-colors" data-testid="button-copy">
                            <Copy className="w-4 h-4" />
                          </button>
                        </div>
                      )}
                    </div>

                    {msg.role === 'user' && (
                      <div className="w-8 h-8 rounded-full bg-white/20 flex items-center justify-center shrink-0">
                        <User className="w-4 h-4 text-white" />
                      </div>
                    )}
                  </motion.div>
                ))}

                {chatSteps > 0 && chatSteps < MOCK_FLOW.length && (
                  <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="flex gap-4">
                    <div className="w-8 h-8 rounded-full bg-gradient-to-br from-cyan-500 to-blue-600 flex items-center justify-center shrink-0">
                      <Shield className="w-4 h-4 text-white" />
                    </div>
                    <div className="bg-transparent p-2 flex gap-1 items-center">
                      <div className="w-2 h-2 rounded-full bg-cyan-500 animate-bounce" style={{ animationDelay: '0ms' }} />
                      <div className="w-2 h-2 rounded-full bg-cyan-500 animate-bounce" style={{ animationDelay: '150ms' }} />
                      <div className="w-2 h-2 rounded-full bg-cyan-500 animate-bounce" style={{ animationDelay: '300ms' }} />
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>

            <div className="border-t border-white/5 px-6 py-3 flex items-center justify-between">
              <span className="text-xs text-slate-500">This is a demonstration with sample data.</span>
              <button
                onClick={() => setLocation('/console')}
                data-testid="button-try-own-content"
                className="text-xs text-cyan-400 hover:text-cyan-300 transition-colors"
              >
                Try with your own content →
              </button>
            </div>
          </motion.div>
        </div>
      </div>
    </section>
  );
}

function TypewriterText({ text }: { text: string }) {
  const { displayedText } = useTypingEffect(text, 20);
  return <div className="text-sm md:text-base text-slate-200 leading-relaxed">{displayedText}</div>;
}
