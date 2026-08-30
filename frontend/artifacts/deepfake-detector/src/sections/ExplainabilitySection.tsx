import { motion } from 'framer-motion';

export function ExplainabilitySection() {
  return (
    <section id="explainability" className="py-32 relative overflow-hidden">
      <div className="max-w-6xl mx-auto px-6">
        <motion.div
          className="text-center mb-24"
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
        >
          <h2 className="text-3xl md:text-5xl font-heading font-bold mb-4">How Conclusions Are Reached</h2>
          <p className="text-slate-400 max-w-2xl mx-auto">
            VERITAS doesn&apos;t just give you an answer. It shows its reasoning — so you can understand why it reached that conclusion.
          </p>
        </motion.div>

        <div className="relative max-w-4xl mx-auto hidden md:block h-[500px]">
          <svg className="absolute inset-0 w-full h-full pointer-events-none z-0" style={{ overflow: 'visible' }}>
            <motion.path d="M 200 100 C 400 100, 400 250, 500 250" fill="none" stroke="rgba(6,182,212,0.3)" strokeWidth="2" strokeDasharray="5 5"
              initial={{ pathLength: 0 }} whileInView={{ pathLength: 1 }} viewport={{ once: true }} transition={{ duration: 1.5 }} />
            <motion.path d="M 200 200 C 400 200, 400 250, 500 250" fill="none" stroke="rgba(59,130,246,0.3)" strokeWidth="2" strokeDasharray="5 5"
              initial={{ pathLength: 0 }} whileInView={{ pathLength: 1 }} viewport={{ once: true }} transition={{ duration: 1.5, delay: 0.2 }} />
            <motion.path d="M 200 300 C 400 300, 400 250, 500 250" fill="none" stroke="rgba(16,185,129,0.3)" strokeWidth="2" strokeDasharray="5 5"
              initial={{ pathLength: 0 }} whileInView={{ pathLength: 1 }} viewport={{ once: true }} transition={{ duration: 1.5, delay: 0.4 }} />
            <motion.path d="M 200 400 C 400 400, 400 250, 500 250" fill="none" stroke="rgba(168,85,247,0.3)" strokeWidth="2" strokeDasharray="5 5"
              initial={{ pathLength: 0 }} whileInView={{ pathLength: 1 }} viewport={{ once: true }} transition={{ duration: 1.5, delay: 0.6 }} />
            <motion.path d="M 680 250 L 780 250" fill="none" stroke="rgba(255,255,255,0.3)" strokeWidth="2" strokeDasharray="5 5"
              initial={{ pathLength: 0 }} whileInView={{ pathLength: 1 }} viewport={{ once: true }} transition={{ duration: 1, delay: 1 }} />
          </svg>

          <div className="absolute left-0 top-[60px] w-48 bg-white/5 border border-cyan-500/30 p-4 rounded-xl backdrop-blur z-10 shadow-[0_0_20px_rgba(6,182,212,0.1)]">
            <h4 className="text-cyan-400 font-bold font-heading text-sm mb-1">Visual Signals</h4>
            <div className="h-1 w-full bg-cyan-500/20 rounded mt-2"><div className="h-full bg-cyan-400 w-[85%] rounded" /></div>
          </div>

          <div className="absolute left-0 top-[160px] w-48 bg-white/5 border border-blue-500/30 p-4 rounded-xl backdrop-blur z-10 shadow-[0_0_20px_rgba(59,130,246,0.1)]">
            <h4 className="text-blue-400 font-bold font-heading text-sm mb-1">Motion Signals</h4>
            <div className="h-1 w-full bg-blue-500/20 rounded mt-2"><div className="h-full bg-blue-400 w-[92%] rounded" /></div>
          </div>

          <div className="absolute left-0 top-[260px] w-48 bg-white/5 border border-emerald-500/30 p-4 rounded-xl backdrop-blur z-10 shadow-[0_0_20px_rgba(16,185,129,0.1)]">
            <h4 className="text-emerald-400 font-bold font-heading text-sm mb-1">Audio Signals</h4>
            <div className="h-1 w-full bg-emerald-500/20 rounded mt-2"><div className="h-full bg-emerald-400 w-[78%] rounded" /></div>
          </div>

          <div className="absolute left-0 top-[360px] w-48 bg-white/5 border border-purple-500/30 p-4 rounded-xl backdrop-blur z-10 shadow-[0_0_20px_rgba(168,85,247,0.1)]">
            <h4 className="text-purple-400 font-bold font-heading text-sm mb-1">Context Signals</h4>
            <div className="h-1 w-full bg-purple-500/20 rounded mt-2"><div className="h-full bg-purple-400 w-[83%] rounded" /></div>
          </div>

          <motion.div
            className="absolute left-[450px] top-[200px] w-56 h-[100px] bg-white/5 border border-white/20 p-4 rounded-xl backdrop-blur flex flex-col justify-center items-center z-10"
            initial={{ scale: 0.8, opacity: 0 }}
            whileInView={{ scale: 1, opacity: 1 }}
            viewport={{ once: true }}
            transition={{ delay: 1, duration: 0.5 }}
          >
            <h4 className="text-white font-bold font-heading text-lg text-center">Combined Evidence</h4>
            <p className="text-xs text-slate-400 text-center mt-1">All signals weighed together</p>
          </motion.div>

          <motion.div
            className="absolute left-[780px] top-[180px] w-64 bg-gradient-to-br from-cyan-500/10 to-purple-500/10 border border-cyan-500/50 p-6 rounded-2xl backdrop-blur z-10 shadow-[0_0_30px_rgba(6,182,212,0.2)]"
            initial={{ x: 20, opacity: 0 }}
            whileInView={{ x: 0, opacity: 1 }}
            viewport={{ once: true }}
            transition={{ delay: 1.5, duration: 0.5 }}
          >
            <h4 className="text-white font-bold font-heading text-xl mb-3 text-gradient">VERITAS Verdict</h4>
            <p className="text-sm text-slate-300 leading-relaxed">
              &ldquo;After examining visual, motion, audio, and contextual signals, VERITAS concluded this content shows multiple signs of manipulation. Confidence: High.&rdquo;
            </p>
          </motion.div>
        </div>

        {/* Mobile Fallback */}
        <div className="md:hidden flex flex-col gap-6">
          <div className="glass-card p-6 rounded-xl border border-cyan-500/30">
            <h4 className="text-cyan-400 font-bold font-heading text-sm mb-2">Step 1: Four Systems Check Your Content</h4>
            <p className="text-sm text-slate-400">Each system independently examines a different aspect of your upload.</p>
          </div>
          <div className="glass-card p-6 rounded-xl border border-white/20">
            <h4 className="text-white font-bold font-heading text-sm mb-2">Step 2: Findings Are Combined</h4>
            <p className="text-sm text-slate-400">All signals are weighed together to form a single, coherent picture.</p>
          </div>
          <div className="glass-card p-6 rounded-xl border border-purple-500/50">
            <h4 className="text-purple-400 font-bold font-heading text-sm mb-2">Step 3: A Clear Verdict Is Delivered</h4>
            <p className="text-sm text-slate-400">VERITAS explains what it found in plain language you can act on.</p>
          </div>
        </div>
      </div>
    </section>
  );
}
