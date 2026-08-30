import { motion } from 'framer-motion';
import { Eye, Layers, Globe, MessageSquareText, ShieldAlert } from 'lucide-react';

const features = [
  {
    icon: <Eye className="w-6 h-6 text-cyan-400" />,
    title: "Detect AI-Generated Media",
    description: "Whether it's a deepfake video, a cloned voice, or a synthetic image — VERITAS is built to spot it.",
    delay: 0.1
  },
  {
    icon: <Layers className="w-6 h-6 text-blue-400" />,
    title: "Analyse Any Format",
    description: "Images, videos, audio recordings, and social media links — VERITAS handles them all in one place.",
    delay: 0.2
  },
  {
    icon: <Globe className="w-6 h-6 text-emerald-400" />,
    title: "Verify Online Context",
    description: "We cross-reference your content with trusted online sources to check whether the story adds up.",
    delay: 0.3
  },
  {
    icon: <MessageSquareText className="w-6 h-6 text-purple-400" />,
    title: "Receive Clear Explanations",
    description: "No confusing scores or technical terms. VERITAS tells you what it found in plain, honest language.",
    delay: 0.4
  }
];

export function WhyVeritasSection() {
  return (
    <section id="why" className="py-32 relative bg-background">
      <div className="max-w-6xl mx-auto px-6">
        <motion.div
          className="text-center mb-16"
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
        >
          <h2 className="text-3xl md:text-5xl font-heading font-bold mb-4">Why VERITAS?</h2>
          <p className="text-slate-400 max-w-2xl mx-auto">
            Misinformation spreads fast. VERITAS helps you stay one step ahead.
          </p>
        </motion.div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-20">
          {features.map((feature, i) => (
            <motion.div
              key={i}
              className="glass-card p-6 rounded-2xl group hover:-translate-y-2 transition-transform duration-300 relative overflow-hidden"
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: feature.delay, duration: 0.6 }}
            >
              <div className="absolute inset-0 bg-gradient-to-br from-white/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300" />
              <div className="flex items-start gap-4">
                <div className="w-12 h-12 rounded-xl bg-white/5 flex items-center justify-center shrink-0 group-hover:scale-110 transition-transform duration-300">
                  {feature.icon}
                </div>
                <div>
                  <h3 className="text-xl font-semibold mb-2 text-white">{feature.title}</h3>
                  <p className="text-slate-400 text-sm leading-relaxed">{feature.description}</p>
                </div>
              </div>
            </motion.div>
          ))}
        </div>

        {/* Example Report Card */}
        <motion.div
          className="text-center mb-10"
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
        >
          <h3 className="text-2xl md:text-3xl font-heading font-bold mb-2 text-white">An Example Report</h3>
          <p className="text-slate-400">This is the kind of clear verdict VERITAS delivers.</p>
        </motion.div>

        <motion.div
          className="max-w-2xl mx-auto relative"
          initial={{ opacity: 0, scale: 0.97 }}
          whileInView={{ opacity: 1, scale: 1 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6 }}
        >
          {/* Gradient border wrapper */}
          <div className="absolute -inset-px rounded-2xl bg-gradient-to-r from-cyan-500/50 via-blue-500/30 to-purple-500/50 blur-sm" />
          <div className="relative glass-card rounded-2xl p-8 border border-white/10">
            <div className="flex items-center gap-3 mb-6">
              <div className="w-10 h-10 rounded-xl bg-red-500/20 flex items-center justify-center">
                <ShieldAlert className="w-5 h-5 text-red-400" />
              </div>
              <div>
                <p className="text-xs text-slate-500 uppercase tracking-widest font-medium">Verdict</p>
                <p className="text-xl font-heading font-bold text-red-400">Likely AI Generated</p>
              </div>
            </div>

            <p className="text-slate-300 leading-relaxed text-sm mb-6">
              VERITAS analysed multiple aspects of the uploaded content before reaching this conclusion. Several visual, motion, audio, and contextual inconsistencies were identified, suggesting the media has likely been manipulated. Review the explanation below for a full breakdown.
            </p>

            <div className="flex flex-wrap gap-2">
              {[
                { label: 'Visual Analysis', color: 'cyan' },
                { label: 'Motion Analysis', color: 'blue' },
                { label: 'Audio Analysis', color: 'emerald' },
                { label: 'Context Verification', color: 'purple' },
              ].map((badge) => (
                <span
                  key={badge.label}
                  className={`px-3 py-1 rounded-full text-xs font-medium bg-${badge.color}-500/10 text-${badge.color}-400 border border-${badge.color}-500/20`}
                >
                  {badge.label} ✓
                </span>
              ))}
            </div>
          </div>
        </motion.div>
      </div>
    </section>
  );
}
