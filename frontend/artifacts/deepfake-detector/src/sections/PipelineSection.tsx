import { motion } from 'framer-motion';
import { Upload, Cpu, GitMerge, BrainCircuit, ShieldAlert } from 'lucide-react';

const stages = [
  { id: 1, icon: Upload, title: "You upload content", desc: "Share an image, video, audio clip, or paste a social media link." },
  { id: 2, icon: Cpu, title: "Four systems get to work", desc: "Visual, motion, audio, and context checks all happen simultaneously." },
  { id: 3, icon: GitMerge, title: "Findings are combined", desc: "All results are brought together to form a complete picture." },
  { id: 4, icon: BrainCircuit, title: "A clear explanation is written", desc: "VERITAS turns the findings into plain language you can understand." },
  { id: 5, icon: ShieldAlert, title: "You receive a verdict", desc: "Authentic, Likely Manipulated, or Uncertain — with a clear explanation." }
];

export function PipelineSection() {
  return (
    <section id="pipeline" className="py-32 relative overflow-hidden bg-white/[0.02]">
      <div className="max-w-7xl mx-auto px-6">
        <motion.div
          className="text-center mb-24"
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
        >
          <h2 className="text-3xl md:text-5xl font-heading font-bold mb-4">How We Analyse Your Content</h2>
          <p className="text-slate-400 max-w-2xl mx-auto">
            VERITAS combines insights from multiple specialised systems before reaching a final conclusion.
          </p>
        </motion.div>

        <div className="relative">
          {/* Connecting Line Desktop */}
          <div className="hidden lg:block absolute top-12 left-[10%] right-[10%] h-0.5 bg-white/10 z-0">
            <motion.div
              className="absolute inset-0 bg-gradient-to-r from-cyan-400 via-blue-500 to-purple-500 origin-left"
              initial={{ scaleX: 0 }}
              whileInView={{ scaleX: 1 }}
              viewport={{ once: true }}
              transition={{ duration: 1.5, ease: "easeInOut" }}
            />
          </div>

          {/* Connecting Line Mobile */}
          <div className="lg:hidden absolute top-[10%] bottom-[10%] left-12 w-0.5 bg-white/10 z-0">
            <motion.div
              className="absolute inset-0 bg-gradient-to-b from-cyan-400 via-blue-500 to-purple-500 origin-top"
              initial={{ scaleY: 0 }}
              whileInView={{ scaleY: 1 }}
              viewport={{ once: true }}
              transition={{ duration: 1.5, ease: "easeInOut" }}
            />
          </div>

          <div className="flex flex-col lg:flex-row justify-between relative z-10 gap-12 lg:gap-4">
            {stages.map((stage, i) => (
              <motion.div
                key={stage.id}
                className="flex lg:flex-col items-center gap-6 lg:gap-6 w-full lg:w-1/5"
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: i * 0.2 }}
              >
                <div className="w-24 h-24 shrink-0 rounded-2xl glass-card flex items-center justify-center relative group">
                  <div className="absolute inset-0 rounded-2xl bg-white/5 opacity-0 group-hover:opacity-100 transition-opacity" />
                  <stage.icon className="w-8 h-8 text-white group-hover:text-cyan-400 transition-colors" />
                  <div className="absolute -top-3 -right-3 w-8 h-8 rounded-full bg-slate-800 border-2 border-background flex items-center justify-center text-xs font-mono font-bold text-white">
                    0{stage.id}
                  </div>
                </div>

                <div className="lg:text-center flex-1">
                  <h3 className="text-lg font-heading font-semibold text-white mb-2">{stage.title}</h3>
                  <p className="text-sm text-slate-400 leading-relaxed">{stage.desc}</p>
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}
