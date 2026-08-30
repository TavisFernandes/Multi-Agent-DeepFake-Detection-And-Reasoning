import { useState } from 'react';
import { motion } from 'framer-motion';
import { Layers, Zap, MessageSquareText, Target } from 'lucide-react';
import { CardModal } from '@/components/ui/CardModal';
import { featureModalData } from '@/data/featureModalData';

const features = [
  {
    icon: <Layers className="w-6 h-6 text-cyan-400" />,
    title: "Multiple Sources of Evidence",
    description: "VERITAS checks images, videos, audio, and online sources simultaneously — so nothing slips through.",
    delay: 0.1,
  },
  {
    icon: <Zap className="w-6 h-6 text-blue-400" />,
    title: "Everything Considered Together",
    description: "Rather than giving you four separate results, VERITAS combines all findings into one clear conclusion.",
    delay: 0.2,
  },
  {
    icon: <MessageSquareText className="w-6 h-6 text-purple-400" />,
    title: "Easy-to-Understand Explanations",
    description: "No jargon. No percentages. VERITAS explains its findings the way a trusted friend would.",
    delay: 0.3,
  },
  {
    icon: <Target className="w-6 h-6 text-emerald-400" />,
    title: "Honest About Uncertainty",
    description: "When VERITAS isn't sure, it says so. Results are expressed as High, Moderate, or Low confidence.",
    delay: 0.4,
  },
];

const hintColors = ['text-cyan-400', 'text-blue-400', 'text-purple-400', 'text-emerald-400'];

export function AboutSection() {
  const [openIndex, setOpenIndex] = useState<number | null>(null);

  return (
    <section id="about" className="py-32 relative">
      <div className="max-w-6xl mx-auto px-6">
        <motion.div
          className="text-center mb-16"
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
        >
          <h2 className="text-3xl md:text-5xl font-heading font-bold mb-4">How VERITAS Works</h2>
          <p className="text-slate-400 max-w-2xl mx-auto">
            VERITAS combines multiple sources of evidence to reach a conclusion. Rather than relying on a single check, it looks at visuals, movement, audio, and online context together.
          </p>
        </motion.div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          {features.map((feature, i) => (
            <motion.button
              key={i}
              className="glass-card p-6 rounded-2xl group hover:-translate-y-2 transition-transform duration-300 relative overflow-hidden text-left cursor-pointer focus:outline-none focus-visible:ring-2 focus-visible:ring-white/30"
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: feature.delay, duration: 0.6 }}
              onClick={() => setOpenIndex(i)}
              whileTap={{ scale: 0.98 }}
            >
              <div className="absolute inset-0 bg-gradient-to-br from-white/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300" />
              <div className="w-12 h-12 rounded-xl bg-white/5 flex items-center justify-center mb-6 group-hover:scale-110 transition-transform duration-300">
                {feature.icon}
              </div>
              <h3 className="text-xl font-semibold mb-3 text-white">{feature.title}</h3>
              <p className="text-slate-400 text-sm leading-relaxed mb-4">{feature.description}</p>
              <span className={`text-xs font-medium ${hintColors[i]} opacity-0 group-hover:opacity-100 transition-opacity duration-200 flex items-center gap-1`}>
                Learn more →
              </span>
            </motion.button>
          ))}
        </div>
      </div>

      <CardModal
        isOpen={openIndex !== null}
        onClose={() => setOpenIndex(null)}
        cards={featureModalData}
        currentIndex={openIndex ?? 0}
        onNavigate={setOpenIndex}
      />
    </section>
  );
}
