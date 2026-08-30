import { useState } from 'react';
import { motion, useMotionTemplate, useMotionValue } from 'framer-motion';
import { MouseEvent } from 'react';
import { CardModal } from '@/components/ui/CardModal';
import { agentModalData } from '@/data/agentModalData';

const agents = [
  {
    id: "visual",
    name: "Visual Analysis",
    color: "cyan",
    accent: "cyan-400",
    bg: "from-cyan-500/20 to-transparent",
    description: "We inspect images and video frames closely, looking for signs of editing, blending, or artificial generation that the human eye might miss.",
    visual: () => (
      <div className="absolute inset-0 overflow-hidden opacity-50 flex items-center justify-center">
        <div className="grid grid-cols-4 grid-rows-4 gap-1 w-full h-full p-8">
          {Array.from({ length: 16 }).map((_, i) => (
            <motion.div
              key={i}
              className="bg-cyan-500/20 rounded-sm"
              animate={{ opacity: [0.2, 0.8, 0.2] }}
              transition={{ duration: 1.5 + (i % 4) * 0.3, repeat: Infinity, delay: (i % 5) * 0.2 }}
            />
          ))}
        </div>
      </div>
    )
  },
  {
    id: "motion",
    name: "Motion Analysis",
    color: "blue",
    accent: "blue-400",
    bg: "from-blue-500/20 to-transparent",
    description: "We look for unnatural movements and inconsistencies across video frames — the kind of subtle flickering that reveals a manipulated clip.",
    visual: () => (
      <div className="absolute inset-0 overflow-hidden opacity-50 flex flex-col items-center justify-center gap-4 p-8">
        <div className="w-full h-1 bg-white/10 relative rounded-full overflow-hidden">
          <motion.div
            className="absolute top-0 left-0 bottom-0 w-1/4 bg-blue-500/50"
            animate={{ x: ['-100%', '400%'] }}
            transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
          />
        </div>
        <div className="flex gap-2 w-full justify-center">
          {Array.from({ length: 5 }).map((_, i) => (
            <motion.div
              key={i}
              className="w-12 h-16 border border-blue-500/30 rounded"
              animate={{ skewX: [0, 10, -10, 0] }}
              transition={{ duration: 3, repeat: Infinity, delay: i * 0.2 }}
            />
          ))}
        </div>
      </div>
    )
  },
  {
    id: "audio",
    name: "Audio Analysis",
    color: "emerald",
    accent: "emerald-400",
    bg: "from-emerald-500/20 to-transparent",
    description: "We listen carefully to speech and background sounds, checking for signs that the audio has been synthesised or altered.",
    visual: () => (
      <div className="absolute inset-0 overflow-hidden opacity-50 flex items-center justify-center gap-1 p-8">
        {Array.from({ length: 20 }).map((_, i) => (
          <motion.div
            key={i}
            className="w-2 bg-emerald-500/40 rounded-full"
            animate={{ height: ['20%', '80%', '20%'] }}
            transition={{ duration: 0.6 + (i % 5) * 0.1, repeat: Infinity, delay: (i % 4) * 0.15 }}
          />
        ))}
      </div>
    )
  },
  {
    id: "context",
    name: "Context Verification",
    color: "purple",
    accent: "purple-400",
    bg: "from-purple-500/20 to-transparent",
    description: "We compare available information with trusted online sources to check whether the content matches what it claims to be.",
    visual: () => {
      const nodes = [
        { cx: 20, cy: 30, delay: 0 }, { cx: 50, cy: 15, delay: 0.3 },
        { cx: 80, cy: 25, delay: 0.6 }, { cx: 65, cy: 55, delay: 0.9 },
        { cx: 30, cy: 65, delay: 1.2 }, { cx: 50, cy: 80, delay: 1.5 },
        { cx: 75, cy: 75, delay: 1.8 }, { cx: 10, cy: 50, delay: 2.1 },
      ];
      return (
        <div className="absolute inset-0 overflow-hidden opacity-50">
          <svg className="w-full h-full" viewBox="0 0 100 100" preserveAspectRatio="none">
            {nodes.map((n, i) => (
              <motion.circle
                key={`c-${i}`}
                cx={n.cx}
                cy={n.cy}
                initial={{ r: 2 }}
                fill="rgba(168, 85, 247, 0.5)"
                animate={{ r: [2, 4, 2] }}
                transition={{ duration: 2, repeat: Infinity, delay: n.delay }}
              />
            ))}
            <motion.path
              d="M 20 30 Q 40 50 60 20 T 80 70"
              fill="none"
              stroke="rgba(168, 85, 247, 0.3)"
              strokeWidth="0.5"
              initial={{ pathLength: 0 }}
              animate={{ pathLength: [0, 1] }}
              transition={{ duration: 2, repeat: Infinity }}
            />
          </svg>
        </div>
      );
    }
  }
];

function AgentCard({
  agent,
  index,
  onClick,
}: {
  agent: typeof agents[number];
  index: number;
  onClick: () => void;
}) {
  const mouseX = useMotionValue(0);
  const mouseY = useMotionValue(0);

  function handleMouseMove({ currentTarget, clientX, clientY }: MouseEvent) {
    const { left, top } = currentTarget.getBoundingClientRect();
    mouseX.set(clientX - left);
    mouseY.set(clientY - top);
  }

  return (
    <motion.div
      className="relative group rounded-3xl overflow-hidden glass-card border-white/10 cursor-pointer focus:outline-none focus-visible:ring-2 focus-visible:ring-white/30"
      initial={{ opacity: 0, scale: 0.95 }}
      whileInView={{ opacity: 1, scale: 1 }}
      viewport={{ once: true }}
      transition={{ delay: index * 0.1, duration: 0.5 }}
      onMouseMove={handleMouseMove}
      style={{ transformStyle: "preserve-3d" }}
      whileHover={{ scale: 1.02, rotateX: 2, rotateY: -2 }}
      whileTap={{ scale: 0.99 }}
      onClick={onClick}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => e.key === 'Enter' && onClick()}
    >
      <motion.div
        className="pointer-events-none absolute -inset-px rounded-3xl opacity-0 transition duration-300 group-hover:opacity-100"
        style={{
          background: useMotionTemplate`
            radial-gradient(
              650px circle at ${mouseX}px ${mouseY}px,
              rgba(255,255,255,0.1),
              transparent 80%
            )
          `,
        }}
      />
      <div className={`absolute inset-0 bg-gradient-to-br ${agent.bg} mix-blend-overlay`} />

      <div className="h-48 relative border-b border-white/10 bg-background/50">
        <agent.visual />
      </div>

      <div className="p-8 relative z-10">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-3">
            <div className={`w-2 h-2 rounded-full bg-${agent.color}-400 animate-pulse`} />
            <h3 className="text-2xl font-heading font-bold text-white">{agent.name}</h3>
          </div>
          <span className={`text-xs text-${agent.color}-400 opacity-0 group-hover:opacity-100 transition-opacity duration-200 font-medium`}>
            Learn more →
          </span>
        </div>
        <p className="text-slate-400 leading-relaxed">{agent.description}</p>
      </div>
    </motion.div>
  );
}

export function AgentShowcase() {
  const [openIndex, setOpenIndex] = useState<number | null>(null);

  return (
    <section id="agents" className="py-32 relative">
      <div className="max-w-7xl mx-auto px-6">
        <motion.div
          className="text-center mb-16"
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
        >
          <h2 className="text-3xl md:text-5xl font-heading font-bold mb-4">Meet Your Analysts</h2>
          <p className="text-slate-400 max-w-2xl mx-auto">
            VERITAS uses four specialised systems that each examine a different aspect of your content. Together, they build a complete picture.
          </p>
        </motion.div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          {agents.map((agent, i) => (
            <AgentCard
              key={agent.id}
              agent={agent}
              index={i}
              onClick={() => setOpenIndex(i)}
            />
          ))}
        </div>
      </div>

      <CardModal
        isOpen={openIndex !== null}
        onClose={() => setOpenIndex(null)}
        cards={agentModalData}
        currentIndex={openIndex ?? 0}
        onNavigate={setOpenIndex}
      />
    </section>
  );
}
