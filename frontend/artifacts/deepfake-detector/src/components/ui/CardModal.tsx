import { motion, AnimatePresence } from 'framer-motion';
import { X, ChevronLeft, ChevronRight } from 'lucide-react';
import { useEffect, useRef, useState, ReactNode, MouseEvent } from 'react';

export interface CardModalSection {
  heading: string;
  items?: string[];
  body?: string;
}

export interface CardModalData {
  icon: ReactNode;
  accentColor: string; // tailwind color name: cyan | blue | purple | emerald
  gradientFrom: string; // e.g. "from-cyan-500/30"
  gradientTo: string;
  title: string;
  introduction: string;
  sections: CardModalSection[];
  privacyNote?: string;
  illustration: () => ReactNode;
}

interface CardModalProps {
  isOpen: boolean;
  onClose: () => void;
  cards: CardModalData[];
  currentIndex: number;
  onNavigate: (index: number) => void;
}

// Floating particles behind the icon
function FloatingParticles({ color }: { color: string }) {
  const colorMap: Record<string, string> = {
    cyan: 'bg-cyan-400',
    blue: 'bg-blue-400',
    purple: 'bg-purple-400',
    emerald: 'bg-emerald-400',
  };
  const cls = colorMap[color] ?? 'bg-white';

  return (
    <div className="absolute inset-0 overflow-hidden pointer-events-none">
      {Array.from({ length: 12 }).map((_, i) => (
        <motion.div
          key={i}
          className={`absolute rounded-full ${cls} opacity-30`}
          style={{
            width: 4 + (i % 4) * 2,
            height: 4 + (i % 4) * 2,
            left: `${10 + (i * 7.5) % 80}%`,
            top: `${5 + (i * 11) % 85}%`,
          }}
          animate={{
            y: [-8, 8, -8],
            opacity: [0.1, 0.4, 0.1],
            scale: [1, 1.4, 1],
          }}
          transition={{
            duration: 3 + (i % 4) * 0.7,
            repeat: Infinity,
            delay: (i * 0.25) % 2,
            ease: 'easeInOut',
          }}
        />
      ))}
    </div>
  );
}

export function CardModal({ isOpen, onClose, cards, currentIndex, onNavigate }: CardModalProps) {
  const card = cards[currentIndex];
  const containerRef = useRef<HTMLDivElement>(null);
  const [sheenPos, setSheenPos] = useState({ x: 0, y: 0 });

  // Keyboard navigation
  useEffect(() => {
    if (!isOpen) return;
    const handler = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
      if (e.key === 'ArrowLeft') onNavigate(currentIndex === 0 ? cards.length - 1 : currentIndex - 1);
      if (e.key === 'ArrowRight') onNavigate(currentIndex === cards.length - 1 ? 0 : currentIndex + 1);
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [isOpen, currentIndex, cards.length, onClose, onNavigate]);

  // Lock body scroll while open
  useEffect(() => {
    document.body.style.overflow = isOpen ? 'hidden' : '';
    return () => { document.body.style.overflow = ''; };
  }, [isOpen]);

  function handleMouseMove(e: MouseEvent<HTMLDivElement>) {
    const rect = containerRef.current?.getBoundingClientRect();
    if (!rect) return;
    setSheenPos({ x: e.clientX - rect.left, y: e.clientY - rect.top });
  }

  const accentBorder: Record<string, string> = {
    cyan: 'from-cyan-400/60 via-cyan-400/20 to-transparent',
    blue: 'from-blue-400/60 via-blue-400/20 to-transparent',
    purple: 'from-purple-400/60 via-purple-400/20 to-transparent',
    emerald: 'from-emerald-400/60 via-emerald-400/20 to-transparent',
  };

  const accentGlow: Record<string, string> = {
    cyan: 'shadow-cyan-500/20',
    blue: 'shadow-blue-500/20',
    purple: 'shadow-purple-500/20',
    emerald: 'shadow-emerald-500/20',
  };

  const accentText: Record<string, string> = {
    cyan: 'text-cyan-400',
    blue: 'text-blue-400',
    purple: 'text-purple-400',
    emerald: 'text-emerald-400',
  };

  const accentDivider: Record<string, string> = {
    cyan: 'from-transparent via-cyan-500/30 to-transparent',
    blue: 'from-transparent via-blue-500/30 to-transparent',
    purple: 'from-transparent via-purple-500/30 to-transparent',
    emerald: 'from-transparent via-emerald-500/30 to-transparent',
  };

  const accentBg: Record<string, string> = {
    cyan: 'bg-cyan-400/10',
    blue: 'bg-blue-400/10',
    purple: 'bg-purple-400/10',
    emerald: 'bg-emerald-400/10',
  };

  if (!card) return null;

  return (
    <AnimatePresence>
      {isOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
          {/* Backdrop */}
          <motion.div
            className="absolute inset-0 bg-black/75 backdrop-blur-sm"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.25 }}
            onClick={onClose}
          />

          {/* Modal */}
          <motion.div
            className="relative w-full max-w-2xl"
            initial={{ opacity: 0, scale: 0.95, y: 16 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.95, y: 16 }}
            transition={{ type: 'spring', damping: 28, stiffness: 320 }}
          >
            {/* Gradient border wrapper */}
            <div className={`rounded-3xl p-[1px] bg-gradient-to-br ${accentBorder[card.accentColor]}`}>
              <div
                ref={containerRef}
                onMouseMove={handleMouseMove}
                className={`relative rounded-3xl overflow-hidden bg-slate-950/90 backdrop-blur-2xl shadow-2xl ${accentGlow[card.accentColor]}`}
                style={{ maxHeight: '88vh' }}
              >
                {/* Cursor-reactive glass sheen */}
                <div
                  className="pointer-events-none absolute -inset-px rounded-3xl z-10 transition-opacity duration-300 opacity-0 group-hover:opacity-100"
                  style={{
                    background: `radial-gradient(500px circle at ${sheenPos.x}px ${sheenPos.y}px, rgba(255,255,255,0.04), transparent 70%)`,
                  }}
                />

                {/* Sticky close button */}
                <button
                  onClick={onClose}
                  className="absolute top-4 right-4 z-20 w-9 h-9 rounded-full bg-white/10 hover:bg-white/20 border border-white/15 flex items-center justify-center transition-colors"
                >
                  <X className="w-4 h-4 text-white" />
                </button>

                {/* Scrollable content */}
                <div className="overflow-y-auto" style={{ maxHeight: '88vh' }}>
                  {/* Animated gradient header */}
                  <div className={`relative px-8 pt-8 pb-6 bg-gradient-to-br ${card.gradientFrom} ${card.gradientTo}`}>
                    <FloatingParticles color={card.accentColor} />

                    {/* Illustration */}
                    <div className="relative h-28 mb-6 flex items-center justify-center">
                      <card.illustration />
                    </div>

                    {/* Icon */}
                    <motion.div
                      className={`w-14 h-14 rounded-2xl ${accentBg[card.accentColor]} border border-white/10 flex items-center justify-center mb-5`}
                      initial={{ scale: 0.8, opacity: 0 }}
                      animate={{ scale: 1, opacity: 1 }}
                      transition={{ delay: 0.1, type: 'spring', damping: 20 }}
                    >
                      {card.icon}
                    </motion.div>

                    {/* Progress indicator */}
                    <div className="flex items-center gap-1.5 mb-3">
                      {cards.map((_, i) => (
                        <button
                          key={i}
                          onClick={() => onNavigate(i)}
                          className={`h-1 rounded-full transition-all duration-300 ${
                            i === currentIndex
                              ? `w-6 ${accentBg[card.accentColor].replace('bg-', 'bg-').replace('/10', '')} opacity-100`
                              : 'w-2 bg-white/20 hover:bg-white/40'
                          }`}
                        />
                      ))}
                      <span className={`text-xs ml-1 ${accentText[card.accentColor]} font-medium`}>
                        {currentIndex + 1} / {cards.length}
                      </span>
                    </div>

                    <h2 className="text-2xl font-heading font-bold text-white leading-snug pr-10">
                      {card.title}
                    </h2>
                  </div>

                  {/* Body */}
                  <div className="px-8 py-6 space-y-6">
                    {/* Introduction */}
                    <p className="text-slate-300 leading-relaxed text-[15px]">
                      {card.introduction}
                    </p>

                    {/* Sections */}
                    {card.sections.map((section, i) => (
                      <div key={i}>
                        <div className={`h-px w-full bg-gradient-to-r ${accentDivider[card.accentColor]} mb-5`} />
                        <h3 className={`text-xs uppercase tracking-widest font-semibold mb-3 ${accentText[card.accentColor]}`}>
                          {section.heading}
                        </h3>
                        {section.items ? (
                          <ul className="space-y-2">
                            {section.items.map((item, j) => (
                              <li key={j} className="flex items-start gap-2.5 text-slate-300 text-sm">
                                <span className={`mt-1.5 w-1.5 h-1.5 rounded-full flex-shrink-0 ${accentBg[card.accentColor].replace('/10', '/60')}`} />
                                {item}
                              </li>
                            ))}
                          </ul>
                        ) : (
                          <p className="text-slate-300 text-sm leading-relaxed whitespace-pre-line">
                            {section.body}
                          </p>
                        )}
                      </div>
                    ))}

                    {/* Privacy note */}
                    {card.privacyNote && (
                      <div className="rounded-xl bg-white/5 border border-white/8 px-4 py-3">
                        <p className="text-slate-400 text-xs leading-relaxed">
                          🔒 {card.privacyNote}
                        </p>
                      </div>
                    )}
                  </div>

                  {/* Navigation footer */}
                  <div className="px-8 pb-6 flex items-center justify-between border-t border-white/8 pt-4">
                    <button
                      onClick={() => onNavigate(currentIndex === 0 ? cards.length - 1 : currentIndex - 1)}
                      className="flex items-center gap-2 text-sm text-slate-400 hover:text-white transition-colors group"
                    >
                      <ChevronLeft className="w-4 h-4 group-hover:-translate-x-0.5 transition-transform" />
                      Previous
                    </button>

                    <span className="text-xs text-slate-600">← → to navigate  ·  ESC to close</span>

                    <button
                      onClick={() => onNavigate(currentIndex === cards.length - 1 ? 0 : currentIndex + 1)}
                      className="flex items-center gap-2 text-sm text-slate-400 hover:text-white transition-colors group"
                    >
                      Next
                      <ChevronRight className="w-4 h-4 group-hover:translate-x-0.5 transition-transform" />
                    </button>
                  </div>
                </div>
              </div>
            </div>
          </motion.div>
        </div>
      )}
    </AnimatePresence>
  );
}
