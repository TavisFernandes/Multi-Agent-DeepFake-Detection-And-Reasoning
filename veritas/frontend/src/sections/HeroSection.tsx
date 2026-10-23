import { motion, useScroll, useTransform } from 'framer-motion';
import GradientASCII from '@/components/ui/GradientASCII';

const ASCII_COLORS = ['#1e3fd8', '#4f7bff', '#a5d8ff', '#eaf3ff'];
const EASE = [0.16, 1, 0.3, 1] as const;

// `ready` is false while the intro overlay is up, so the ASCII decode plays as the reveal.
export function HeroSection({ ready }: { ready: boolean }) {
  // As the models sheet rises, the hero recedes like a page behind a modal.
  const { scrollY } = useScroll();
  const progress = useTransform(scrollY, (y) => Math.min(y / window.innerHeight, 1));
  const scale = useTransform(progress, [0, 1], [1, 0.93]);
  const radius = useTransform(progress, [0, 1], [0, 28]);
  const scrim = useTransform(progress, [0, 1], [0, 0.25]);

  return (
    <section id="hero" className="sticky top-0 h-[100svh] min-h-[560px] w-full overflow-hidden bg-[#dde3ee]">
      <motion.div style={{ scale, borderRadius: radius }} className="relative h-full w-full overflow-hidden bg-paper">
        {ready && (
          <GradientASCII
            style={{ position: 'absolute', inset: 0 }}
            colors={ASCII_COLORS}
            shape="rise"
            lightMode
            cellSize={16}
            density={0.8}
            scale={1.1}
            glow={0.15}
            aria-hidden="true"
          />
        )}

        <div className="relative z-10 flex h-full flex-col items-center justify-center px-6 pb-16 text-center">
          <motion.h1
            initial={{ opacity: 0, y: 14 }}
            animate={ready ? { opacity: 1, y: 0 } : undefined}
            transition={{ duration: 1, delay: 0.6, ease: EASE }}
            className="hero-title max-w-[13ch] text-ink"
          >
            Find out if a video is real.
          </motion.h1>
          <motion.p
            initial={{ opacity: 0 }}
            animate={ready ? { opacity: 1 } : undefined}
          transition={{ duration: 1, delay: 1, ease: EASE }}
          className="mt-6 max-w-[46ch] text-base leading-relaxed text-ink/70 sm:text-lg"
        >
          VERITAS runs four forensic models over the faces, motion, audio and context in your media,
          then shows you the evidence behind every verdict.
        </motion.p>
      </div>
      <motion.div style={{ opacity: scrim }} className="pointer-events-none absolute inset-0 bg-ink" aria-hidden="true" />
      </motion.div>
    </section>
  );
}
