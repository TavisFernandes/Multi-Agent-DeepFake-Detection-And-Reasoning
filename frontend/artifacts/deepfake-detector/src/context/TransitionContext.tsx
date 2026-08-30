import { createContext, useContext, useState, useCallback, ReactNode } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useLocation } from 'wouter';

// ─── Types ────────────────────────────────────────────────────────────────────

type Phase = 'idle' | 'covering' | 'uncovering';

interface TransitionContextValue {
  navigateTo: (path: string) => void;
}

// ─── Context ──────────────────────────────────────────────────────────────────

const TransitionContext = createContext<TransitionContextValue>({
  navigateTo: () => {},
});

export const usePageTransition = () => useContext(TransitionContext);

// ─── Reduced-motion detection ─────────────────────────────────────────────────

const prefersReducedMotion =
  typeof window !== 'undefined'
    ? window.matchMedia('(prefers-reduced-motion: reduce)').matches
    : false;

// ─── Timing (ms) ─────────────────────────────────────────────────────────────
// Cover animation:  260ms  (circle iris expands to fill screen)
// Route change:    at 260ms
// Uncover anim:    180ms  (overlay fades away)
// Total:           ~440ms

const COVER_DURATION = 260;
const UNCOVER_DURATION = 180;

// ─── Provider ─────────────────────────────────────────────────────────────────

export function TransitionProvider({ children }: { children: ReactNode }) {
  const [, setLocation] = useLocation();
  const [phase, setPhase] = useState<Phase>('idle');

  const navigateTo = useCallback(
    (path: string) => {
      if (prefersReducedMotion) {
        // Simplified: quick fade crossfade
        setPhase('covering');
        setTimeout(() => {
          setLocation(path);
          setPhase('uncovering');
          setTimeout(() => setPhase('idle'), 100);
        }, 90);
        return;
      }

      setPhase('covering');
      setTimeout(() => {
        setLocation(path);
        setPhase('uncovering');
        setTimeout(() => setPhase('idle'), UNCOVER_DURATION + 50);
      }, COVER_DURATION);
    },
    [setLocation],
  );

  return (
    <TransitionContext.Provider value={{ navigateTo }}>
      {children}

      {/* ── Cinematic overlay ── */}
      <AnimatePresence>
        {phase !== 'idle' && (
          <motion.div
            key="page-transition-overlay"
            className="fixed inset-0 overflow-hidden pointer-events-none"
            style={{ zIndex: 9999, background: 'hsl(222 47% 4%)' }}
            // Reduced-motion variant: simple opacity fade
            {...(prefersReducedMotion
              ? {
                  initial: { opacity: 0 },
                  animate: { opacity: phase === 'covering' ? 0.97 : 0 },
                  exit: { opacity: 0 },
                  transition: { duration: 0.18 },
                }
              : {
                  // Full cinematic: iris wipe from centre
                  initial: { clipPath: 'circle(0% at 50% 50%)', opacity: 1 },
                  animate:
                    phase === 'covering'
                      ? { clipPath: 'circle(150% at 50% 50%)', opacity: 1 }
                      : { clipPath: 'circle(150% at 50% 50%)', opacity: 0 },
                  exit: { opacity: 0 },
                  transition:
                    phase === 'covering'
                      ? { duration: COVER_DURATION / 1000, ease: [0.25, 0.46, 0.45, 0.94] }
                      : { duration: UNCOVER_DURATION / 1000, ease: [0.55, 0, 1, 0.45] },
                })}
          >
          </motion.div>
        )}
      </AnimatePresence>
    </TransitionContext.Provider>
  );
}
