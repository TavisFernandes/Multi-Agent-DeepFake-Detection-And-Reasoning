import { useEffect, useState } from 'react';
import { AnimatePresence, motion } from 'framer-motion';

const HOLD_MS = 1700;

// Full-screen wordmark that fades off; onReveal fires as the fade starts so the hero can begin underneath it.
export function Intro({ onReveal }: { onReveal: () => void }) {
  const [visible, setVisible] = useState(true);

  useEffect(() => {
    const timer = setTimeout(() => {
      setVisible(false);
      onReveal();
    }, HOLD_MS);
    return () => clearTimeout(timer);
  }, [onReveal]);

  return (
    <AnimatePresence>
      {visible && (
        <motion.div
          key="intro"
          className="fixed inset-0 z-[100] flex items-center justify-center bg-paper"
          exit={{ opacity: 0 }}
          transition={{ duration: 0.9, ease: [0.65, 0, 0.35, 1] }}
        >
          <motion.span
            className="intro-mark"
            initial={{ opacity: 0, letterSpacing: '0.7em', filter: 'blur(10px)' }}
            animate={{ opacity: 1, letterSpacing: '0.28em', filter: 'blur(0px)' }}
            transition={{ duration: 1.3, ease: [0.16, 1, 0.3, 1] }}
          >
            VERITAS
          </motion.span>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
