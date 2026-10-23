import React, { useEffect, useState } from 'react';
import { usePageTransition } from '@/context/TransitionContext';
import { useLenis } from '@/components/layout/SmoothScroll';

function NavLink({ children, onClick }: { children: React.ReactNode; onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      className="rounded-full px-3.5 py-2 text-sm text-ink/70 transition-colors hover:bg-white/60 hover:text-ink focus-visible:outline-2 focus-visible:outline-signal cursor-pointer"
    >
      {children}
    </button>
  );
}

// One nav element that morphs between a bottom island (over the hero) and a full-width top bar (everywhere else).
export function NavDock() {
  const { navigateTo } = usePageTransition();
  const { scrollTo } = useLenis();
  const [docked, setDocked] = useState(false);

  useEffect(() => {
    const update = () => setDocked(window.scrollY > window.innerHeight * 0.5);
    update();
    window.addEventListener('scroll', update, { passive: true });
    return () => window.removeEventListener('scroll', update);
  }, []);

  return (
    <nav aria-label="Primary" data-docked={docked} className="island-nav liquid-glass">
      <div className="grid grid-cols-[1fr_auto_1fr] items-center gap-1">
        <div className="flex justify-end">
          <NavLink onClick={() => scrollTo('#models', { offset: 0 })}>Models</NavLink>
        </div>

        <button
          onClick={() => scrollTo(0)}
          aria-label="VERITAS, back to top"
          className="wordmark mx-2 h-11 rounded-full px-2 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-signal cursor-pointer"
        >
          VERITAS
        </button>

        <div className="flex justify-start">
          <button
            onClick={() => navigateTo('/console')}
            data-testid="button-launch-console"
            className="try-it ml-1 rounded-full px-5 py-2.5 text-sm font-medium text-white focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-signal cursor-pointer"
          >
            Try it
          </button>
        </div>
      </div>
    </nav>
  );
}
