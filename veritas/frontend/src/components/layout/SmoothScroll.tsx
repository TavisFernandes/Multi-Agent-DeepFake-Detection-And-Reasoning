import React, { useEffect, useRef, useState, useCallback, createContext, useContext } from 'react';
import Lenis from 'lenis';
import 'lenis/dist/lenis.css';
import gsap from 'gsap';

interface LenisContextValue {
  lenis: Lenis | null;
  scrollTo: (target: string | HTMLElement | number, options?: { offset?: number; duration?: number }) => void;
}

const LenisContext = createContext<LenisContextValue>({
  lenis: null,
  scrollTo: () => {},
});

export const useLenis = () => useContext(LenisContext);

interface SmoothScrollProps {
  children: React.ReactNode;
}

export const SmoothScroll: React.FC<SmoothScrollProps> = ({ children }) => {
  const lenisRef = useRef<Lenis | null>(null);
  // Use state so Provider re-renders with live instance after useEffect fires
  const [lenisInstance, setLenisInstance] = useState<Lenis | null>(null);

  useEffect(() => {
    // Respect user's reduced-motion preference
    const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    if (prefersReducedMotion) {
      return;
    }


    // Initialize Lenis with refined, restrained easing curve
    const lenis = new Lenis({
      duration: 1.1,
      easing: (t) => Math.min(1, 1.001 - Math.pow(2, -10 * t)), // refined easeOutExpo
      orientation: 'vertical',
      gestureOrientation: 'vertical',
      smoothWheel: true,
      wheelMultiplier: 0.95,
      touchMultiplier: 1.5,
      infinite: false,
    });

    lenisRef.current = lenis;
    setLenisInstance(lenis);


    // Drive Lenis from GSAP's single RAF loop
    const tickerUpdate = (time: number) => {
      lenis.raf(time * 1000);
    };

    gsap.ticker.add(tickerUpdate);
    gsap.ticker.lagSmoothing(0);

    // Global anchor link interceptor for smooth scrolling to sections
    const handleAnchorClick = (e: MouseEvent) => {
      const target = e.target as HTMLElement;
      const anchor = target.closest('a');
      if (anchor && anchor.hash && anchor.pathname === window.location.pathname) {
        const targetElement = document.querySelector(anchor.hash);
        if (targetElement) {
          e.preventDefault();
          lenis.scrollTo(targetElement as HTMLElement, {
            offset: -80,
            duration: 1.1,
          });
        }
      }
    };

    document.addEventListener('click', handleAnchorClick);

    // Cleanup lifecycle
    return () => {
      gsap.ticker.remove(tickerUpdate);
      document.removeEventListener('click', handleAnchorClick);
      lenis.destroy();
      lenisRef.current = null;
      setLenisInstance(null);
    };
  }, []);

  // scrollTo is stable across renders — always reads lenisRef.current at call time
  const scrollTo = useCallback(
    (target: string | HTMLElement | number, options?: { offset?: number; duration?: number }) => {
      if (lenisRef.current) {
        lenisRef.current.scrollTo(target as any, {
          offset: options?.offset ?? -80,
          duration: options?.duration ?? 1.1,
        });
      } else {
        if (typeof target === 'number') {
          window.scrollTo({ top: target, behavior: 'smooth' });
        } else {
          const el = typeof target === 'string' ? document.querySelector(target) : target;
          if (el) el.scrollIntoView({ behavior: 'smooth' });
        }
      }
    },
    []
  );

  return (
    <LenisContext.Provider value={{ lenis: lenisInstance, scrollTo }}>
      {children}
    </LenisContext.Provider>
  );
};
