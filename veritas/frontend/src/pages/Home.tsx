import { useCallback, useState } from 'react';
import { MotionConfig } from 'framer-motion';
import { NavDock } from '@/components/layout/NavDock';
import { Footer } from '@/components/layout/Footer';
import { SmoothScroll } from '@/components/layout/SmoothScroll';
import { Intro } from '@/components/layout/Intro';
import { HeroSection } from '@/sections/HeroSection';
import { ModelsSheet, VerifyCta } from '@/sections/ModelsSheet';

// Play the intro once per page load, not again when returning from the console.
let introPlayed =
  typeof window !== 'undefined' && window.matchMedia('(prefers-reduced-motion: reduce)').matches;

export default function Home() {
  const [showIntro] = useState(!introPlayed);
  const [ready, setReady] = useState(introPlayed);
  const reveal = useCallback(() => {
    introPlayed = true;
    setReady(true);
  }, []);

  return (
    <SmoothScroll>
      <MotionConfig reducedMotion="user">
        <div className="min-h-screen w-full bg-paper text-ink selection:bg-[#a5d8ff]">
          {showIntro && <Intro onReveal={reveal} />}
          <NavDock />
          <main>
            {/* Wrapper bounds the sticky hero so it releases (and stops rendering) once the sheet covers it */}
            <div className="relative">
              <HeroSection ready={ready} />
              <ModelsSheet />
            </div>
            <VerifyCta />
          </main>
          <Footer />
        </div>
      </MotionConfig>
    </SmoothScroll>
  );
}
