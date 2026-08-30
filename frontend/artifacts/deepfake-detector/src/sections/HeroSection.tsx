import { motion } from 'framer-motion';
import { NeuralCanvas } from '@/components/ui/NeuralCanvas';
import { ArrowRight, ChevronDown } from 'lucide-react';
import { useScrollProgress } from '@/hooks/useScrollProgress';
import { usePageTransition } from '@/context/TransitionContext';

export function HeroSection() {
  const scrollProgress = useScrollProgress();
  const { navigateTo } = usePageTransition();

  const scrollTo = (href: string) => {
    const el = document.querySelector(href);
    if (el) el.scrollIntoView({ behavior: 'smooth' });
  };

  return (
    <section id="hero" className="relative min-h-screen flex items-center justify-center overflow-hidden pt-20">
      <NeuralCanvas />

      {/* Background Glow Blobs */}
      <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-cyan-500/10 rounded-full blur-[120px] pointer-events-none mix-blend-screen" />
      <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-purple-500/10 rounded-full blur-[120px] pointer-events-none mix-blend-screen" />

      <div className="relative z-10 max-w-5xl mx-auto px-6 text-center">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 1, delay: 0.2 }}
        >
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full border border-white/10 bg-white/5 backdrop-blur-sm text-sm text-cyan-400 mb-8">
            <span className="relative flex h-2 w-2">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-cyan-400 opacity-75"></span>
              <span className="relative inline-flex rounded-full h-2 w-2 bg-cyan-500"></span>
            </span>
            VERITAS is Online &bull; Ready to Analyse
          </div>

          <h1 className="text-5xl md:text-7xl lg:text-8xl font-heading font-bold tracking-tight leading-tight mb-6 text-white">
            Uncover the Truth <br className="hidden md:block" />
            <span className="text-gradient">Behind Digital Content.</span>
          </h1>

          <p className="text-lg md:text-xl text-slate-400 max-w-2xl mx-auto mb-10 leading-relaxed font-light">
            Upload an image, video, audio clip, or paste a social media link. VERITAS analyses the content and explains whether it appears authentic or AI-generated — in clear, easy-to-understand language.
          </p>

          <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
            <button
              onClick={() => navigateTo('/console')}
              data-testid="button-hero-launch-console"
              className="w-full sm:w-auto px-8 py-4 bg-white text-background font-medium rounded-full hover:bg-slate-200 transition-colors flex items-center justify-center gap-2 group"
            >
              Launch Console
              <ArrowRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
            </button>
            <button
              onClick={() => scrollTo('#about')}
              data-testid="button-hero-learn-more"
              className="w-full sm:w-auto px-8 py-4 glass-card text-white font-medium rounded-full hover:bg-white/10 transition-colors flex items-center justify-center gap-2"
            >
              Learn More
            </button>
          </div>
        </motion.div>
      </div>

      {/* Scroll indicator */}
      <motion.div
        className="absolute bottom-8 left-1/2 -translate-x-1/2 text-slate-500 animate-bounce"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 2, duration: 1 }}
      >
        <ChevronDown className="w-6 h-6" />
      </motion.div>

      {/* Right Edge Scroll Progress */}
      <div className="fixed right-0 top-0 bottom-0 w-1 bg-white/5 z-50">
        <div
          className="w-full bg-gradient-to-b from-cyan-400 to-purple-500 origin-top"
          style={{ height: `${scrollProgress}%` }}
        />
      </div>
    </section>
  );
}
