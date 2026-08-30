import { useState, useEffect } from 'react';
import { Shield } from 'lucide-react';
import { motion } from 'framer-motion';
import { usePageTransition } from '@/context/TransitionContext';

const NAV_ITEMS = [
  { label: 'Home', href: '#hero' },
  { label: 'How It Works', href: '#about' },
  { label: 'Our Analysts', href: '#agents' },
  { label: 'The Process', href: '#pipeline' },
  { label: 'Why VERITAS', href: '#why' },
  { label: 'Results', href: '#stats' },
];

export function NavDock() {
  const [activeId, setActiveId] = useState('hero');
  const { navigateTo } = usePageTransition();

  useEffect(() => {
    const handleScroll = () => {
      const scrollPosition = window.scrollY + window.innerHeight / 3;
      const sections = NAV_ITEMS.map(item => ({
        id: item.href.substring(1),
        element: document.getElementById(item.href.substring(1))
      }));
      for (let i = sections.length - 1; i >= 0; i--) {
        const section = sections[i];
        if (section.element && scrollPosition >= section.element.offsetTop) {
          setActiveId(section.id);
          break;
        }
      }
    };
    window.addEventListener('scroll', handleScroll, { passive: true });
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  const scrollTo = (href: string) => {
    const el = document.querySelector(href);
    if (el) el.scrollIntoView({ behavior: 'smooth' });
  };

  return (
    <motion.div
      initial={{ y: -100, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      transition={{ duration: 0.8, ease: 'easeOut' }}
      className="fixed top-6 left-0 right-0 z-40 flex justify-center px-4"
    >
      <div className="glass-card rounded-full px-6 py-3 flex items-center gap-8 shadow-2xl">
        <a
          href="#hero"
          onClick={(e) => { e.preventDefault(); scrollTo('#hero'); }}
          className="flex items-center gap-2 text-white hover:text-cyan-400 transition-colors"
        >
          <Shield className="w-5 h-5 text-cyan-400" />
          <span className="font-heading font-bold tracking-wide hidden sm:block">VERITAS</span>
        </a>

        <nav className="hidden md:flex items-center gap-1">
          {NAV_ITEMS.map((item) => (
            <a
              key={item.href}
              href={item.href}
              onClick={(e) => { e.preventDefault(); scrollTo(item.href); }}
              className={`px-4 py-1.5 rounded-full text-sm font-medium transition-all duration-300 ${
                activeId === item.href.substring(1)
                  ? 'bg-white/10 text-white'
                  : 'text-slate-400 hover:text-white hover:bg-white/5'
              }`}
            >
              {item.label}
            </a>
          ))}
        </nav>

        <button
          onClick={() => navigateTo('/console')}
          data-testid="button-launch-console"
          className="relative bg-cyan-500/10 text-cyan-400 hover:bg-cyan-500/20 border border-cyan-500/40 px-5 py-2 rounded-full text-sm font-semibold transition-all duration-300 shadow-[0_0_15px_rgba(6,182,212,0.2)] hover:shadow-[0_0_25px_rgba(6,182,212,0.4)]"
        >
          Launch Console
        </button>
      </div>
    </motion.div>
  );
}
