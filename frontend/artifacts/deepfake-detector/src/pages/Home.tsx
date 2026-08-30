import { motion } from 'framer-motion';
import { NavDock } from '@/components/layout/NavDock';
import { Footer } from '@/components/layout/Footer';
import { CursorSpotlight } from '@/components/ui/CursorSpotlight';
import { HeroSection } from '@/sections/HeroSection';
import { AboutSection } from '@/sections/AboutSection';
import { AgentShowcase } from '@/sections/AgentsSection';
import { PipelineSection } from '@/sections/PipelineSection';
import { DemoSection } from '@/sections/DemoSection';
import { WhyVeritasSection } from '@/sections/WhyVeritasSection';
import { ExplainabilitySection } from '@/sections/ExplainabilitySection';
import { StatsSection } from '@/sections/StatsSection';

export default function Home() {
  return (
    <motion.div
      className="min-h-screen w-full bg-background text-foreground selection:bg-cyan-500/30"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.45, ease: 'easeOut' }}
    >
      <CursorSpotlight />
      <NavDock />

      <main>
        <HeroSection />
        <AboutSection />
        <AgentShowcase />
        <PipelineSection />
        <DemoSection />
        <WhyVeritasSection />
        <ExplainabilitySection />
        <StatsSection />
      </main>

      <Footer />
    </motion.div>
  );
}
