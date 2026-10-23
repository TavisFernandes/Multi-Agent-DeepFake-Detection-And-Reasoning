import React, { useState, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useLocation } from 'wouter';
import { ForensicsLabCanvas } from './ForensicsLabCanvas';
import { LabHudOverlay } from './LabHudOverlay';
import { StationId } from './types';
import { 
  ArrowRight, 
  ChevronDown, 
  Compass, 
  Cpu, 
  Scan, 
  Layers, 
  Activity, 
  Network
} from 'lucide-react';

export function Lab3DSection() {
  const [, setLocation] = useLocation();
  const [activeStationId, setActiveStationId] = useState<StationId>('overview');
  const [hoveredStationId, setHoveredStationId] = useState<StationId | null>(null);
  const [hasEnteredLab, setHasEnteredLab] = useState(false);

  const handleSelectStation = useCallback((id: StationId) => {
    setActiveStationId(id);
    if (!hasEnteredLab && id !== 'overview') {
      setHasEnteredLab(true);
    }
  }, [hasEnteredLab]);

  const scrollToBriefing = () => {
    const el = document.getElementById('about');
    if (el) el.scrollIntoView({ behavior: 'smooth' });
  };

  return (
    <section id="hero" className="relative w-full h-screen min-h-[700px] bg-[#040812] text-zinc-100 overflow-hidden">
      {/* 1. Full-bleed 3D Interactive Three.js Canvas */}
      <ForensicsLabCanvas
        activeStationId={activeStationId}
        onSelectStation={handleSelectStation}
        onHoverStation={setHoveredStationId}
        className="absolute inset-0 z-0"
      />

      {/* 2. Top-level HUD and Station Dossier Overlay */}
      <LabHudOverlay
        activeStationId={activeStationId}
        onSelectStation={handleSelectStation}
        hoveredStationId={hoveredStationId}
      />

      {/* 3. Hero Storytelling Overlay (Shown when in Overview before or during exploration) */}
      <AnimatePresence>
        {!hasEnteredLab && activeStationId === 'overview' && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            transition={{ duration: 0.6, ease: 'easeOut' }}
            className="absolute inset-0 pointer-events-none z-10 flex flex-col items-center justify-center text-center px-6"
          >
            {/* Cinematic Cybernetic Badge */}
            <div className="inline-flex items-center gap-2.5 px-4 py-1.5 rounded-full border border-cyan-500/30 bg-cyan-950/40 backdrop-blur-md text-xs font-mono text-cyan-300 mb-6 shadow-lg shadow-cyan-500/10 pointer-events-auto">
              <span className="relative flex h-2 w-2">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-cyan-400 opacity-75" />
                <span className="relative inline-flex rounded-full h-2 w-2 bg-cyan-500" />
              </span>
              <span>NEURAL FORENSIC ENGINE // MULTI-MODAL EVIDENCE SYNTHESIS</span>
            </div>

            {/* Hero Main Headline */}
            <h1 className="text-4xl sm:text-6xl md:text-7xl lg:text-8xl font-black tracking-tight text-white mb-6 uppercase max-w-5xl leading-tight">
              Unmask <span className="text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 via-sky-300 to-indigo-400">Synthetics</span>. <br />
              Prove Authenticity.
            </h1>

            {/* Sub-headline */}
            <p className="text-sm sm:text-base md:text-lg text-zinc-300/90 max-w-2xl mx-auto mb-10 leading-relaxed font-normal">
              An interactive 3D laboratory inspecting digital media with three neural detectors (spatial boundaries, temporal coherence, acoustic synthesis) and a cross-modal explanation of how they agree.
            </p>

            {/* Action Buttons */}
            <div className="flex flex-col sm:flex-row items-center justify-center gap-4 pointer-events-auto">
              <button
                onClick={() => {
                  setHasEnteredLab(true);
                  setActiveStationId('spatial');
                }}
                className="w-full sm:w-auto flex items-center justify-center gap-3 px-8 py-3.5 rounded-xl bg-cyan-500 hover:bg-cyan-400 text-black font-bold text-xs sm:text-sm tracking-wider uppercase transition-all shadow-xl shadow-cyan-500/25 hover:shadow-cyan-500/40 cursor-pointer"
              >
                <Compass className="w-4 h-4" />
                <span>Enter The 3D Lab</span>
              </button>

              <button
                onClick={() => setLocation('/console')}
                data-testid="button-hero-launch-console"
                className="w-full sm:w-auto flex items-center justify-center gap-2.5 px-8 py-3.5 rounded-xl bg-zinc-900/90 hover:bg-zinc-800 border border-zinc-700/80 text-white font-semibold text-xs sm:text-sm tracking-wider uppercase transition-all backdrop-blur-md cursor-pointer"
              >
                <span>Launch Live Console</span>
                <ArrowRight className="w-4 h-4 text-cyan-400" />
              </button>

              <button
                onClick={scrollToBriefing}
                data-testid="button-hero-learn-more"
                className="hidden"
                aria-hidden="true"
              >
                Learn More
              </button>
            </div>

            {/* Quick Agent Direct Teleports */}
            <div className="mt-12 flex flex-wrap items-center justify-center gap-2 pointer-events-auto max-w-2xl">
              <span className="text-[11px] font-mono uppercase text-zinc-500 mr-2">Stations:</span>
              <button
                onClick={() => { setHasEnteredLab(true); setActiveStationId('spatial'); }}
                className="flex items-center gap-1.5 px-3 py-1 rounded-lg bg-zinc-950/70 border border-cyan-500/30 text-xs font-mono text-cyan-300 hover:bg-cyan-950/40 cursor-pointer transition-colors"
              >
                <Scan className="w-3.5 h-3.5" />
                <span>Spatial</span>
              </button>
              <button
                onClick={() => { setHasEnteredLab(true); setActiveStationId('temporal'); }}
                className="flex items-center gap-1.5 px-3 py-1 rounded-lg bg-zinc-950/70 border border-blue-500/30 text-xs font-mono text-blue-300 hover:bg-blue-950/40 cursor-pointer transition-colors"
              >
                <Layers className="w-3.5 h-3.5" />
                <span>Temporal</span>
              </button>
              <button
                onClick={() => { setHasEnteredLab(true); setActiveStationId('audio'); }}
                className="flex items-center gap-1.5 px-3 py-1 rounded-lg bg-zinc-950/70 border border-emerald-500/30 text-xs font-mono text-emerald-300 hover:bg-emerald-950/40 cursor-pointer transition-colors"
              >
                <Activity className="w-3.5 h-3.5" />
                <span>Audio</span>
              </button>
              <button
                onClick={() => { setHasEnteredLab(true); setActiveStationId('context'); }}
                className="flex items-center gap-1.5 px-3 py-1 rounded-lg bg-zinc-950/70 border border-purple-500/30 text-xs font-mono text-purple-300 hover:bg-purple-950/40 cursor-pointer transition-colors"
              >
                <Network className="w-3.5 h-3.5" />
                <span>Context</span>
              </button>
              <button
                onClick={() => { setHasEnteredLab(true); setActiveStationId('fusion'); }}
                className="flex items-center gap-1.5 px-3 py-1 rounded-lg bg-zinc-950/70 border border-amber-500/30 text-xs font-mono text-amber-300 hover:bg-amber-950/40 cursor-pointer transition-colors"
              >
                <Cpu className="w-3.5 h-3.5" />
                <span>Fusion Core</span>
              </button>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* 4. Bottom Scroll Down Indicator to Briefing & Documentation */}
      <div className="absolute bottom-6 left-1/2 -translate-x-1/2 z-10 pointer-events-auto">
        <button
          onClick={scrollToBriefing}
          className="flex flex-col items-center gap-1.5 text-zinc-400 hover:text-cyan-300 transition-colors cursor-pointer group"
          title="Scroll down to Architecture Briefing"
        >
          <span className="text-[10px] font-mono tracking-widest uppercase opacity-70 group-hover:opacity-100">
            Scroll to Briefing
          </span>
          <ChevronDown className="w-5 h-5 animate-bounce text-cyan-400" />
        </button>
      </div>
    </section>
  );
}
