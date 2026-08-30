import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { Activity, CheckCircle, Clock } from 'lucide-react';

const initialAgents = [
  { id: 'spatial', name: 'Spatial Analysis', color: 'cyan', baseConf: 0, status: 'idle', task: 'Awaiting input...' },
  { id: 'temporal', name: 'Temporal Analysis', color: 'blue', baseConf: 0, status: 'idle', task: 'Awaiting input...' },
  { id: 'audio', name: 'Spectral Audio', color: 'emerald', baseConf: 0, status: 'idle', task: 'Awaiting input...' },
  { id: 'context', name: 'Contextual Verify', color: 'purple', baseConf: 0, status: 'idle', task: 'Awaiting input...' }
];

export function DashboardSection() {
  const [agents, setAgents] = useState(initialAgents);
  const [isRunning, setIsRunning] = useState(false);

  useEffect(() => {
    // We simulate a continuous background running state for the dashboard specifically
    const interval = setInterval(() => {
      setAgents(current => 
        current.map(agent => {
          // Simulate some random fluctuations
          if (Math.random() > 0.7) {
            return {
              ...agent,
              status: Math.random() > 0.5 ? 'analyzing' : 'complete',
              baseConf: Math.min(100, Math.max(0, agent.baseConf + (Math.random() * 10 - 3))),
              task: Math.random() > 0.5 ? 'Processing tensor blocks...' : 'Extracting feature maps...'
            };
          }
          return agent;
        })
      );
    }, 2000);

    return () => clearInterval(interval);
  }, []);

  return (
    <section id="dashboard" className="py-32 relative bg-background">
      <div className="max-w-7xl mx-auto px-6">
        <motion.div 
          className="flex flex-col md:flex-row justify-between items-end mb-16 gap-6"
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
        >
          <div>
            <h2 className="text-3xl md:text-5xl font-heading font-bold mb-4">Telemetry</h2>
            <p className="text-slate-400 max-w-2xl">
              Live monitoring of multi-agent consensus and individual node confidence scores.
            </p>
          </div>
          <div className="flex items-center gap-3 bg-white/5 border border-white/10 px-4 py-2 rounded-full">
            <div className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
            <span className="text-sm font-mono text-emerald-400">CLUSTER ONLINE</span>
          </div>
        </motion.div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          {agents.map((agent, i) => (
            <motion.div
              key={agent.id}
              className="glass-card p-6 rounded-2xl relative overflow-hidden"
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: i * 0.1 }}
            >
              <div className="flex justify-between items-start mb-6">
                <div>
                  <h3 className="text-lg font-heading font-semibold text-white mb-1">{agent.name}</h3>
                  <div className="flex items-center gap-2 text-xs font-mono">
                    {agent.status === 'idle' ? <Clock className="w-3 h-3 text-slate-500" /> :
                     agent.status === 'analyzing' ? <Activity className="w-3 h-3 text-cyan-500" /> :
                     <CheckCircle className="w-3 h-3 text-emerald-500" />}
                    <span className="uppercase text-slate-400">{agent.status}</span>
                  </div>
                </div>
              </div>

              <div className="flex items-center justify-center py-6">
                <div className="relative w-32 h-32 flex items-center justify-center">
                  <svg className="absolute inset-0 w-full h-full -rotate-90">
                    <circle cx="64" cy="64" r="56" fill="none" stroke="rgba(255,255,255,0.05)" strokeWidth="8" />
                    <circle 
                      cx="64" cy="64" r="56" 
                      fill="none" 
                      stroke={`var(--color-${agent.color}-500)`} 
                      strokeWidth="8"
                      strokeDasharray="351.8"
                      strokeDashoffset={351.8 - (351.8 * agent.baseConf) / 100}
                      className="transition-all duration-1000 ease-out"
                    />
                  </svg>
                  <div className="text-center">
                    <span className="text-3xl font-heading font-bold text-white">
                      {agent.baseConf.toFixed(1)}
                    </span>
                    <span className="text-sm text-slate-500 block">%</span>
                  </div>
                </div>
              </div>

              <div className="mt-4 bg-black/20 rounded p-3">
                <p className="text-xs font-mono text-slate-400 truncate">
                  &gt; {agent.task}
                </p>
              </div>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}
