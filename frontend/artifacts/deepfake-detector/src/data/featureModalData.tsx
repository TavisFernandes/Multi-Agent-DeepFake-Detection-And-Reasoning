import { motion } from 'framer-motion';
import { Layers, Zap, MessageSquareText, Target } from 'lucide-react';
import { CardModalData } from '@/components/ui/CardModal';

// Illustration: overlapping rings — evidence layers
function LayersIllustration() {
  return (
    <div className="relative flex items-center justify-center w-full h-full">
      {[0, 1, 2, 3].map((i) => (
        <motion.div
          key={i}
          className="absolute rounded-full border border-cyan-400/30"
          style={{ width: 40 + i * 28, height: 40 + i * 28 }}
          animate={{ scale: [1, 1.06, 1], opacity: [0.3 + i * 0.1, 0.7, 0.3 + i * 0.1] }}
          transition={{ duration: 2.5, repeat: Infinity, delay: i * 0.4, ease: 'easeInOut' }}
        />
      ))}
      <motion.div
        className="w-10 h-10 rounded-full bg-cyan-500/20 flex items-center justify-center"
        animate={{ scale: [1, 1.1, 1] }}
        transition={{ duration: 2, repeat: Infinity, ease: 'easeInOut' }}
      >
        <Layers className="w-5 h-5 text-cyan-400" />
      </motion.div>
    </div>
  );
}

// Illustration: nodes merging into one point
function MergeIllustration() {
  const nodes = [
    { x: 15, y: 30 }, { x: 15, y: 60 }, { x: 85, y: 30 }, { x: 85, y: 60 },
  ];
  return (
    <div className="relative w-full h-full flex items-center justify-center">
      <svg className="w-full h-full" viewBox="0 0 100 80" preserveAspectRatio="xMidYMid meet">
        {nodes.map((n, i) => (
          <g key={i}>
            <motion.line
              x1={n.x} y1={n.y} x2="50" y2="40"
              stroke="rgba(96,165,250,0.4)" strokeWidth="0.8"
              initial={{ pathLength: 0 }}
              animate={{ pathLength: [0, 1, 1, 0] }}
              transition={{ duration: 2.5, repeat: Infinity, delay: i * 0.3 }}
            />
            <motion.circle
              cx={n.x} cy={n.y} r="4"
              fill="rgba(96,165,250,0.25)" stroke="rgba(96,165,250,0.5)" strokeWidth="0.8"
              animate={{ r: [3, 5, 3], opacity: [0.5, 1, 0.5] }}
              transition={{ duration: 2, repeat: Infinity, delay: i * 0.3 }}
            />
          </g>
        ))}
        <motion.circle
          cx="50" cy="40" r="6"
          fill="rgba(96,165,250,0.3)" stroke="rgba(96,165,250,0.7)" strokeWidth="1"
          animate={{ r: [5, 8, 5] }}
          transition={{ duration: 2, repeat: Infinity, ease: 'easeInOut' }}
        />
      </svg>
    </div>
  );
}

// Illustration: text lines appearing clearly
function TextIllustration() {
  return (
    <div className="flex flex-col items-center justify-center gap-2 w-full h-full px-8">
      {[1, 0.7, 0.5].map((opacity, i) => (
        <motion.div
          key={i}
          className="h-2 bg-purple-400/40 rounded-full"
          style={{ width: `${80 - i * 15}%` }}
          animate={{ opacity: [opacity * 0.5, opacity, opacity * 0.5] }}
          transition={{ duration: 2, repeat: Infinity, delay: i * 0.3 }}
        />
      ))}
    </div>
  );
}

// Illustration: confidence meter
function ConfidenceIllustration() {
  return (
    <div className="flex items-center justify-center gap-4 w-full h-full">
      {['High', 'Mod.', 'Low'].map((label, i) => {
        const heights = [80, 50, 25];
        const colors = ['bg-emerald-400/50', 'bg-amber-400/40', 'bg-red-400/30'];
        return (
          <div key={i} className="flex flex-col items-center gap-1">
            <motion.div
              className={`w-8 rounded-t-md ${colors[i]}`}
              animate={{ height: [heights[i] * 0.6, heights[i], heights[i] * 0.6] }}
              transition={{ duration: 2 + i * 0.4, repeat: Infinity, ease: 'easeInOut', delay: i * 0.3 }}
              style={{ height: heights[i] }}
            />
            <span className="text-[9px] text-slate-400">{label}</span>
          </div>
        );
      })}
    </div>
  );
}

export const featureModalData: CardModalData[] = [
  {
    icon: <Layers className="w-7 h-7 text-cyan-400" />,
    accentColor: 'cyan',
    gradientFrom: 'from-cyan-950/60',
    gradientTo: 'to-slate-950/0',
    title: 'Looking at More Than One Clue',
    introduction:
      "VERITAS doesn't rely on just one signal to decide whether something is real or fake. Instead, it examines several different types of evidence before reaching a conclusion.",
    sections: [
      {
        heading: 'How it works',
        items: [
          'The visual appearance of images and videos',
          'Movement between video frames',
          'Audio quality and speech characteristics',
          'Information available from trusted online sources',
        ],
      },
      {
        heading: 'Why this matters',
        body: "Some fake content looks convincing visually but contains unusual audio. Others have realistic audio but inconsistent movement. Looking at multiple clues helps reduce mistakes and makes the overall conclusion more reliable.",
      },
      {
        heading: 'Example',
        body: "Imagine a video where the person's face looks realistic, the voice sounds artificial, and the original source cannot be verified. Taken together, these signs make the content more suspicious than any one clue alone.",
      },
    ],
    privacyNote: 'Your uploaded content is analysed only to generate the requested result.',
    illustration: LayersIllustration,
  },
  {
    icon: <Zap className="w-7 h-7 text-blue-400" />,
    accentColor: 'blue',
    gradientFrom: 'from-blue-950/60',
    gradientTo: 'to-slate-950/0',
    title: 'One Clear Answer, Not Four Separate Opinions',
    introduction:
      "Instead of showing separate results for each type of analysis, VERITAS combines all findings into one easy-to-understand explanation.",
    sections: [
      {
        heading: 'How it works',
        body: "Different analyses may sometimes disagree. Visual inspection may suggest the content appears authentic, while audio analysis detects unusual speech patterns. VERITAS weighs these observations together before producing a final explanation.",
      },
      {
        heading: 'Why this matters',
        body: "People shouldn't need to understand technical reports. VERITAS simplifies everything into one final conclusion with supporting reasoning that anyone can follow.",
      },
      {
        heading: 'Example',
        body: `Rather than listing:\n• Visual: OK\n• Audio: Suspicious\n• Context: Unknown\n\nVERITAS explains:\n"While the visuals appear natural, unusual audio characteristics and missing supporting context reduce confidence in the authenticity of this content."`,
      },
    ],
    illustration: MergeIllustration,
  },
  {
    icon: <MessageSquareText className="w-7 h-7 text-purple-400" />,
    accentColor: 'purple',
    gradientFrom: 'from-purple-950/60',
    gradientTo: 'to-slate-950/0',
    title: 'Clear Answers Without Technical Jargon',
    introduction:
      "AI tools often produce confusing scores and unfamiliar terminology. VERITAS explains its findings in plain language that anyone can understand.",
    sections: [
      {
        heading: 'How it works',
        items: [
          'What was observed',
          'Why it matters',
          'How those observations influenced the conclusion',
          'Suggestions for interpreting the result',
        ],
      },
      {
        heading: 'Why this matters',
        body: "People should understand why a conclusion was reached instead of simply accepting an automated decision. Transparency builds trust.",
      },
      {
        heading: 'Example',
        body: `Instead of:\n"Inference confidence = 0.91"\n\nVERITAS says:\n"We found several visual and audio inconsistencies that are commonly associated with AI-generated media."`,
      },
    ],
    illustration: TextIllustration,
  },
  {
    icon: <Target className="w-7 h-7 text-emerald-400" />,
    accentColor: 'emerald',
    gradientFrom: 'from-emerald-950/60',
    gradientTo: 'to-slate-950/0',
    title: "Sometimes the Answer Isn't Certain",
    introduction:
      "No automated system can guarantee perfect accuracy. VERITAS communicates uncertainty honestly rather than pretending every answer is definitive.",
    sections: [
      {
        heading: 'How it works',
        items: [
          'High Confidence — strong evidence supports the conclusion',
          'Moderate Confidence — some indicators present, but not conclusive',
          'Low Confidence — insufficient evidence for a reliable judgement',
        ],
      },
      {
        heading: 'Why this matters',
        body: "Being transparent about uncertainty helps you make informed decisions rather than relying blindly on automation. An honest low-confidence result is more useful than a false certainty.",
      },
      {
        heading: 'Example',
        body: "A heavily compressed video with poor audio may not contain enough information for a confident decision. In those situations, VERITAS clearly explains why the result is less certain.",
      },
    ],
    illustration: ConfidenceIllustration,
  },
];
