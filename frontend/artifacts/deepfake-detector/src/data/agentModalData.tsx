import { motion } from 'framer-motion';
import { Eye, Film, Mic, Globe } from 'lucide-react';
import { CardModalData } from '@/components/ui/CardModal';

// Face mesh illustration
function FaceMeshIllustration() {
  const points = [
    [50, 18], [35, 28], [65, 28], [28, 42], [50, 38], [72, 42],
    [24, 58], [38, 55], [50, 60], [62, 55], [76, 58],
    [32, 72], [50, 78], [68, 72],
  ];
  const edges = [
    [0, 1], [0, 2], [1, 3], [2, 5], [1, 4], [2, 4], [3, 6], [4, 7],
    [4, 8], [4, 9], [5, 10], [7, 11], [8, 11], [8, 12], [8, 13], [9, 13], [10, 13],
  ];
  return (
    <svg className="w-full h-full" viewBox="0 0 100 100" preserveAspectRatio="xMidYMid meet">
      {edges.map(([a, b], i) => (
        <motion.line
          key={i}
          x1={points[a][0]} y1={points[a][1]}
          x2={points[b][0]} y2={points[b][1]}
          stroke="rgba(34,211,238,0.35)" strokeWidth="0.6"
          initial={{ opacity: 0 }}
          animate={{ opacity: [0.2, 0.6, 0.2] }}
          transition={{ duration: 2, repeat: Infinity, delay: (i * 0.08) % 1.5 }}
        />
      ))}
      {points.map(([x, y], i) => (
        <motion.circle
          key={`p-${i}`}
          cx={x} cy={y} r={1.5}
          fill="rgba(34,211,238,0.7)"
          animate={{ r: [1.2, 2.2, 1.2], opacity: [0.5, 1, 0.5] }}
          transition={{ duration: 1.8, repeat: Infinity, delay: (i * 0.12) % 1.5 }}
        />
      ))}
    </svg>
  );
}

// Timeline frames illustration
function TimelineIllustration() {
  return (
    <div className="flex flex-col items-center justify-center gap-3 w-full h-full px-4">
      <div className="w-full h-1 bg-white/10 rounded-full relative overflow-hidden">
        <motion.div
          className="absolute top-0 left-0 bottom-0 w-1/4 bg-blue-400/60 rounded-full"
          animate={{ x: ['-100%', '400%'] }}
          transition={{ duration: 2.5, repeat: Infinity, ease: 'linear' }}
        />
      </div>
      <div className="flex gap-1.5 w-full justify-center">
        {Array.from({ length: 7 }).map((_, i) => (
          <motion.div
            key={i}
            className="flex-1 border border-blue-500/30 rounded bg-blue-500/5"
            style={{ height: 36 }}
            animate={{ borderColor: ['rgba(96,165,250,0.2)', 'rgba(96,165,250,0.7)', 'rgba(96,165,250,0.2)'] }}
            transition={{ duration: 2, repeat: Infinity, delay: i * 0.2 }}
          />
        ))}
      </div>
    </div>
  );
}

// Waveform illustration
function WaveformIllustration() {
  const heights = [20, 45, 70, 55, 80, 35, 60, 75, 40, 65, 50, 30, 70, 45, 55, 20, 60, 80, 35, 50];
  return (
    <div className="flex items-center justify-center gap-1 w-full h-full px-4">
      {heights.map((h, i) => (
        <motion.div
          key={i}
          className="w-2 rounded-full bg-emerald-400/50"
          animate={{ height: [`${h * 0.4}%`, `${h}%`, `${h * 0.4}%`] }}
          transition={{
            duration: 0.6 + (i % 5) * 0.1,
            repeat: Infinity,
            delay: (i % 7) * 0.08,
          }}
          style={{ height: `${h * 0.5}%` }}
        />
      ))}
    </div>
  );
}

// Network graph illustration
function NetworkIllustration() {
  const nodes = [
    { cx: 20, cy: 30 }, { cx: 50, cy: 15 }, { cx: 80, cy: 28 },
    { cx: 65, cy: 55 }, { cx: 30, cy: 62 }, { cx: 50, cy: 78 },
    { cx: 78, cy: 72 }, { cx: 12, cy: 52 },
  ];
  const edges = [[0,1],[1,2],[2,3],[3,4],[4,5],[5,6],[6,3],[0,4],[1,3],[7,0],[7,4]];
  return (
    <svg className="w-full h-full" viewBox="0 0 100 100" preserveAspectRatio="xMidYMid meet">
      {edges.map(([a, b], i) => (
        <motion.line
          key={i}
          x1={nodes[a].cx} y1={nodes[a].cy}
          x2={nodes[b].cx} y2={nodes[b].cy}
          stroke="rgba(168,85,247,0.3)" strokeWidth="0.6"
          animate={{ opacity: [0.2, 0.7, 0.2] }}
          transition={{ duration: 2.5, repeat: Infinity, delay: i * 0.2 }}
        />
      ))}
      {nodes.map((n, i) => (
        <motion.circle
          key={i}
          cx={n.cx} cy={n.cy}
          fill="rgba(168,85,247,0.4)" stroke="rgba(168,85,247,0.6)" strokeWidth="0.8"
          animate={{ r: [3, 5, 3], opacity: [0.5, 1, 0.5] }}
          transition={{ duration: 2, repeat: Infinity, delay: (i * 0.3) % 2 }}
          r={3}
        />
      ))}
    </svg>
  );
}

export const agentModalData: CardModalData[] = [
  {
    icon: <Eye className="w-7 h-7 text-cyan-400" />,
    accentColor: 'cyan',
    gradientFrom: 'from-cyan-950/60',
    gradientTo: 'to-slate-950/0',
    title: 'Visual Analysis',
    introduction:
      "VERITAS carefully examines images and individual video frames for visual clues that may indicate digital manipulation. It looks for subtle inconsistencies that are often difficult to notice with the naked eye.",
    sections: [
      {
        heading: 'Checks performed',
        items: [
          'Facial proportions and alignment',
          'Lighting and shadow consistency',
          'Skin texture and blending around edges',
          'Reflections and background consistency',
          'Image quality variations across the frame',
        ],
      },
      {
        heading: 'Why it matters',
        body: "Many manipulated images contain tiny imperfections that become noticeable when examined closely. These are easy to miss with the naked eye but appear clearly under careful inspection.",
      },
      {
        heading: 'Example',
        body: "An image may appear realistic at first glance but contain inconsistent lighting around the face, or blurred edges that suggest the face was digitally placed onto a different body.",
      },
    ],
    illustration: FaceMeshIllustration,
  },
  {
    icon: <Film className="w-7 h-7 text-blue-400" />,
    accentColor: 'blue',
    gradientFrom: 'from-blue-950/60',
    gradientTo: 'to-slate-950/0',
    title: 'Motion Analysis',
    introduction:
      "Videos are made of many individual frames. VERITAS studies how movement changes across those frames, looking for the kind of subtle flickering that reveals a manipulated clip.",
    sections: [
      {
        heading: 'Checks performed',
        items: [
          'Natural facial movement and expression flow',
          'Lip synchronisation with speech',
          'Eye blinking patterns',
          'Head and body movement smoothness',
          'Object and scene consistency across frames',
        ],
      },
      {
        heading: 'Why it matters',
        body: "AI-generated videos sometimes produce small visual glitches that only appear while the video is playing. A static screenshot may look fine, but the motion reveals the manipulation.",
      },
      {
        heading: 'Example',
        body: "A person's face may briefly flicker or change shape between frames — even though the overall video appears realistic when watched casually.",
      },
    ],
    illustration: TimelineIllustration,
  },
  {
    icon: <Mic className="w-7 h-7 text-emerald-400" />,
    accentColor: 'emerald',
    gradientFrom: 'from-emerald-950/60',
    gradientTo: 'to-slate-950/0',
    title: 'Audio Analysis',
    introduction:
      "VERITAS analyses speech and sound for signs that the audio may have been generated, cloned, or modified — even when it sounds convincing to the human ear.",
    sections: [
      {
        heading: 'Checks performed',
        items: [
          'Voice consistency throughout the recording',
          'Natural speech rhythm and pacing',
          'Background sound patterns',
          'Unusual distortions or artefacts',
          'Abrupt transitions and overall audio quality',
        ],
      },
      {
        heading: 'Why it matters',
        body: "Synthetic voices often contain subtle patterns that differ from naturally recorded speech. These are nearly imperceptible to humans but clearly visible to careful analysis.",
      },
      {
        heading: 'Example',
        body: "A voice may sound convincing but contain repeating background noise or unusually consistent pronunciation — patterns that natural speech rarely produces.",
      },
    ],
    illustration: WaveformIllustration,
  },
  {
    icon: <Globe className="w-7 h-7 text-purple-400" />,
    accentColor: 'purple',
    gradientFrom: 'from-purple-950/60',
    gradientTo: 'to-slate-950/0',
    title: 'Context Verification',
    introduction:
      "VERITAS looks beyond the media itself. It compares available information with trusted online sources to determine whether the content matches what it claims to be.",
    sections: [
      {
        heading: 'Checks performed',
        items: [
          'Source credibility and publication information',
          'Metadata consistency',
          'Related reports and publicly available references',
          'Timeline consistency',
          'Cross-referencing claimed context with known facts',
        ],
      },
      {
        heading: 'Why it matters',
        body: "Even authentic media can be misleading if shared with false claims or taken out of context. Context verification catches these cases that purely visual or audio checks would miss.",
      },
      {
        heading: 'Example',
        body: "A genuine photograph from several years ago may be reposted during a current event with a misleading description — visually authentic, but factually deceptive.",
      },
    ],
    illustration: NetworkIllustration,
  },
];
