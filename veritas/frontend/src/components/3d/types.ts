export type StationId = 'overview' | 'spatial' | 'temporal' | 'audio' | 'context' | 'fusion';

export interface StationConfig {
  id: StationId;
  name: string;
  tagline: string;
  badge: string;
  accentColor: string;
  position: [number, number, number];
  cameraPosition: [number, number, number];
  targetLookAt: [number, number, number];
  modelName: string;
  techDetails: {
    backbone: string;
    inputSpec: string;
    targetArtifacts: string;
    productionStatus: string;
  };
  summary: string;
  detectionPrinciples: string[];
}

export const STATIONS: Record<Exclude<StationId, 'overview'>, StationConfig> = {
  spatial: {
    id: 'spatial',
    name: 'Spatial Analyst',
    tagline: 'Frame-Level Visual & Artifact Inspection',
    badge: 'Spatial Agent',
    accentColor: '#00f0ff',
    position: [-6, 0, -4],
    cameraPosition: [-6, 2.2, -0.6],
    targetLookAt: [-6, 1.2, -4],
    modelName: 'EfficientNet-B0 + GradCAM',
    techDetails: {
      backbone: 'Pretrained EfficientNet-B0 with custom 2-class classifier head',
      inputSpec: 'RGB Frame, 224 × 224 pixels, ImageNet normalized',
      targetArtifacts: 'Pixel boundary blending, facial landmark warping, frequency inconsistencies',
      productionStatus: 'Trained checkpoint loaded (models/spatial_agent.pth)'
    },
    summary: 'Analyzes individual video frames and high-resolution still images for sub-pixel boundary warping, GAN blending seams, and unnatural texture discontinuities using gradient-weighted class activation mapping (GradCAM).',
    detectionPrinciples: [
      'Extracts highest-entropy middle frames from video containers',
      'Computes spatial gradient activation maps highlighting manipulation loci',
      'Outputs calibrated fake probability against 0.50 decision threshold'
    ]
  },
  temporal: {
    id: 'temporal',
    name: 'Temporal Analyst',
    tagline: 'Spatio-Temporal Coherence & Inter-Frame Dynamics',
    badge: 'Temporal Agent',
    accentColor: '#3b82f6',
    position: [6, 0, -4],
    cameraPosition: [6, 2.2, -0.6],
    targetLookAt: [6, 1.2, -4],
    modelName: 'R3D-18 (3D Video ResNet)',
    techDetails: {
      backbone: '18-layer 3D Spatio-Temporal Convolutional Network',
      inputSpec: '16 sampled frames across 2.0s window, 112 × 112 pixels',
      targetArtifacts: 'Inter-frame flickering, motion jitter, expression phase glitches',
      productionStatus: 'Trained checkpoint loaded (models/temporal_agent_checkpoint.pth)'
    },
    summary: 'Captures continuous 3D video tensors across 16 sequential frames to detect temporal micro-flickering, artificial motion dynamics, and unnatural biometric transitions between facial keypoints.',
    detectionPrinciples: [
      'Evaluates 3D spatio-temporal convolutions across time dimension',
      'Trained on identity-aware FakeAVCeleb splits',
      'Operates with validation-calibrated decision threshold of 0.05'
    ]
  },
  audio: {
    id: 'audio',
    name: 'Audio Analyst',
    tagline: 'Acoustic Synthesis & Voice Manipulation Forensic',
    badge: 'Audio Agent',
    accentColor: '#10b981',
    position: [-6, 0, 4],
    cameraPosition: [-6, 2.2, 7.4],
    targetLookAt: [-6, 1.2, 4],
    modelName: 'Mel-Spectrogram CNN (EfficientNet-B0)',
    techDetails: {
      backbone: 'Log-Mel Spectrogram Feature Extractor + CNN Classifier',
      inputSpec: '16 kHz mono WAV, 128 mel-bands, 224 × 224 tensor',
      targetArtifacts: 'Phase vocoder artifacts, synthetic robotic timbre, vocoder harmonics',
      productionStatus: 'Trained checkpoint loaded (models/audio_agent_checkpoint.pth)'
    },
    summary: 'Probes media containers for audio streams, converts voice signals to high-resolution log-mel spectrograms, and detects synthetic speech vocoder artifacts from modern generative audio engines.',
    detectionPrinciples: [
      'Identifies synthetic voice cloning and acoustic vocoder frequency anomalies',
      'Gracefully flags containers without audio streams as "no_audio"',
      'Safely excluded from fusion denominator when audio track is absent'
    ]
  },
  context: {
    id: 'context',
    name: 'Context Analyst',
    tagline: 'Cross-Modal Synchrony & Metadata Verification',
    badge: 'Context Agent',
    accentColor: '#a855f7',
    position: [6, 0, 4],
    cameraPosition: [6, 2.2, 7.4],
    targetLookAt: [6, 1.2, 4],
    modelName: 'Cross-Modal Explanation (no model)',
    techDetails: {
      backbone: 'Rule-based & multi-modal correlation verification',
      inputSpec: 'Visual prediction, Audio prediction, container stream metadata',
      targetArtifacts: 'Real-video/fake-audio mismatch, audio-visual divergence',
      productionStatus: 'Deterministic cross-modal verification logic'
    },
    summary: 'Describes whether the visual and audio scores agree or point in different directions. It restates the other agents’ outputs rather than adding new evidence, so it never votes and never changes the verdict.',
    detectionPrinciples: [
      'Explanation only: no independent model or evidence',
      'Shown only when both visual and audio scores exist',
      'Never changes the fused score or verdict'
    ]
  },
  fusion: {
    id: 'fusion',
    name: 'Fusion Core',
    tagline: 'Multi-Agent Evidence Aggregation Engine',
    badge: 'Production Fusion',
    accentColor: '#cc785c',
    position: [0, 1.2, 0],
    cameraPosition: [0, 2.8, 5.0],
    targetLookAt: [0, 1.2, 0],
    modelName: 'corrected_fixed (Dynamic Denominator Fixed-Weight Fusion)',
    techDetails: {
      backbone: 'Dynamic-denominator normalized multi-agent weighting',
      inputSpec: 'Visual = Spatial 0.40 : Temporal 0.30; fused = 1 − (1 − visual)(1 − audio)',
      targetArtifacts: 'Score dilution, false authenticity on missing streams',
      productionStatus: 'Production Default: corrected_fixed (PPO v4 experimental/disabled)'
    },
    summary: 'The central intelligence nexus of VERITAS. Fuses evidence from all contributing neural agents using dynamic-denominator normalization, guaranteeing that missing modalities never artificially lower the fake score or generate false authentic decisions.',
    detectionPrinciples: [
      'Normalizes weights exclusively across active, trained modalities',
      'Strict three-state decision contract: AUTHENTIC, DEEPFAKE, or UNCERTAIN',
      'Enforces an explicit abstention deadband rather than guessing on ambiguous evidence'
    ]
  }
};

export const OVERVIEW_CAMERA = {
  position: [0, 6.5, 14.5] as [number, number, number],
  lookAt: [0, 1.0, 0] as [number, number, number]
};
