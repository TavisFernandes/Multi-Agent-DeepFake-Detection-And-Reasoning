import type { CSSProperties, HTMLAttributes, ReactElement } from 'react';

export interface GradientASCIIProps extends Omit<HTMLAttributes<HTMLDivElement>, 'style'> {
  colors?: string[];
  shape?: 'horizon' | 'pinch' | 'rise' | 'eclipse';
  characters?: string;
  cellSize?: number;
  density?: number;
  glyphOpacity?: number;
  gradientOpacity?: number;
  glow?: number;
  depthOfField?: number;
  flicker?: number;
  softness?: number;
  scale?: number;
  flow?: number;
  speed?: number;
  grain?: number;
  variation?: number;
  mouseInteraction?: boolean;
  scrambleRadius?: number;
  clickEffect?: 'wave' | 'burst' | 'none';
  intro?: boolean;
  lightMode?: boolean;
  fontFamily?: string;
  dpr?: number;
  style?: CSSProperties;
}

declare const GradientASCII: (props: GradientASCIIProps) => ReactElement;
export default GradientASCII;
