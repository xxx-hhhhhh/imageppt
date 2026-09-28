export type ElementType = 'text' | 'image' | 'rectangle' | 'roundedRectangle' | 'ellipse' | 'line' | 'arrow' | 'group' | 'background';
export type TextAlign = 'left' | 'center' | 'right';

export interface ElementStyle {
  fontFamily?: string;
  fontSize?: number;
  fontWeight?: number;
  fontStyle?: 'normal' | 'italic';
  color?: string;
  align?: TextAlign;
  fill?: string;
  stroke?: string;
  strokeWidth?: number;
  opacity?: number;
  fontClass?: string;
  lineSpacing?: number;
  letterSpacing?: number;
  verticalAlign?: 'top' | 'middle' | 'bottom';
  [key: string]: unknown;
}

export interface LayoutElement {
  id: string;
  type: ElementType;
  x: number;
  y: number;
  width: number;
  height: number;
  rotation: number;
  zIndex: number;
  text?: string | null;
  src?: string | null;
  crop?: { left?: number; right?: number; top?: number; bottom?: number };
  style?: ElementStyle;
  confidence?: number | null;
  metadata?: Record<string, unknown>;
  groupId?: string;
  role?: string;
  componentType?: string;
  lines?: { text: string; bbox?: number[]; baseline?: number; height?: number }[];
}

export interface LayoutJSON {
  version: string;
  slide: { width: number; height: number };
  source?: string | null;
  backgroundUrl?: string | null;
  coordinateSystem?: string;
  metadata?: { planCoverage?: { status?: 'partial' | 'complete'; textCoverage?: number; uncoveredTextIds?: string[] }; [key: string]: unknown };
  elements: LayoutElement[];
}

export interface ProjectInfo {
  id: string;
  name: string;
  createdAt: string;
  imageCount: number;
}

export interface AnalyzeResponse {
  project: ProjectInfo;
  slides: LayoutJSON[];
  provider: string;
  ocrProvider?: string;
  visionProvider?: string;
  aiUsed?: boolean;
  visionModel?: string | null;
  requestedVisionProvider?: string | null;
  fallbackCount?: number;
  visionWarning?: string | null;
  warnings: string[];
}

export interface PagePreview {
  id: string;
  originalPreviewUrl: string;
  resultPreviewUrl: string | null;
  status: 'waiting' | 'ready';
}

export interface RevisionResponse {
  layout: LayoutJSON;
  revisionRound: number;
  accepted: boolean;
  issuesBefore: { elementId?: string; problem: string }[];
  issuesAfter: { elementId?: string; problem: string }[];
  improvedRegions: string[];
  visualBefore: number;
  visualAfter: number;
  editableCoverageBefore: number;
  editableCoverageAfter: number;
  stagnationReason?: string | null;
  assetsBefore: number;
  assetsAfter: number;
  missingAssetCount: number;
  preservedAssetCount: number;
  replacedAssetCount: number;
  inpaintedRegions: number;
  protectedRegions: number;
  rollbackTriggered: boolean;
  regionalAnalysis?: { elementId?: string; problem: string; bbox: number[]; pixelDifference: number; edgeDifference: number; strategyChange: string }[];
}
