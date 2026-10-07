export type ElementType = 'text' | 'image' | 'rectangle' | 'roundedRectangle' | 'ellipse' | 'line' | 'arrow' | 'group' | 'background';
export type TextAlign = 'left' | 'center' | 'right';
export type ObjectOwner = 'editable_text' | 'native_shape' | 'movable_image' | 'background' | 'intentional_ignore';

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
  owner?: ObjectOwner;
  bbox?: { left: number; top: number; width: number; height: number };
  mask?: string;
  contour?: number[][][];
  parent?: string | null;
  source?: string;
  editable?: boolean;
  assetPath?: string | null;
  reconstructionStrategy?: string;
}

export interface LayoutJSON {
  version: string;
  slide: { width: number; height: number };
  source?: string | null;
  backgroundUrl?: string | null;
  previewUrl?: string | null;
  coordinateSystem?: string;
  elements: LayoutElement[];
  sceneVersion?: string;
  metadata?: Record<string, unknown>;
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
