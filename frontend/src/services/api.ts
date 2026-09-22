import axios from 'axios';
import type { AnalyzeResponse, LayoutJSON, ProjectInfo } from '../types/layout';

export const api = axios.create({ baseURL: '/api' });

export function assetUrl(value?: string | null): string {
  if (!value) return '';
  if (value.startsWith('http://') || value.startsWith('https://') || value.startsWith('data:')) return value;
  return value;
}

export async function createProject(name = 'Image2EditablePPT'): Promise<ProjectInfo> {
  const response = await api.post<ProjectInfo>('/projects', { name });
  return response.data;
}

export async function uploadImages(projectId: string, files: File[]): Promise<ProjectInfo> {
  const data = new FormData();
  files.forEach((file) => data.append('files', file));
  const response = await api.post<{ project: ProjectInfo }>(`/projects/${projectId}/images`, data);
  return response.data.project;
}

export async function analyzeProject(projectId: string, mode: 'fast' | 'standard' | 'high_quality' | 'maximum' = 'standard'): Promise<AnalyzeResponse> {
  const response = await api.post<AnalyzeResponse>(`/projects/${projectId}/analyze`, null, { params: { mode } });
  return response.data;
}

export interface VisionStatus {
  provider: string;
  configured: boolean;
  model?: string | null;
  error?: string;
}

export type VisionProviderId = 'qwen' | 'local';

export interface VisionProviderSettings {
  enabled: boolean;
  configured: boolean;
  apiKeyMasked: string;
  baseUrl: string;
  model: string;
}

export interface VisionSettings {
  enabled: boolean;
  selectedProvider: VisionProviderId;
  providers: { qwen: VisionProviderSettings };
  configured: boolean;
  mode: 'fast' | 'standard' | 'high';
  timeout: number;
  proxyMode: 'system' | 'manual' | 'none';
  manualProxy: string;
  network?: { systemProxy?: string; proxyUrl?: string; source?: string; status?: string; proxyTcp?: { status?: string; message?: string } };
}

export interface VisionTestResult {
  success: boolean;
  provider: string;
  model: string;
  message: string;
  stage?: 'auth' | 'model' | 'network' | 'request' | null;
  statusCode?: number | null;
  errorType?: string | null;
  errorCode?: string | null;
  connectionPath?: string | null;
  latencyMs?: number | null;
  diagnostics?: {
    tcp?: boolean;
    httpsReachable?: boolean;
    auth?: boolean;
    modelReachable?: boolean;
    visionRequest?: boolean;
    directTried?: boolean;
    proxyTried?: boolean;
    proxyTcp?: { status?: string; message?: string };
    qwenBaseUrl?: { status?: string; message?: string; httpStatus?: number };
    qwenApi?: { status?: string; message?: string; httpStatus?: number; errorType?: string; errorCode?: string };
    networkPath?: string;
    connectionPath?: string;
  };
}

export async function getVisionStatus(): Promise<VisionStatus> {
  const response = await api.get<VisionStatus>('/vision/status');
  return response.data;
}

export async function getVisionSettings(): Promise<VisionSettings> {
  const response = await api.get<VisionSettings>('/settings/vision');
  return response.data;
}

export async function saveVisionSettings(payload: { enabled: boolean; selectedProvider: 'qwen' | 'local'; mode: 'fast' | 'standard' | 'high'; timeout: number; proxyMode: 'system' | 'manual' | 'none'; manualProxy: string; providers: { qwen: { enabled: boolean; apiKey?: string; baseUrl: string; model: string; clearApiKey?: boolean } } }): Promise<VisionSettings> {
  const response = await api.put<VisionSettings>('/settings/vision', payload);
  return response.data;
}

export async function testVisionSettings(): Promise<VisionTestResult> {
  const response = await api.post<VisionTestResult>('/settings/vision/test');
  return response.data;
}

export async function getVisualScore(projectId: string): Promise<{ overall?: number }> {
  const response = await api.get<{ overall?: number }>(`/projects/${projectId}/artifacts/visual_score.json`);
  return response.data;
}

export function artifactUrl(projectId: string, fileName: string): string {
  return `/api/projects/${projectId}/artifacts/${fileName}`;
}

export async function saveSlide(projectId: string, page: number, layout: LayoutJSON): Promise<LayoutJSON> {
  const response = await api.put<LayoutJSON>(`/projects/${projectId}/slides/${page}`, layout);
  return response.data;
}

export async function exportPptx(projectId: string): Promise<{ downloadUrl: string; validation: Record<string, unknown> }> {
  const response = await api.post<{ downloadUrl: string; validation: Record<string, unknown> }>(`/projects/${projectId}/export/pptx`);
  return response.data;
}
