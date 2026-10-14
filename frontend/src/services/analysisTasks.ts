import { api } from './api';
import type { LayoutJSON, ProjectInfo } from '../types/layout';

export interface AnalysisTask {
  id: string;
  projectId: string;
  page: number;
  mode: 'fast' | 'standard' | 'high_quality' | 'maximum';
  status: 'running' | 'publishing' | 'succeeded' | 'failed' | 'cancelled' | 'timed_out' | 'interrupted';
  stage: string;
  stageLabel: string;
  elapsedSeconds: number;
  deadlineSeconds: number;
  message: string;
  errorCode?: string;
}
export interface RestoredProject {
  project: ProjectInfo;
  slides: LayoutJSON[];
  approvedPages: number[];
  images: { id: string; source_url: string }[];
  task: AnalysisTask | null;
}
export const taskIsActive = (task: AnalysisTask | null) => !!task && ['running', 'publishing'].includes(task.status);
const options = { timeout: 15000 };
export async function restoreProject(id: string): Promise<RestoredProject> {
  return (await api.get<RestoredProject>(`/projects/${id}`, options)).data;
}
export async function startTask(id: string, page: number, mode: AnalysisTask['mode'], allowFallback = false): Promise<AnalysisTask> {
  return (await api.post<AnalysisTask>(`/projects/${id}/analysis-tasks`, { page, mode, allowFallback }, options)).data;
}
export async function readTask(id: string, taskId: string): Promise<AnalysisTask> {
  return (await api.get<AnalysisTask>(`/projects/${id}/analysis-tasks/${taskId}`, options)).data;
}
export async function cancelTask(id: string, taskId: string): Promise<AnalysisTask> {
  return (await api.post<AnalysisTask>(`/projects/${id}/analysis-tasks/${taskId}/cancel`, null, options)).data;
}
