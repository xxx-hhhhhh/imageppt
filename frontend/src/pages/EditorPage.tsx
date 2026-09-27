import { useEffect, useState } from 'react';
import axios from 'axios';
import { AISettingsModal } from '../components/AISettingsModal';
import { EditorCanvas } from '../editor/EditorCanvas';
import { PageList } from '../components/PageList';
import { PropertyPanel } from '../components/PropertyPanel';
import { Toolbar } from '../components/Toolbar';
import { artifactUrl, getLocalInpaintStatus, getVisionStatus, getVisualScore, type VisionStatus } from '../services/api';
import { useProjectStore } from '../stores/useProjectStore';
import type { LayoutJSON } from '../types/layout';

export function EditorPage() {
  const [error, setError] = useState('');
  const [editing, setEditing] = useState(false);
  const [visionStatus, setVisionStatus] = useState<VisionStatus | null>(null);
  const [localInpaint, setLocalInpaint] = useState<{ enabled: boolean; connected: boolean } | null>(null);
  const [localFallbacks, setLocalFallbacks] = useState(0);
  const [localSuccesses, setLocalSuccesses] = useState(0);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [similarity, setSimilarity] = useState<number | null>(null);
  const [actualVision, setActualVision] = useState<string | null>(null);
  const [professionalInpainting, setProfessionalInpainting] = useState<string | null>(null);
  const [aiBackgroundRepairs, setAiBackgroundRepairs] = useState<number | null>(null);
  const [revisionStatus, setRevisionStatus] = useState('');
  const [revisionRound, setRevisionRound] = useState(0);
  const [stagnationReason, setStagnationReason] = useState<string | null>(null);
  const [dismissedDecision, setDismissedDecision] = useState('');
  const { project, slides, activePage, selectedIds, busy, message, aiPaused, reviewVersion, approvedPages, create, upload, analyze, approveAndNext, useBasicFallback, reanalyzeCurrent, downgradeCurrent, acceptCurrent, persistPage, updateActive, setSelection, setActivePage, moveLayer, export: exportProject } = useProjectStore();

  useEffect(() => { if (!project) void create(); }, [project, create]);
  useEffect(() => { void getVisionStatus().then(setVisionStatus).catch(() => setVisionStatus(null)); }, []);
  useEffect(() => { void getLocalInpaintStatus().then(setLocalInpaint).catch(() => setLocalInpaint(null)); }, [reviewVersion]);
  useEffect(() => {
      if (!project || !slides.length) { setSimilarity(null); setActualVision(null); setProfessionalInpainting(null); setAiBackgroundRepairs(null); setLocalFallbacks(0); setLocalSuccesses(0); setRevisionStatus(''); setRevisionRound(0); setStagnationReason(null); return; }
    void getVisualScore(project.id, activePage + 1).then((score) => { setSimilarity(typeof score.overall === 'number' ? score.overall : null); setActualVision(score.visionProvider === 'qwen' ? score.visionModel || '千问视觉' : '本地解析'); setProfessionalInpainting(typeof score.professionalInpaintingAttempts === 'number' ? `${score.inpaintingProvider || '专业修图'} 尝试 ${score.professionalInpaintingAttempts} 次，待修 ${score.professionalRepairPending || 0} 处` : null); setAiBackgroundRepairs(typeof score.aiBackgroundRepairs === 'number' ? score.aiBackgroundRepairs : null); setLocalFallbacks(score.localInpaintFallbacks || 0); setLocalSuccesses(score.localInpaintSuccesses || 0); setRevisionStatus(score.revisionStatus || ''); setRevisionRound(score.revisionRound || 0); setStagnationReason(score.stagnationReason || null); }).catch(() => { setSimilarity(null); setActualVision(null); setProfessionalInpainting(null); setAiBackgroundRepairs(null); setLocalFallbacks(0); setLocalSuccesses(0); setRevisionStatus(''); });
  }, [project?.id, slides.length, activePage, reviewVersion]);
  const page = slides[activePage];
  const handle = async (operation: () => Promise<void>) => { try { setError(''); await operation(); } catch (err) { const detail = axios.isAxiosError(err) ? err.response?.data?.detail : null; setError(typeof detail?.message === 'string' ? detail.message : typeof detail === 'string' ? detail : err instanceof Error ? err.message : '操作失败'); } };
  const decisionKey = `${project?.id || ''}:${activePage}:${revisionRound}`;
  const handleDecision = async (operation: () => Promise<void>) => {
    setDismissedDecision(decisionKey);
    try { await operation(); } catch (err) { setDismissedDecision(''); const detail = axios.isAxiosError(err) ? err.response?.data?.detail : null; setError(typeof detail?.message === 'string' ? detail.message : typeof detail === 'string' ? detail : err instanceof Error ? err.message : '操作失败'); }
  };
  const saveCurrent = (next: LayoutJSON) => { updateActive(() => next); };
  const exportPpt = async () => { const url = await exportProject(); window.open(url, '_blank'); };
  const approved = approvedPages.includes(activePage + 1);
  const reviewing = Boolean(page && !approved && !editing);
  const originalFile = activePage === 0 ? 'original.png' : `original_${activePage + 1}.png`;

  return <div className="app-shell">
    <Toolbar busy={busy} hasImages={Boolean(project?.imageCount)} hasPage={Boolean(page)} visionStatus={visionStatus} onOpenSettings={() => setSettingsOpen(true)} onUpload={(files) => void handle(() => upload(files))} onAnalyze={() => void handle(analyze)} onEdit={() => setEditing(true)} onExport={() => void handle(exportPpt)} />
    <div className="workspace">
      <PageList slides={slides} activePage={activePage} approvedPages={approvedPages} onSelect={(index) => { setEditing(false); setActivePage(index); }} />
      <main className="editor-area">
        <div className="status-row"><span>{page ? `第 ${activePage + 1} / ${project?.imageCount || slides.length} 页` : project?.name || 'Image2EditablePPT'}</span><span className={busy ? 'status busy' : 'status'} role="status">{error || message}</span><span className="inpaint-status" title={localInpaint?.connected ? '本地修图服务可用' : '本地修图不可用，自动使用原有修复方式'}>{localFallbacks > 0 ? `LaMa 已回退 ${localFallbacks} 次` : localSuccesses > 0 ? `LaMa 已修复 ${localSuccesses} 次` : localInpaint?.connected ? '本地 LaMa 已连接' : '本地 LaMa 未连接'}</span></div>
        {aiPaused && !busy && <div className="review-actions" role="alert"><span>AI 未完成本页规划，处理已暂停。</span><button onClick={() => void handle(analyze)}>重试 AI</button><button onClick={() => void handle(useBasicFallback)}>使用基础模式</button></div>}
        {reviewing && revisionStatus === 'stagnated' && !busy && dismissedDecision !== decisionKey && <div className="review-actions" role="alert"><span>局部优化已停滞{stagnationReason === 'no_targetable_issues' ? '：没有新的可修复区域' : '，请选择下一步'}。</span><button onClick={() => void handleDecision(reanalyzeCurrent)}>继续自动优化</button><button onClick={() => void handleDecision(downgradeCurrent)}>图片主体＋可编辑文字</button><button onClick={() => void handleDecision(acceptCurrent)}>使用当前结果</button></div>}
        {reviewing && !busy && <div className="review-actions page-decision"><span>对照原图检查本页，确认后继续下一页</span><div><button className="decision-primary" onClick={() => void handle(approveAndNext)}>通过本页</button><button onClick={() => void handle(reanalyzeCurrent)}>继续自动优化</button><button onClick={() => setEditing(true)}>手动编辑</button></div></div>}
        {page ? reviewing && project ? <div className="review-stage"><section className="review-pane"><h2>原图</h2><div className="original-frame"><img src={artifactUrl(project.id, originalFile)} alt={`第 ${activePage + 1} 页原图`} /></div></section><section className="review-pane"><h2>当前重建结果</h2><EditorCanvas page={page} selectedIds={selectedIds} onSelection={setSelection} onChange={saveCurrent} /></section></div> : <><div className="editing-heading"><span>{approved ? '本页已通过' : '手动编辑'}</span>{!approved && <button onClick={() => setEditing(false)}>返回逐页确认</button>}</div><EditorCanvas page={page} selectedIds={selectedIds} onSelection={setSelection} onChange={saveCurrent} /></> : <div className="welcome"><div className="welcome-icon">P</div><h1>图片转可编辑 PPT</h1><p>上传 JPG、JPEG、PNG 或 WEBP，开始逐页转换与确认。</p><label className="dropzone">选择图片<input hidden type="file" multiple accept="image/png,image/jpeg,image/webp" onChange={(event) => { const files = event.currentTarget.files; if (files) void handle(() => upload(Array.from(files))); event.currentTarget.value = ''; }} /></label></div>}
        <div className="editor-footer"><span>OCR: RapidOCR · 已配置视觉模型：{visionStatus?.configured && visionStatus.model ? visionStatus.model : '本地模式'} · 本页实际：{actualVision || '待解析'}{professionalInpainting ? ` · 复杂背景修复：${professionalInpainting}，成功 ${aiBackgroundRepairs || 0} 处` : ''} · 视觉相似度：{similarity === null ? '—' : `${Math.round(similarity * 100)}%`}</span><button onClick={() => page && void handle(() => persistPage(activePage, page))} disabled={!page || busy}>保存当前页面</button></div>
      </main>
      <PropertyPanel page={page} selectedIds={selectedIds} onLayer={moveLayer} onChange={(updater) => { if (page) updateActive(updater); }} />
    </div>
    <AISettingsModal open={settingsOpen} onClose={() => setSettingsOpen(false)} onSaved={() => void getVisionStatus().then(setVisionStatus).catch(() => undefined)} />
  </div>;
}
