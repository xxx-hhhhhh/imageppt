import { useEffect, useState } from 'react';
import axios from 'axios';
import { AISettingsModal } from '../components/AISettingsModal';
import { EditorCanvas } from '../editor/EditorCanvas';
import { PageList } from '../components/PageList';
import { PropertyPanel } from '../components/PropertyPanel';
import { Toolbar } from '../components/Toolbar';
import { getLocalInpaintStatus, getVisionStatus, getVisualScore, type VisionStatus } from '../services/api';
import { useProjectStore } from '../stores/useProjectStore';
import type { LayoutJSON } from '../types/layout';

type ViewMode = 'result' | 'original' | 'compare';

export function EditorPage() {
  const [error, setError] = useState('');
  const [editing, setEditing] = useState(false);
  const [viewMode, setViewMode] = useState<ViewMode>('result');
  const [zoomFactor, setZoomFactor] = useState(1);
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
  const [reviewAction, setReviewAction] = useState('');
  const { project, slides, pagePreviews, activePage, selectedIds, busy, message, aiPaused, reviewVersion, approvedPages, create, upload, analyze, approveAndNext, useBasicFallback, reanalyzeCurrent, downgradeCurrent, acceptCurrent, persistPage, updateActive, setSelection, setActivePage, moveLayer, export: exportProject } = useProjectStore();

  useEffect(() => { if (!project) void create(); }, [project, create]);
  useEffect(() => { void getVisionStatus().then(setVisionStatus).catch(() => setVisionStatus(null)); }, []);
  useEffect(() => { void getLocalInpaintStatus().then(setLocalInpaint).catch(() => setLocalInpaint(null)); }, [reviewVersion]);
  useEffect(() => {
      if (!project || !slides.length) { setSimilarity(null); setActualVision(null); setProfessionalInpainting(null); setAiBackgroundRepairs(null); setLocalFallbacks(0); setLocalSuccesses(0); setRevisionStatus(''); setRevisionRound(0); setStagnationReason(null); return; }
    void getVisualScore(project.id, activePage + 1).then((score) => { setSimilarity(typeof score.overall === 'number' ? score.overall : null); setActualVision(score.visionProvider === 'qwen' ? score.visionModel || '千问视觉' : '本地解析'); setProfessionalInpainting(typeof score.professionalInpaintingAttempts === 'number' ? `${score.inpaintingProvider || '专业修图'} 尝试 ${score.professionalInpaintingAttempts} 次，待修 ${score.professionalRepairPending || 0} 处` : null); setAiBackgroundRepairs(typeof score.aiBackgroundRepairs === 'number' ? score.aiBackgroundRepairs : null); setLocalFallbacks(score.localInpaintFallbacks || 0); setLocalSuccesses(score.localInpaintSuccesses || 0); setRevisionStatus(score.revisionStatus || ''); setRevisionRound(score.revisionRound || 0); setStagnationReason(score.stagnationReason || null); }).catch(() => { setSimilarity(null); setActualVision(null); setProfessionalInpainting(null); setAiBackgroundRepairs(null); setLocalFallbacks(0); setLocalSuccesses(0); setRevisionStatus(''); });
  }, [project?.id, slides.length, activePage, reviewVersion]);
  const page = slides[activePage];
  const actualOCR = typeof page?.metadata?.ocrProvider === 'string' ? page.metadata.ocrProvider : '待解析';
  const partialPlan = page?.metadata?.planCoverage?.status === 'partial' ? page.metadata.planCoverage : null;
  const pagePreview = pagePreviews[activePage];
  const handle = async (operation: () => Promise<void>) => { try { setError(''); await operation(); } catch (err) { const detail = axios.isAxiosError(err) ? err.response?.data?.detail : null; setError(typeof detail?.message === 'string' ? detail.message : typeof detail === 'string' ? detail : err instanceof Error ? err.message : '操作失败'); } };
  const decisionKey = `${project?.id || ''}:${activePage}:${revisionRound}`;
  const handleDecision = async (operation: () => Promise<void>, label = '正在优化第 ' + (revisionRound + 1) + ' 轮…') => {
    setDismissedDecision(decisionKey);
    setReviewAction(label);
    try { await operation(); } catch (err) { setDismissedDecision(''); const detail = axios.isAxiosError(err) ? err.response?.data?.detail : null; setError(typeof detail?.message === 'string' ? detail.message : typeof detail === 'string' ? detail : err instanceof Error ? err.message : '操作失败'); } finally { setReviewAction(''); }
  };
  const approvePage = async () => { setReviewAction(`正在确认第 ${activePage + 1} 页…`); try { await handle(approveAndNext); } finally { setReviewAction(''); } };
  const saveCurrent = (next: LayoutJSON) => { updateActive(() => next); };
  const exportPpt = async () => { const url = await exportProject(); window.open(url, '_blank'); };
  const approved = approvedPages.includes(activePage + 1);
  const reviewing = Boolean(page && !approved && !editing);
  const originalPane = pagePreview && <section className="review-pane"><h2>原图</h2><div className="original-frame"><img src={pagePreview.originalPreviewUrl} alt={`第 ${activePage + 1} 页原图`} style={{ width: `${zoomFactor * 100}%`, height: `${zoomFactor * 100}%`, maxWidth: 'none', maxHeight: 'none' }} /></div></section>;
  const resultPane = <section className="review-pane"><h2>当前重建结果</h2>{page ? <EditorCanvas page={page} zoomFactor={zoomFactor} selectedIds={selectedIds} onSelection={setSelection} onChange={saveCurrent} /> : <div className="result-pending" role="status"><span className={busy ? 'processing-spinner' : 'result-placeholder-icon'} aria-hidden="true">{busy ? '' : '◇'}</span><strong>{busy ? '正在重建本页' : '本页尚未生成结果'}</strong><span>{busy ? '请稍候，完成后将显示当前重建结果。' : '点击顶部“AI解析 / 开始转换”后查看结果。'}</span></div>}</section>;

  return <div className="app-shell">
    <Toolbar busy={busy} hasImages={Boolean(project?.imageCount)} hasPage={Boolean(page)} visionStatus={visionStatus} onOpenSettings={() => setSettingsOpen(true)} onUpload={(files) => void handle(() => upload(files))} onAnalyze={() => void handle(analyze)} onEdit={() => { setEditing(true); setViewMode('result'); }} onExport={() => void handle(exportPpt)} />
    <div className="workspace">
      <PageList pages={pagePreviews} activePage={activePage} approvedPages={approvedPages} busy={busy} onAdd={(files) => void handle(() => upload(files))} onSelect={(index) => { setEditing(false); setActivePage(index); }} />
      <main className="editor-area">
        <div className="workspace-heading"><div><span className="eyebrow">AI PRESENTATION STUDIO</span><h1>{pagePreview ? `第 ${activePage + 1} 页 · ${page ? editing ? '手动编辑' : approved ? '已通过' : '逐页确认' : '待解析'}` : '创作工作台'}</h1></div><div className="workspace-chips"><span className="quality-badge">最高质量</span><span className="inpaint-status" title={localInpaint?.connected ? '本地修图服务可用' : '本地修图不可用，自动使用原有修复方式'}>{localFallbacks > 0 ? `LaMa 已回退 ${localFallbacks} 次` : localSuccesses > 0 ? `LaMa 已修复 ${localSuccesses} 次` : localInpaint?.connected ? '本地 LaMa 已连接' : '本地 LaMa 未连接'}</span></div></div>
        <div className="status-row"><span className={busy ? 'status busy' : 'status'} role="status">{reviewAction || error || message}</span>{partialPlan && <span className="quality-badge" role="status">AI 模块已规划 · {partialPlan.uncoveredTextIds?.length || 0} 条文字由 OCR 补齐</span>}{pagePreview && <span className="page-counter">{String(activePage + 1).padStart(2, '0')} / {String(project?.imageCount || pagePreviews.length).padStart(2, '0')}</span>}</div>
        {pagePreview && <div className="view-toolbar"><div className="view-segments" role="group" aria-label="查看模式"><button type="button" className={viewMode === 'original' ? 'selected' : ''} aria-pressed={viewMode === 'original'} onClick={() => setViewMode('original')}>原图</button><button type="button" className={viewMode === 'result' ? 'selected' : ''} aria-pressed={viewMode === 'result'} onClick={() => setViewMode('result')}>结果</button><button type="button" className={viewMode === 'compare' ? 'selected' : ''} aria-pressed={viewMode === 'compare'} onClick={() => setViewMode('compare')}>对比查看</button></div><div className="zoom-controls" role="group" aria-label="画布缩放"><button type="button" aria-label="缩小" disabled={zoomFactor <= 0.5} onClick={() => setZoomFactor((value) => Math.max(0.5, Math.round((value - .25) * 100) / 100))}>−</button><span>{Math.round(zoomFactor * 100)}%</span><button type="button" aria-label="放大" disabled={zoomFactor >= 2} onClick={() => setZoomFactor((value) => Math.min(2, Math.round((value + .25) * 100) / 100))}>＋</button></div></div>}
        {aiPaused && !busy && <div className="review-actions" role="alert"><span>{error || message || 'AI 处理已暂停，请重试。'}</span><button onClick={() => void handle(analyze)}>重试 AI</button><button onClick={() => void handle(useBasicFallback)}>使用基础模式</button></div>}
        {reviewing && revisionStatus === 'stagnated' && !busy && dismissedDecision !== decisionKey && <div className="review-actions" role="alert"><span>局部优化已停滞{stagnationReason === 'no_targetable_issues' ? '：没有新的可修复区域' : '，请选择下一步'}。</span><button onClick={() => void handleDecision(reanalyzeCurrent)}>继续自动优化</button><button onClick={() => void handleDecision(downgradeCurrent, '正在调整问题区域…')}>图片主体＋可编辑文字</button><button onClick={() => void handleDecision(acceptCurrent, '正在保留当前结果…')}>使用当前结果</button></div>}
        {reviewing && !busy && (revisionStatus !== 'stagnated' || dismissedDecision === decisionKey) && <div className="review-actions page-decision"><span>检查当前结果，可切换到对比查看</span><div><button className="decision-primary" onClick={() => void approvePage()}>通过本页</button><button onClick={() => void handleDecision(reanalyzeCurrent)}>继续自动优化</button><button onClick={() => { setEditing(true); setViewMode('result'); }}>手动编辑</button></div></div>}
        {reviewing && busy && <div className="review-processing" role="status"><span className="processing-spinner" aria-hidden="true" />{reviewAction || message}</div>}
        {pagePreview ? <>{editing && page && <div className="editing-heading"><span>手动编辑当前结果</span>{!approved && <button onClick={() => setEditing(false)}>返回逐页确认</button>}</div>}<div className={`review-stage ${viewMode === 'compare' ? 'compare-mode' : 'single-mode'}`}>{viewMode !== 'result' && originalPane}{viewMode !== 'original' && resultPane}</div></> : <div className="welcome"><div className="welcome-icon">P</div><h1>图片转可编辑 PPT</h1><p>上传 JPG、JPEG、PNG 或 WEBP，开始逐页转换与确认。</p><label className="dropzone">选择图片<input hidden type="file" multiple accept="image/png,image/jpeg,image/webp" onChange={(event) => { const files = event.currentTarget.files; if (files) void handle(() => upload(Array.from(files))); event.currentTarget.value = ''; }} /></label></div>}
        <div className="editor-footer"><span>OCR: {actualOCR} · 已配置视觉模型：{visionStatus?.configured && visionStatus.model ? visionStatus.model : '本地模式'} · 本页实际：{actualVision || '待解析'}{professionalInpainting ? ` · 复杂背景修复：${professionalInpainting}，成功 ${aiBackgroundRepairs || 0} 处` : ''} · 视觉相似度：{similarity === null ? '—' : `${Math.round(similarity * 100)}%`}</span><button onClick={() => page && void handle(() => persistPage(activePage, page))} disabled={!page || busy}>保存当前页面</button></div>
      </main>
      <PropertyPanel page={page} selectedIds={selectedIds} onLayer={moveLayer} onChange={(updater) => { if (page) updateActive(updater); }} />
    </div>
    <AISettingsModal open={settingsOpen} onClose={() => setSettingsOpen(false)} onSaved={() => void getVisionStatus().then(setVisionStatus).catch(() => undefined)} />
  </div>;
}
