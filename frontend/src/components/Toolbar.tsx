import type { VisionStatus } from '../services/api';

interface Props {
  busy: boolean;
  hasImages: boolean;
  hasPage: boolean;
  visionStatus: VisionStatus | null;
  onOpenSettings: () => void;
  onUpload: (files: File[]) => void;
  onAnalyze: () => void;
  onEdit: () => void;
  onExport: () => void;
}

export function Toolbar({ busy, hasImages, hasPage, visionStatus, onOpenSettings, onUpload, onAnalyze, onEdit, onExport }: Props) {
  return <header className="toolbar">
    <div className="brand"><span className="brand-mark" aria-hidden="true">I</span><span>Image2EditablePPT</span></div>
    <nav className="toolbar-workflow" aria-label="主要操作">
      <label className={`toolbar-button upload-button ${busy ? 'disabled' : ''}`} role="button" tabIndex={busy ? -1 : 0} onKeyDown={(event) => { if (!busy && (event.key === 'Enter' || event.key === ' ')) { event.preventDefault(); event.currentTarget.querySelector('input')?.click(); } }}>上传图片<input hidden disabled={busy} type="file" multiple accept="image/png,image/jpeg,image/webp" onChange={(event) => { if (event.target.files) onUpload(Array.from(event.target.files)); event.currentTarget.value = ''; }} /></label>
      <button className="toolbar-button primary" disabled={busy || !hasImages} onClick={onAnalyze}>AI解析 / 开始转换</button>
      <button className="toolbar-button" onClick={onOpenSettings}>AI设置</button>
      <button className="toolbar-button" disabled={!hasPage || busy} onClick={onEdit}>手动编辑</button>
      <button className="toolbar-button export" disabled={!hasPage || busy} onClick={onExport}>导出PPT</button>
    </nav>
    <div className="toolbar-meta" aria-label="转换状态"><span className="quality-badge">最高质量</span><span className="vision-status" title={visionStatus?.error || ''}>{visionStatus?.configured ? 'AI 已配置' : 'AI 未配置'}</span></div>
  </header>;
}
