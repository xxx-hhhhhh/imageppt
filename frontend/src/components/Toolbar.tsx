import type { ElementType } from '../types/layout';
import type { VisionStatus } from '../services/api';

interface Props {
  busy: boolean;
  conversionMode: 'fast' | 'standard' | 'high_quality' | 'maximum';
  viewMode: 'reconstruction' | 'original' | 'difference';
  onModeChange: (mode: 'fast' | 'standard' | 'high_quality' | 'maximum') => void;
  visionStatus: VisionStatus | null;
  onOpenSettings: () => void;
  onViewChange: (mode: 'reconstruction' | 'original' | 'difference') => void;
  onUpload: (files: File[]) => void;
  onAnalyze: () => void;
  onExport: () => void;
  onAddShape: (type: ElementType) => void;
  onDelete: () => void;
  onCopy: () => void;
  onLayer: (direction: 'up' | 'down' | 'top' | 'bottom') => void;
}

export function Toolbar({ busy, conversionMode, viewMode, visionStatus, onModeChange, onViewChange, onOpenSettings, onUpload, onAnalyze, onExport, onAddShape, onDelete, onCopy, onLayer }: Props) {
  return <header className="toolbar">
    <div className="brand"><span className="brand-mark">I</span><span>Image2EditablePPT</span></div>
    <label className="toolbar-button primary">上传图片<input hidden type="file" multiple accept="image/png,image/jpeg,image/webp" onChange={(event) => { if (event.target.files) onUpload(Array.from(event.target.files)); event.currentTarget.value = ''; }} /></label>
    <button className="toolbar-button" disabled={busy} onClick={onAnalyze}>AI解析</button>
    <button className="toolbar-button ai-settings-button" onClick={onOpenSettings}>AI设置 / API Key</button>
    <select className="mode-select" value={conversionMode} onChange={(event) => onModeChange(event.target.value as Props['conversionMode'])} disabled={busy} aria-label="转换模式"><option value="maximum">最高质量</option><option value="high_quality">高精度</option><option value="standard">标准</option><option value="fast">快速</option></select>
    <span className="vision-status" title={visionStatus?.error || ''}>AI视觉理解：{visionStatus?.configured ? '● 已连接' : '○ 未配置'}{visionStatus?.configured ? ` · ${visionStatus.model || '视觉模型'}` : ''}</span>
    <div className="view-switch" role="group" aria-label="对照视图"><button className={viewMode === 'reconstruction' ? 'selected' : ''} onClick={() => onViewChange('reconstruction')}>重建</button><button className={viewMode === 'original' ? 'selected' : ''} onClick={() => onViewChange('original')}>原图</button><button className={viewMode === 'difference' ? 'selected' : ''} onClick={() => onViewChange('difference')}>对比</button></div>
    <button className="toolbar-button export" disabled={busy} onClick={onExport}>导出PPT</button>
    <span className="toolbar-divider" />
    <button className="icon-button" title="矩形" onClick={() => onAddShape('rectangle')}>矩形</button>
    <button className="icon-button" title="圆角矩形" onClick={() => onAddShape('roundedRectangle')}>圆角</button>
    <button className="icon-button" title="椭圆" onClick={() => onAddShape('ellipse')}>椭圆</button>
    <button className="icon-button" title="线条" onClick={() => onAddShape('line')}>线条</button>
    <button className="icon-button" title="箭头" onClick={() => onAddShape('arrow')}>箭头</button>
    <button className="icon-button" title="复制" onClick={onCopy}>复制</button>
    <button className="icon-button danger" title="删除" onClick={onDelete}>删除</button>
    <span className="layer-tools">
      <button className="icon-button" onClick={() => onLayer('up')}>上移</button>
      <button className="icon-button" onClick={() => onLayer('down')}>下移</button>
      <button className="icon-button" onClick={() => onLayer('top')}>置顶</button>
      <button className="icon-button" onClick={() => onLayer('bottom')}>置底</button>
    </span>
  </header>;
}
