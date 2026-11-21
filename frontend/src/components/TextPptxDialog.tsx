import { useEffect, useRef, useState } from 'react';
import type { FormEvent } from 'react';
import axios from 'axios';
import { generateTextPptx, type TextDeckStyle } from '../services/api';

interface Props { onClose: () => void; }

export function TextPptxDialog({ onClose }: Props) {
  const dialog = useRef<HTMLDialogElement>(null);
  const input = useRef<HTMLTextAreaElement>(null);
  const [content, setContent] = useState('');
  const [slideCount, setSlideCount] = useState(6);
  const [style, setStyle] = useState<TextDeckStyle>('business');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [result, setResult] = useState<{ file: Blob; slides: number }>();

  useEffect(() => {
    const node = dialog.current;
    const opener = document.activeElement;
    if (node && !node.open) node.showModal();
    input.current?.focus();
    return () => {
      node?.close();
      if (opener instanceof HTMLElement && opener.isConnected) opener.focus();
    };
  }, []);

  function download(file: Blob) {
    const url = URL.createObjectURL(file);
    const link = document.createElement('a');
    link.href = url;
    link.download = 'text-ppt-master.pptx';
    link.click();
    window.setTimeout(() => URL.revokeObjectURL(url), 1500);
  }

  async function generate(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (busy || !content.trim()) return;
    setBusy(true);
    setError('');
    setResult(undefined);
    try {
      const file = await generateTextPptx({ content: content.trim(), slideCount, style });
      setResult({ file, slides: slideCount });
      download(file);
    } catch (err) {
      let message = '生成未完成，请稍后重试。输入内容已保留。';
      if (axios.isAxiosError(err) && err.response?.data instanceof Blob) {
        const data = await err.response.data.text().then(text => JSON.parse(text)).catch(() => null);
        if (typeof data?.detail === 'string') message = data.detail;
      }
      setError(message);
    } finally {
      setBusy(false);
    }
  }

  return <dialog ref={dialog} className="ai-modal text-pptx-dialog" aria-labelledby="text-pptx-title" onCancel={event => { event.preventDefault(); if (!busy) onClose(); }}>
    <form onSubmit={event => void generate(event)} aria-busy={busy}>
      <div className="ai-modal-header"><h2 id="text-pptx-title">文字生成 PPT</h2><button type="button" className="modal-close" aria-label="关闭" disabled={busy} onClick={onClose}>×</button></div>
      <div className="ai-modal-body">
        <p className="field-hint">粘贴主题、提纲或正文。千问按 PPT Master 的规划指南分页，生成后可在 PowerPoint 中编辑。</p>
        <fieldset disabled={busy} onChange={() => { setResult(undefined); setError(''); }}>
          <label className="field"><span>主题或正文</span><textarea ref={input} aria-describedby="text-pptx-input-hint" required maxLength={12000} rows={8} value={content} onChange={event => setContent(event.target.value)} placeholder="例如：面向研究生介绍如何整理一份可复现的实验记录，内容包括实验设置、数据记录和复查步骤。" /></label>
          <p className="text-pptx-counter field-hint" id="text-pptx-input-hint">{content.length.toLocaleString()} / 12,000 字</p>
          <div className="fields-grid">
            <label className="field"><span>页数</span><select value={slideCount} onChange={event => setSlideCount(Number(event.target.value))}>{[3, 4, 5, 6, 7, 8, 9, 10, 11, 12].map(count => <option key={count} value={count}>{count} 页</option>)}</select></label>
            <label className="field"><span>风格</span><select value={style} onChange={event => setStyle(event.target.value as TextDeckStyle)}><option value="business">商务汇报</option><option value="academic">学术报告</option><option value="minimal">简洁演示</option></select></label>
          </div>
        </fieldset>
        <p className="field-hint">生成时，输入的文字会发送给已配置的千问服务。</p>
        {busy && <p className="ai-status" role="status">正在整理内容并生成 PPTX，请保留此窗口…</p>}
        {error && <p className="ai-error" role="alert">{error}</p>}
        {result && <p className="ai-status" role="status">已生成 {result.slides} 页可编辑 PPTX。<button type="button" className="toolbar-button export" onClick={() => download(result.file)}>重新下载</button></p>}
      </div>
      <div className="ai-modal-footer"><button type="button" className="toolbar-button" disabled={busy} onClick={onClose}>{result ? '关闭' : '取消'}</button><button type="submit" className="toolbar-button primary" disabled={busy || !content.trim()}>{busy ? '生成中…' : '生成可编辑 PPT'}</button></div>
    </form>
  </dialog>;
}
