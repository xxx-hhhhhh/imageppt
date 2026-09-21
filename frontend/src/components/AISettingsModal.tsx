import { useEffect, useState } from 'react';
import { getVisionSettings, saveVisionSettings, testVisionSettings, type VisionProviderSettings, type VisionSettings } from '../services/api';

interface Props { open: boolean; onClose: () => void; onSaved?: (settings: VisionSettings) => void; }

const emptyQwen: VisionProviderSettings = {
  enabled: true,
  configured: false,
  apiKeyMasked: '',
  baseUrl: 'https://dashscope.aliyuncs.com/compatible-mode/v1',
  model: 'qwen3-vl-flash',
};

const empty: VisionSettings = {
  enabled: true,
  selectedProvider: 'qwen',
  configured: false,
  mode: 'standard',
  timeout: 120,
  proxyMode: 'system',
  manualProxy: '',
  network: { systemProxy: '读取中…', status: '● 可用' },
  providers: { qwen: emptyQwen },
};

type Diagnostics = {
  proxyTcp?: { status?: string; message?: string };
  qwenBaseUrl?: { status?: string; message?: string };
  qwenApi?: { status?: string; message?: string };
  networkPath?: string;
};

export function AISettingsModal({ open, onClose, onSaved }: Props) {
  const [settings, setSettings] = useState<VisionSettings>(empty);
  const [draftKey, setDraftKey] = useState('');
  const [clearKey, setClearKey] = useState(false);
  const [visibleKey, setVisibleKey] = useState(false);
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState('');
  const [error, setError] = useState('');
  const [diagnostics, setDiagnostics] = useState<Diagnostics>({});

  useEffect(() => {
    if (!open) return;
    setError(''); setStatus(''); setDiagnostics({}); setDraftKey(''); setClearKey(false);
    void getVisionSettings().then((value) => {
      setSettings({ ...empty, ...value, selectedProvider: value.selectedProvider === 'local' ? 'local' : 'qwen', providers: { qwen: { ...emptyQwen, ...(value.providers?.qwen || {}) } } });
    }).catch(() => setError('无法读取本机 Qwen 设置'));
  }, [open]);

  const qwen = settings.providers.qwen || emptyQwen;
  const update = (patch: Partial<VisionSettings>) => setSettings((current) => ({ ...current, ...patch }));
  const updateQwen = (patch: Partial<VisionProviderSettings>) => setSettings((current) => ({ ...current, providers: { qwen: { ...current.providers.qwen, ...patch } } }));

  const payload = () => ({
    enabled: settings.enabled,
    selectedProvider: settings.selectedProvider,
    mode: settings.mode,
    timeout: settings.timeout,
    proxyMode: settings.proxyMode,
    manualProxy: settings.manualProxy,
    providers: {
      qwen: {
        enabled: qwen.enabled,
        apiKey: draftKey,
        baseUrl: qwen.baseUrl || 'https://dashscope.aliyuncs.com/compatible-mode/v1',
        model: qwen.model || 'qwen3-vl-flash',
        clearApiKey: clearKey,
      },
    },
  });

  const save = async () => {
    setBusy(true); setError(''); setStatus('');
    try { const saved = await saveVisionSettings(payload()); setSettings(saved); setDraftKey(''); setClearKey(false); onSaved?.(saved); setStatus('已保存到本机设置'); }
    catch { setError('保存失败，请检查设置内容'); }
    finally { setBusy(false); }
  };

  const testCurrent = async () => {
    setBusy(true); setError(''); setStatus('正在测试 Qwen3-VL-Flash…');
    try {
      await saveVisionSettings(payload());
      const result = await testVisionSettings();
      setDiagnostics(result.diagnostics || {});
      setStatus(result.success ? 'Qwen3-VL-Flash ● 连接成功' : result.message);
    } catch { setError('测试失败，请检查 Qwen API Key、代理和网络设置'); }
    finally { setBusy(false); }
  };

  if (!open) return null;
  return <div className="modal-backdrop" role="presentation"><section className="ai-modal ai-settings-modal-large" role="dialog" aria-modal="true" aria-labelledby="ai-settings-title">
    <div className="ai-modal-header"><h2 id="ai-settings-title">AI视觉模型设置</h2><button className="modal-close" onClick={onClose} aria-label="关闭">×</button></div>
    <div className="ai-modal-body">
      <label className="toggle-row"><span>启用 AI视觉理解</span><input type="checkbox" checked={settings.enabled} onChange={(event) => update({ enabled: event.target.checked, selectedProvider: event.target.checked ? 'qwen' : 'local' })} /></label>
      <label className="field"><span>AI Provider</span><select value={settings.selectedProvider} onChange={(event) => update({ selectedProvider: event.target.value as 'qwen' | 'local', enabled: event.target.value === 'qwen' })}><option value="local">关闭AI，仅本地解析</option><option value="qwen">Qwen3-VL-Flash</option></select></label>
      <div className="provider-section-title">Qwen3-VL-Flash</div>
      <div className="provider-card">
        <div className="provider-key-label"><span>Qwen API Key</span><span className={qwen.configured ? 'provider-ok' : 'provider-muted'}>{qwen.configured ? '● 已配置' : '○ 未配置'}</span></div>
        <div className="key-input"><input type={visibleKey ? 'text' : 'password'} placeholder={qwen.apiKeyMasked || '输入 Qwen API Key'} value={draftKey} onChange={(event) => { setDraftKey(event.target.value); setClearKey(false); }} /><button type="button" onClick={() => setVisibleKey((value) => !value)}>{visibleKey ? '隐藏' : '显示'}</button><button type="button" onClick={() => { setDraftKey(''); setClearKey(true); }} disabled={!qwen.configured && !draftKey}>清除 Key</button></div>
        <label className="field"><span>Qwen Base URL</span><input value={qwen.baseUrl} onChange={(event) => updateQwen({ baseUrl: event.target.value })} /></label>
        <label className="field"><span>Model</span><input value={qwen.model} onChange={(event) => updateQwen({ model: event.target.value })} /></label>
      </div>
      <div className="provider-section-title">网络代理</div>
      <div className="network-card">
        <div className="network-row"><span>系统代理：{settings.network?.systemProxy || '直连'}</span><strong className={settings.network?.status?.includes('可用') ? 'provider-ok' : 'provider-muted'}>{settings.network?.status || '○ 未检测'}</strong></div>
        <div className="network-mode"><label><input type="radio" checked={settings.proxyMode === 'system'} onChange={() => update({ proxyMode: 'system' })} /> 自动系统代理</label><label><input type="radio" checked={settings.proxyMode === 'manual'} onChange={() => update({ proxyMode: 'manual' })} /> 手动代理</label><label><input type="radio" checked={settings.proxyMode === 'none'} onChange={() => update({ proxyMode: 'none' })} /> 不使用代理</label></div>
        {settings.proxyMode === 'manual' && <label className="field"><span>代理地址</span><input value={settings.manualProxy} placeholder="http://127.0.0.1:7897" onChange={(event) => update({ manualProxy: event.target.value })} /></label>}
        <div className="network-hint">优先级：手动代理 → Windows 系统代理 → HTTPS_PROXY → HTTP_PROXY → 直连</div>
      </div>
      <div className="field"><span>模式</span><div className="radio-row"><label><input type="radio" checked={settings.mode === 'fast'} onChange={() => update({ mode: 'fast' })} /> 快速</label><label><input type="radio" checked={settings.mode === 'standard'} onChange={() => update({ mode: 'standard' })} /> 标准</label><label><input type="radio" checked={settings.mode === 'high'} onChange={() => update({ mode: 'high' })} /> 高精度</label></div></div>
      <button className="test-connection" disabled={busy} onClick={() => void testCurrent()}>测试当前模型</button>
      {(diagnostics.proxyTcp || diagnostics.qwenBaseUrl || diagnostics.qwenApi) && <div className="network-test-results"><div>代理 TCP：{diagnostics.proxyTcp?.status || '—'}</div><div>Qwen Base URL：{diagnostics.qwenBaseUrl?.status || '—'}</div><div>Qwen API：{diagnostics.qwenApi?.status || '—'}</div><div>实际网络路径：{diagnostics.networkPath || '—'}</div></div>}
      {status && <div className="ai-status">状态：{status}</div>}{error && <div className="ai-error">{error}</div>}
      <div className="local-secret-note">API Key 仅保存在本机设置目录，不写入 Google Drive 项目目录。</div>
    </div>
    <div className="ai-modal-footer"><button className="toolbar-button" onClick={onClose}>取消</button><button className="toolbar-button primary" disabled={busy} onClick={() => void save()}>保存</button></div>
  </section></div>;
}
