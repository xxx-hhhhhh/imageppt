import { useEffect, useRef, useState } from 'react';
import type { CSSProperties, MouseEvent as ReactMouseEvent, PointerEvent as ReactPointerEvent } from 'react';
import { Canvas, Ellipse, FabricImage, Line, Rect, Textbox } from 'fabric';
import type { LayoutElement, LayoutJSON } from '../types/layout';
import { assetUrl } from '../services/api';

interface Props {
  page: LayoutJSON | undefined;
  zoomFactor?: number;
  selectedIds: string[];
  onSelection: (ids: string[]) => void;
  onChange: (layout: LayoutJSON) => void;
}

const objectId = (object: any): string | undefined => object?.elementId;
const isVisibleElement = (element: LayoutElement): boolean =>
  element.type !== 'group' && !element.metadata?.suppressed && !element.metadata?.suppressRender && !element.metadata?.ownedBy;

function movableModuleIds(page: LayoutJSON, elementId: string): string[] {
  const plate = page.elements.find((item) => item.metadata?.layerRole === 'container' &&
    (item.id === elementId || (Array.isArray(item.metadata.moduleMemberIds) && item.metadata.moduleMemberIds.includes(elementId))));
  if (!plate) return [elementId];
  return [plate.id, ...(Array.isArray(plate.metadata?.moduleMemberIds) ? plate.metadata.moduleMemberIds.filter((id): id is string => typeof id === 'string') : [])]
    .filter((id) => page.elements.some((item) => item.id === id && isVisibleElement(item)));
}

function boundedModuleDelta(page: LayoutJSON, ids: string[], dx: number, dy: number): [number, number] {
  const members = page.elements.filter((item) => ids.includes(item.id));
  const left = Math.min(...members.map((item) => item.x));
  const top = Math.min(...members.map((item) => item.y));
  const right = Math.max(...members.map((item) => item.x + item.width));
  const bottom = Math.max(...members.map((item) => item.y + item.height));
  return [Math.max(-left, Math.min(page.slide.width - right, dx)), Math.max(-top, Math.min(page.slide.height - bottom, dy))];
}

function colorWithOpacity(color: string | undefined, opacity = 1): string {
  if (!color || opacity >= 1) return color || '#17365D';
  const raw = color.replace('#', '');
  if (raw.length !== 6) return color;
  const alpha = Math.round(opacity * 255).toString(16).padStart(2, '0');
  return `#${raw}${alpha}`;
}

export function EditorCanvas({ page, zoomFactor = 1, selectedIds, onSelection, onChange }: Props) {
  const canvasElement = useRef<HTMLCanvasElement | null>(null);
  const wrapper = useRef<HTMLDivElement | null>(null);
  const stageRef = useRef<HTMLDivElement | null>(null);
  const fabricRef = useRef<Canvas | null>(null);
  const [zoom, setZoom] = useState(1);

  useEffect(() => {
    if (!canvasElement.current || !wrapper.current || !page) return;
    const fabricCanvas = new Canvas(canvasElement.current, { preserveObjectStacking: true, selection: true });
    fabricRef.current = fabricCanvas;
    const availableWidth = Math.max(1, wrapper.current.clientWidth - 24);
    const availableHeight = Math.max(1, wrapper.current.clientHeight - 24);
    const zoom = Math.min(availableWidth / page.slide.width, availableHeight / page.slide.height, 1) * zoomFactor;
    setZoom(zoom);
    fabricCanvas.setDimensions({ width: page.slide.width, height: page.slide.height });
    fabricCanvas.setZoom(1);
    if (stageRef.current) {
      stageRef.current.style.transform = `scale(${zoom})`;
      stageRef.current.style.transformOrigin = 'top left';
    }

    const emitSelection = (event: any) => onSelection((event.selected || []).map(objectId).filter(Boolean) as string[]);
    fabricCanvas.on('selection:created', emitSelection);
    fabricCanvas.on('selection:updated', emitSelection);
    fabricCanvas.on('selection:cleared', () => onSelection([]));
    fabricCanvas.on('object:modified', (event: any) => {
      const next = readLayout(fabricCanvas, page);
      const id = objectId(event.target);
      const before = page.elements.find((item) => item.id === id);
      const after = next.elements.find((item) => item.id === id);
      if (id && before && after && event.transform?.action === 'drag') {
        const ids = movableModuleIds(page, id);
        const [dx, dy] = boundedModuleDelta(page, ids, after.x - before.x, after.y - before.y);
        onChange({ ...next, elements: next.elements.map((item) => ids.includes(item.id) && item.id !== id ? { ...item, x: item.x + dx, y: item.y + dy } : item) });
      } else onChange(next);
    });
    fabricCanvas.on('text:changed', () => onChange(readLayout(fabricCanvas, page)));

    let disposed = false;
    (async () => {
      for (const element of page.elements.filter(isVisibleElement).sort((a, b) => a.zIndex - b.zIndex)) {
        if (disposed) return;
        let object: any = null;
        try {
          object = await createFabricObject(element);
        } catch {
          // A missing image asset must not prevent editable text and shapes from rendering.
          object = null;
        }
        if (!object) continue;
        (object as any).elementId = element.id;
        (object as any).elementType = element.type;
        fabricCanvas.add(object);
      }
      fabricCanvas.renderAll();
    })();

    const resize = () => {
      if (!wrapper.current || !fabricRef.current) return;
      const width = Math.max(1, wrapper.current.clientWidth - 24);
      const height = Math.max(1, wrapper.current.clientHeight - 24);
      const nextZoom = Math.min(width / page.slide.width, height / page.slide.height, 1) * zoomFactor;
      setZoom(nextZoom);
      if (stageRef.current) {
        stageRef.current.style.transform = `scale(${nextZoom})`;
        stageRef.current.style.transformOrigin = 'top left';
      }
      fabricCanvas.renderAll();
    };
    const observer = new ResizeObserver(resize);
    observer.observe(wrapper.current);
    return () => {
      disposed = true;
      observer.disconnect();
      fabricCanvas.dispose();
      fabricRef.current = null;
    };
  }, [page, onChange, onSelection, zoomFactor]);

  return <div className="canvas-wrapper" ref={wrapper}><div className="canvas-viewport" style={{ width: (page?.slide.width || 1) * zoom, height: (page?.slide.height || 1) * zoom }}><div className="canvas-stage" ref={stageRef} style={{ width: page?.slide.width || 1, height: page?.slide.height || 1 }}><canvas ref={canvasElement} /><div className="canvas-overlay">{page?.elements.filter(isVisibleElement).sort((a, b) => a.zIndex - b.zIndex).map((element) => <VisualElement key={element.id} element={element} selected={selectedIds.includes(element.id)} onSelect={onSelection} onChange={onChange} page={page} />)}</div></div></div></div>;
}

function VisualElement({ element, selected, onSelect, onChange, page }: { element: LayoutElement; selected: boolean; onSelect: (ids: string[]) => void; onChange: (layout: LayoutJSON) => void; page: LayoutJSON }) {
  const style = element.style || {};
  const lowConfidence = element.confidence != null && element.confidence < 0.65;
  const base: CSSProperties = { position: 'absolute', left: element.x, top: element.y, width: Math.max(1, element.width), height: Math.max(1, element.height), transform: `rotate(${element.rotation}deg)`, opacity: style.opacity ?? 1, pointerEvents: element.type === 'background' ? 'none' : 'auto', outline: selected ? '2px solid #2563a6' : undefined, outlineOffset: 2, boxShadow: lowConfidence ? '0 0 0 2px #f59e0b66' : undefined };
  const beginDrag = (event: ReactPointerEvent<HTMLElement>) => {
    if (element.type === 'background') return;
    if ((event.target as HTMLElement).closest('[contenteditable="true"]')) return;
    event.preventDefault();
    event.stopPropagation();
    const movingIds = movableModuleIds(page, element.id);
    onSelect([element.id]);
    const target = event.currentTarget;
    const startX = event.clientX;
    const startY = event.clientY;
    const stageBounds = (event.currentTarget.closest('.canvas-stage') as HTMLElement | null)?.getBoundingClientRect();
    const scaleX = stageBounds && page.slide.width ? stageBounds.width / page.slide.width : 1;
    const scaleY = stageBounds && page.slide.height ? stageBounds.height / page.slide.height : 1;
    const move = (moveEvent: PointerEvent) => {
      const [dx, dy] = boundedModuleDelta(page, movingIds, (moveEvent.clientX - startX) / scaleX, (moveEvent.clientY - startY) / scaleY);
      target.closest('.canvas-overlay')?.querySelectorAll<HTMLElement>('[data-element-id]').forEach((node) => {
        if (movingIds.includes(node.dataset.elementId || '')) {
          const member = page.elements.find((item) => item.id === node.dataset.elementId);
          node.style.transform = `translate(${dx}px, ${dy}px) rotate(${member?.rotation || 0}deg)`;
        }
      });
    };
    const up = (upEvent: PointerEvent) => {
      window.removeEventListener('pointermove', move); window.removeEventListener('pointerup', up);
      const [dx, dy] = boundedModuleDelta(page, movingIds, (upEvent.clientX - startX) / scaleX, (upEvent.clientY - startY) / scaleY);
      target.closest('.canvas-overlay')?.querySelectorAll<HTMLElement>('[data-element-id]').forEach((node) => {
        if (movingIds.includes(node.dataset.elementId || '')) {
          const member = page.elements.find((item) => item.id === node.dataset.elementId);
          node.style.transform = `rotate(${member?.rotation || 0}deg)`;
        }
      });
      onChange({ ...page, elements: page.elements.map((item) => movingIds.includes(item.id) ? { ...item, x: item.x + dx, y: item.y + dy } : item) });
    };
    window.addEventListener('pointermove', move);
    window.addEventListener('pointerup', up, { once: true });
  };
  const beginResize = (event: React.PointerEvent<HTMLSpanElement>) => {
    event.preventDefault(); event.stopPropagation(); onSelect([element.id]);
    const startX = event.clientX; const startY = event.clientY; const startW = element.width; const startH = element.height; const target = event.currentTarget.parentElement;
    const stageBounds = (event.currentTarget.closest('.canvas-stage') as HTMLElement | null)?.getBoundingClientRect();
    const scaleX = stageBounds && page.slide.width ? stageBounds.width / page.slide.width : 1;
    const scaleY = stageBounds && page.slide.height ? stageBounds.height / page.slide.height : 1;
    const move = (moveEvent: PointerEvent) => { if (target) { target.style.width = `${Math.max(12, startW + (moveEvent.clientX - startX) / scaleX)}px`; target.style.height = `${Math.max(12, startH + (moveEvent.clientY - startY) / scaleY)}px`; } };
    const up = (upEvent: PointerEvent) => { window.removeEventListener('pointermove', move); window.removeEventListener('pointerup', up); onChange({ ...page, elements: page.elements.map((item) => item.id === element.id ? { ...item, width: Math.max(12, Math.min(page.slide.width - item.x, startW + (upEvent.clientX - startX) / scaleX)), height: Math.max(12, Math.min(page.slide.height - item.y, startH + (upEvent.clientY - startY) / scaleY)) } : item) }); };
    window.addEventListener('pointermove', move); window.addEventListener('pointerup', up, { once: true });
  };
  const beginTextEdit = (event: ReactMouseEvent<HTMLDivElement>) => {
    if (element.type !== 'text') return;
    const target = event.currentTarget.querySelector<HTMLElement>('[data-text-content]');
    if (!target || target.isContentEditable) return;
    // Editing must never remove React-owned sibling controls (Ctrl+A/Delete).
    target.contentEditable = 'true';
    target.focus();
    const finish = () => {
      target.contentEditable = 'false';
      onChange({ ...page, elements: page.elements.map((item) => item.id === element.id ? { ...item, text: target.innerText } : item) });
      target.removeEventListener('blur', finish);
    };
    target.addEventListener('blur', finish);
  };
  if (element.type === 'background' && element.src) return <img data-element-id={element.id} className="visual-image" src={assetUrl(element.src)} alt="" style={{ ...base, objectFit: 'cover' }} />;
  if (element.type === 'image' && element.src) {
    const crop = element.crop || {};
    const left = Math.min(.95, Math.max(0, crop.left || 0));
    const right = Math.min(.95 - left, Math.max(0, crop.right || 0));
    const top = Math.min(.95, Math.max(0, crop.top || 0));
    const bottom = Math.min(.95 - top, Math.max(0, crop.bottom || 0));
    return <div data-element-id={element.id} className="visual-image" style={base} onPointerDown={beginDrag}><div style={{ position: 'absolute', inset: 0, overflow: 'hidden', pointerEvents: 'none' }}><img src={assetUrl(element.src)} alt="" draggable={false} style={{ position: 'absolute', width: `${100 / (1 - left - right)}%`, height: `${100 / (1 - top - bottom)}%`, left: `${-left * 100 / (1 - left - right)}%`, top: `${-top * 100 / (1 - top - bottom)}%`, objectFit: 'fill', pointerEvents: 'none' }} /></div>{selected && <span className="resize-handle" onPointerDown={beginResize} />}</div>;
  }
  if (element.type === 'text') return <div data-element-id={element.id} className="visual-text" onPointerDown={beginDrag} onDoubleClick={beginTextEdit} style={{ ...base, color: style.color || '#111827', fontFamily: style.fontFamily || 'Microsoft YaHei', fontSize: style.fontSize || 24, fontWeight: style.fontWeight || 400, fontStyle: style.fontStyle || 'normal', textAlign: style.align || 'left', whiteSpace: 'pre-wrap', overflow: 'hidden' }}><span data-text-content style={{ display: 'block', width: '100%', height: '100%' }}>{element.text}</span>{selected && <span className="resize-handle" onPointerDown={beginResize} />}</div>;
  if (element.type === 'line' || element.type === 'arrow') return <div data-element-id={element.id} className={`visual-line ${element.type}`} onPointerDown={beginDrag} style={{ ...base, width: element.width, height: 0, top: element.y + element.height / 2, borderTop: `${style.strokeWidth || 1}px solid ${style.stroke || '#17365D'}` }}>{selected && <span className="resize-handle" onPointerDown={beginResize} />}</div>;
  if (['rectangle', 'roundedRectangle', 'ellipse'].includes(element.type)) return <div data-element-id={element.id} className={`visual-shape ${element.type}`} onPointerDown={beginDrag} style={{ ...base, background: style.fill || '#DCE6F1', border: `${style.strokeWidth || 1}px solid ${style.stroke || '#17365D'}`, borderRadius: element.type === 'ellipse' ? '50%' : element.type === 'roundedRectangle' ? 18 : 0 }}>{selected && <span className="resize-handle" onPointerDown={beginResize} />}</div>;
  return null;
}

async function createFabricObject(element: LayoutElement): Promise<any> {
  const style = element.style || {};
  const common = { left: element.x, top: element.y, angle: element.rotation, opacity: style.opacity ?? 1, originX: 'left' as const, originY: 'top' as const };
  // Images use the interactive DOM layer so their crop preview has one owner.
  if (element.type === 'image') return null;
  if (element.type === 'background' && element.src) {
    const image = await FabricImage.fromURL(assetUrl(element.src));
    image.set({ ...common, width: element.width, height: element.height, scaleX: 1, scaleY: 1, selectable: false, evented: false });
    return image;
  }
  if (element.type === 'text') {
    return new Textbox(element.text || '', { ...common, width: element.width, height: element.height, fontFamily: style.fontFamily || 'Microsoft YaHei', fontSize: style.fontSize || 24, fontWeight: style.fontWeight || 400, fontStyle: style.fontStyle || 'normal', fill: style.color || '#111827', textAlign: style.align || 'left', editable: true, splitByGrapheme: false });
  }
  const fill = colorWithOpacity(style.fill, style.opacity ?? 1);
  const stroke = style.stroke || '#17365D';
  if (element.type === 'ellipse') return new Ellipse({ ...common, rx: element.width / 2, ry: element.height / 2, fill, stroke, strokeWidth: style.strokeWidth || 1 });
  if (element.type === 'line' || element.type === 'arrow') {
    const line = new Line([0, 0, Math.max(1, element.width), Math.max(1, element.height)], { ...common, stroke, strokeWidth: style.strokeWidth || 1 });
    if (element.type === 'arrow') (line as any).set({ strokeUniform: true });
    return line;
  }
  if (element.type === 'roundedRectangle') return new Rect({ ...common, width: element.width, height: element.height, rx: Math.min(18, element.height / 4), ry: Math.min(18, element.height / 4), fill, stroke, strokeWidth: style.strokeWidth || 1 });
  if (element.type === 'rectangle') return new Rect({ ...common, width: element.width, height: element.height, fill, stroke, strokeWidth: style.strokeWidth || 1 });
  return null;
}

function readLayout(canvas: Canvas, original: LayoutJSON): LayoutJSON {
  const byId = new Map<string, LayoutElement>(original.elements.map((element) => [element.id, structuredClone(element)]));
  canvas.getObjects().forEach((object: any) => {
    const id = objectId(object);
    if (!id) return;
    const element = byId.get(id);
    if (!element || element.type === 'background') return;
    element.x = object.left || 0;
    element.y = object.top || 0;
    element.width = Math.max(1, object.getScaledWidth ? object.getScaledWidth() : object.width || element.width);
    element.height = Math.max(1, object.getScaledHeight ? object.getScaledHeight() : object.height || element.height);
    element.rotation = object.angle || 0;
    if (element.type === 'text') {
      element.text = object.text || '';
      element.style = { ...element.style, fontSize: object.fontSize, fontFamily: object.fontFamily, fontWeight: object.fontWeight, fontStyle: object.fontStyle, color: object.fill, align: object.textAlign };
    }
  });
  return { ...original, elements: [...byId.values()] };
}
