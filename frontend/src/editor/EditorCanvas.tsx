import { useEffect, useRef } from 'react';
import type { CSSProperties, MouseEvent as ReactMouseEvent, PointerEvent as ReactPointerEvent } from 'react';
import { Canvas, Ellipse, FabricImage, Line, Rect, Textbox } from 'fabric';
import type { LayoutElement, LayoutJSON } from '../types/layout';
import { assetUrl } from '../services/api';

interface Props {
  page: LayoutJSON | undefined;
  selectedIds: string[];
  onSelection: (ids: string[]) => void;
  onChange: (layout: LayoutJSON) => void;
}

const objectId = (object: any): string | undefined => object?.elementId;
const isVisibleElement = (element: LayoutElement): boolean =>
  element.type !== 'group' && !element.metadata?.suppressed && !element.metadata?.suppressRender && !element.metadata?.ownedBy;

function colorWithOpacity(color: string | undefined, opacity = 1): string {
  if (!color || opacity >= 1) return color || '#17365D';
  const raw = color.replace('#', '');
  if (raw.length !== 6) return color;
  const alpha = Math.round(opacity * 255).toString(16).padStart(2, '0');
  return `#${raw}${alpha}`;
}

export function EditorCanvas({ page, selectedIds, onSelection, onChange }: Props) {
  const canvasElement = useRef<HTMLCanvasElement | null>(null);
  const wrapper = useRef<HTMLDivElement | null>(null);
  const stageRef = useRef<HTMLDivElement | null>(null);
  const fabricRef = useRef<Canvas | null>(null);

  useEffect(() => {
    if (!canvasElement.current || !wrapper.current || !page) return;
    const fabricCanvas = new Canvas(canvasElement.current, { preserveObjectStacking: true, selection: true });
    fabricRef.current = fabricCanvas;
    const availableWidth = Math.max(480, wrapper.current.clientWidth - 24);
    const availableHeight = Math.max(340, wrapper.current.clientHeight - 24);
    const zoom = Math.min(availableWidth / page.slide.width, availableHeight / page.slide.height, 1);
    fabricCanvas.setDimensions({ width: page.slide.width, height: page.slide.height });
    fabricCanvas.setZoom(1);
    if (stageRef.current) {
      stageRef.current.style.transform = `scale(${zoom})`;
      stageRef.current.style.transformOrigin = 'center center';
    }

    const emitSelection = (event: any) => onSelection((event.selected || []).map(objectId).filter(Boolean) as string[]);
    fabricCanvas.on('selection:created', emitSelection);
    fabricCanvas.on('selection:updated', emitSelection);
    fabricCanvas.on('selection:cleared', () => onSelection([]));
    fabricCanvas.on('object:modified', () => onChange(readLayout(fabricCanvas, page)));
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
      const width = Math.max(480, wrapper.current.clientWidth - 24);
      const height = Math.max(340, wrapper.current.clientHeight - 24);
      const nextZoom = Math.min(width / page.slide.width, height / page.slide.height, 1);
      if (stageRef.current) {
        stageRef.current.style.transform = `scale(${nextZoom})`;
        stageRef.current.style.transformOrigin = 'center center';
      }
      fabricCanvas.renderAll();
    };
    window.addEventListener('resize', resize);
    return () => {
      disposed = true;
      window.removeEventListener('resize', resize);
      fabricCanvas.dispose();
      fabricRef.current = null;
    };
  }, [page, onChange, onSelection]);

  return <div className="canvas-wrapper" ref={wrapper}><div className="canvas-stage" ref={stageRef} style={{ width: page?.slide.width || 1, height: page?.slide.height || 1 }}><canvas ref={canvasElement} /><div className="canvas-overlay">{page?.elements.filter(isVisibleElement).sort((a, b) => a.zIndex - b.zIndex).map((element) => <VisualElement key={element.id} element={element} selected={selectedIds.includes(element.id)} onSelect={onSelection} onChange={onChange} page={page} />)}</div></div></div>;
}

function VisualElement({ element, selected, onSelect, onChange, page }: { element: LayoutElement; selected: boolean; onSelect: (ids: string[]) => void; onChange: (layout: LayoutJSON) => void; page: LayoutJSON }) {
  const style = element.style || {};
  const lowConfidence = element.confidence != null && element.confidence < 0.65;
  const base: CSSProperties = { position: 'absolute', left: element.x, top: element.y, width: Math.max(1, element.width), height: Math.max(1, element.height), transform: `rotate(${element.rotation}deg)`, opacity: style.opacity ?? 1, pointerEvents: element.type === 'background' ? 'none' : 'auto', outline: selected ? '2px solid #2563a6' : undefined, outlineOffset: 2, boxShadow: lowConfidence ? '0 0 0 2px #f59e0b66' : undefined };
  const beginDrag = (event: ReactPointerEvent<HTMLElement>) => {
    if (element.type === 'background') return;
    event.preventDefault();
    event.stopPropagation();
    onSelect([element.id]);
    const target = event.currentTarget;
    const startX = event.clientX;
    const startY = event.clientY;
    const startLeft = element.x;
    const startTop = element.y;
    const stageBounds = (event.currentTarget.closest('.canvas-stage') as HTMLElement | null)?.getBoundingClientRect();
    const scaleX = stageBounds && page.slide.width ? stageBounds.width / page.slide.width : 1;
    const scaleY = stageBounds && page.slide.height ? stageBounds.height / page.slide.height : 1;
    const move = (moveEvent: PointerEvent) => { target.style.transform = `translate(${(moveEvent.clientX - startX) / scaleX}px, ${(moveEvent.clientY - startY) / scaleY}px) rotate(${element.rotation}deg)`; };
    const up = (upEvent: PointerEvent) => { window.removeEventListener('pointermove', move); window.removeEventListener('pointerup', up); onChange({ ...page, elements: page.elements.map((item) => item.id === element.id ? { ...item, x: Math.max(0, Math.min(page.slide.width - item.width, startLeft + (upEvent.clientX - startX) / scaleX)), y: Math.max(0, Math.min(page.slide.height - item.height, startTop + (upEvent.clientY - startY) / scaleY)) } : item) }); };
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
  const beginTextEdit = (event: ReactMouseEvent<HTMLDivElement>) => { if (element.type !== 'text') return; const target = event.currentTarget; target.contentEditable = 'true'; target.focus(); const finish = () => { target.contentEditable = 'false'; onChange({ ...page, elements: page.elements.map((item) => item.id === element.id ? { ...item, text: target.innerText } : item) }); target.removeEventListener('blur', finish); }; target.addEventListener('blur', finish); };
  if ((element.type === 'background' || element.type === 'image') && element.src) return <img className="visual-image" src={assetUrl(element.src)} alt="" style={{ ...base, objectFit: 'cover' }} onPointerDown={beginDrag} />;
  if (element.type === 'text') return <div className="visual-text" onPointerDown={beginDrag} onDoubleClick={beginTextEdit} style={{ ...base, color: style.color || '#111827', fontFamily: style.fontFamily || 'Microsoft YaHei', fontSize: style.fontSize || 24, fontWeight: style.fontWeight || 400, fontStyle: style.fontStyle || 'normal', textAlign: style.align || 'left', whiteSpace: 'pre-wrap', overflow: 'hidden' }}>{element.text}{selected && <span className="resize-handle" onPointerDown={beginResize} />}</div>;
  if (element.type === 'line' || element.type === 'arrow') return <div className={`visual-line ${element.type}`} onPointerDown={beginDrag} style={{ ...base, width: element.width, height: 0, top: element.y + element.height / 2, borderTop: `${style.strokeWidth || 1}px solid ${style.stroke || '#17365D'}` }}>{selected && <span className="resize-handle" onPointerDown={beginResize} />}</div>;
  if (['rectangle', 'roundedRectangle', 'ellipse'].includes(element.type)) return <div className={`visual-shape ${element.type}`} onPointerDown={beginDrag} style={{ ...base, background: style.fill || '#DCE6F1', border: `${style.strokeWidth || 1}px solid ${style.stroke || '#17365D'}`, borderRadius: element.type === 'ellipse' ? '50%' : element.type === 'roundedRectangle' ? 18 : 0 }}>{selected && <span className="resize-handle" onPointerDown={beginResize} />}</div>;
  return null;
}

async function createFabricObject(element: LayoutElement): Promise<any> {
  const style = element.style || {};
  const common = { left: element.x, top: element.y, angle: element.rotation, opacity: style.opacity ?? 1, originX: 'left' as const, originY: 'top' as const };
  if ((element.type === 'background' || element.type === 'image') && element.src) {
    const image = await FabricImage.fromURL(assetUrl(element.src));
    image.set({ ...common, width: element.width, height: element.height, scaleX: 1, scaleY: 1, selectable: element.type !== 'background', evented: element.type !== 'background' });
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
