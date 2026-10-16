// Alpha is read from the actual rendered asset, including crop and rotation.
// An unreadable/not-yet-loaded asset stays selectable (never silently disappears).
const pixels = new WeakMap<HTMLImageElement, { src: string; data: ImageData }>();
export function imageHit(node: HTMLElement, clientX: number, clientY: number): boolean {
  if (!node.classList.contains('visual-image')) return true;
  const image = node.querySelector('img');
  const stage = node.closest<HTMLElement>('.canvas-stage');
  if (!image || !stage || !image.complete || !image.naturalWidth) return true;
  let cached = pixels.get(image);
  if (!cached || cached.src !== image.currentSrc) {
    try {
      const canvas = document.createElement('canvas');
      canvas.width = image.naturalWidth; canvas.height = image.naturalHeight;
      const ctx = canvas.getContext('2d', { willReadFrequently: true });
      if (!ctx) return true;
      ctx.drawImage(image, 0, 0);
      cached = { src: image.currentSrc, data: ctx.getImageData(0, 0, canvas.width, canvas.height) };
      pixels.set(image, cached);
    } catch { return true; }
  }
  const bounds = stage.getBoundingClientRect();
  const sx = bounds.width / stage.offsetWidth, sy = bounds.height / stage.offsetHeight;
  const style = getComputedStyle(node), origin = style.transformOrigin.split(' ').map(parseFloat);
  const local = new DOMPoint((clientX - bounds.left) / sx - parseFloat(style.left) - origin[0],
    (clientY - bounds.top) / sy - parseFloat(style.top) - origin[1]);
  const point = new DOMMatrix(style.transform === 'none' ? undefined : style.transform).inverse().transformPoint(local);
  const x = point.x + origin[0], y = point.y + origin[1];
  const imageStyle = getComputedStyle(image);
  const px = Math.floor((x - parseFloat(imageStyle.left)) / parseFloat(imageStyle.width) * cached.data.width);
  const py = Math.floor((y - parseFloat(imageStyle.top)) / parseFloat(imageStyle.height) * cached.data.height);
  return px >= 0 && py >= 0 && px < cached.data.width && py < cached.data.height &&
    cached.data.data[(py * cached.data.width + px) * 4 + 3] > 16;
}

export function hitElement(overlay: HTMLElement, x: number, y: number): HTMLElement | undefined {
  const seen = new Set<HTMLElement>();
  for (const hit of document.elementsFromPoint(x, y)) {
    const node = hit.closest<HTMLElement>('[data-element-id]');
    if (!node || !overlay.contains(node) || seen.has(node)) continue;
    seen.add(node);
    if (getComputedStyle(node).pointerEvents !== 'none' && imageHit(node, x, y)) return node;
  }
}
