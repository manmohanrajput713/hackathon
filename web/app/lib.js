export async function api(path, {method = 'GET', body, raw, headers} = {}) {
  const init = {method, headers: {...headers}, cache: 'no-store'};
  if (raw) init.body = raw;
  else if (body !== undefined) { init.body = JSON.stringify(body); init.headers['Content-Type'] = 'application/json'; }
  let response;
  try { response = await fetch('/api' + path, init); }
  catch { throw new Error('Cannot reach the field API. Check that the console is running.'); }
  const text = await response.text();
  let data = null;
  try { data = text ? JSON.parse(text) : null; } catch { /* non-JSON error page */ }
  if (!response.ok) throw new Error(data?.error || `Request failed (${response.status})`);
  return data;
}

export const qs = params => {
  const q = Object.entries(params).filter(([, v]) => v !== undefined && v !== null && v !== '').map(([k, v]) => `${encodeURIComponent(k)}=${encodeURIComponent(v)}`).join('&');
  return q ? '?' + q : '';
};

function save(blob, name) {
  const url = URL.createObjectURL(blob);
  const link = Object.assign(document.createElement('a'), {href: url, download: name});
  document.body.appendChild(link); link.click(); link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export async function download(path) {
  const response = await fetch('/api' + path, {cache: 'no-store'});
  if (!response.ok) {
    let message = `Download failed (${response.status})`;
    try { message = (await response.json()).error || message; } catch {}
    throw new Error(message);
  }
  const match = /filename="([^"]+)"/.exec(response.headers.get('content-disposition') || '');
  save(await response.blob(), match ? match[1] : 'export');
}

export function saveSvg(svg, name, png) {
  const clone = svg.cloneNode(true);
  clone.setAttribute('xmlns', 'http://www.w3.org/2000/svg');
  const style = document.createElement('style');
  style.textContent = 'text{font:12px "Segoe UI",sans-serif;fill:#5a6d66} line{stroke:#e1e9e4}';
  clone.insertBefore(style, clone.firstChild);
  const box = svg.viewBox.baseVal;
  const bg = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
  Object.entries({x: 0, y: 0, width: box.width, height: box.height, fill: '#fff'}).forEach(([k, v]) => bg.setAttribute(k, v));
  clone.insertBefore(bg, style.nextSibling);
  const text = new XMLSerializer().serializeToString(clone);
  if (!png) return save(new Blob([text], {type: 'image/svg+xml'}), name + '.svg');
  const img = new Image();
  img.onload = () => {
    const canvas = Object.assign(document.createElement('canvas'), {width: box.width * 2, height: box.height * 2});
    const ctx = canvas.getContext('2d'); ctx.scale(2, 2); ctx.drawImage(img, 0, 0);
    canvas.toBlob(b => save(b, name + '.png'));
  };
  img.src = 'data:image/svg+xml;charset=utf-8,' + encodeURIComponent(text);
}

export function upload(path, file) {
  return api(path, {method: 'POST', raw: file, headers: {'x-filename': file.name, 'Content-Type': 'application/octet-stream'}});
}

export const fmt = (value, digits = 2) =>
  value == null || value === '' || Number.isNaN(Number(value)) ? '—' : Number(value).toLocaleString('en-US', {maximumFractionDigits: digits});
export const pct = (value, digits = 1) => value == null ? '—' : fmt(value * 100, digits) + '%';
export const when = iso => iso ? new Date(iso).toLocaleString(undefined, {dateStyle: 'medium', timeStyle: 'short'}) : '—';
export const header = col => col.unit ? `${col.label} (${col.unit})` : col.label;
export const typeTone = {esp: 'esp', gas_lift: 'gas', producer: 'crm', injector: 'inj'};
export const roleOf = (schema, w) => schema.wellTypes[w.type].role;
export const RUN_ICON = {queued: '◷', running: '◔', done: '●', failed: '✕'};
