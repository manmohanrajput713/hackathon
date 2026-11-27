'use client';
import { useRef, useState } from 'react';
import { Modal } from './components';

/** Parse CSV text (comma, semicolon or tab; quoted fields allowed) into rows of strings. */
export function parseCsv(text) {
  text = text.replace(/^﻿/, '');
  const first = text.split(/\r?\n/, 1)[0] || '';
  const delim = [',', ';', '\t'].sort((a, b) => first.split(b).length - first.split(a).length)[0];
  const rows = [];
  let row = [], cell = '', quoted = false;
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (quoted) {
      if (c === '"' && text[i + 1] === '"') { cell += '"'; i++; }
      else if (c === '"') quoted = false;
      else cell += c;
    } else if (c === '"') quoted = true;
    else if (c === delim) { row.push(cell.trim()); cell = ''; }
    else if (c === '\n' || c === '\r') {
      if (c === '\r' && text[i + 1] === '\n') i++;
      row.push(cell.trim()); cell = '';
      if (row.some(v => v !== '')) rows.push(row);
      row = [];
    } else cell += c;
  }
  row.push(cell.trim());
  if (row.some(v => v !== '')) rows.push(row);
  return rows;
}

export function toCsv(header, rows) {
  const esc = v => { const s = v == null ? '' : String(v); return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s; };
  return [...(header ? [header] : []), ...rows].map(r => r.map(esc).join(',')).join('\n') + '\n';
}

const norm = s => String(s).replace(/\(.*?\)|\[.*?\]/g, '').toLowerCase().replace(/[^a-z0-9]+/g, '_').replace(/^_|_$/g, '');

/**
 * Editable table in exactly the layout of a CSV input.
 * columns: [{key, label, aliases?}]; headerless: the CSV has no header row (labels are shown only to guide entry).
 * matrix: columns are wells and can be added or removed (onColumns gets the new labels).
 */
export function CsvTable({title, note, columns, initial = [], headerless, matrix, newColumnLabel, onSave, onClose, saveLabel = 'Save table', minRows = 10, fixedRows}) {
  const [cols, setCols] = useState(columns);
  const [rows, setRows] = useState(() => {
    const base = initial.map(r => cols.map((_, i) => r[i] == null ? '' : String(r[i])));
    while (base.length < minRows) base.push(cols.map(() => ''));
    return base;
  });
  const [page, setPage] = useState(0), [error, setError] = useState(''), [busy, setBusy] = useState(false), [info, setInfo] = useState('');
  const file = useRef(null);
  const PAGE = 100;
  const pages = Math.max(1, Math.ceil(rows.length / PAGE));
  const setCell = (r, c, v) => setRows(rs => rs.map((row, i) => i === r ? row.map((x, j) => j === c ? v : x) : row));
  async function importFile(e) {
    const f = e.target.files[0]; e.target.value = '';
    if (!f) return;
    const parsed = parseCsv(await f.text());
    if (!parsed.length) { setError('The file has no rows.'); return; }
    let body = parsed, mapped = cols.map((_, i) => i), nextCols = cols;
    if (!headerless) {
      const header = parsed[0].map(norm);
      mapped = cols.map(c => header.findIndex(h => [c.key, c.label, ...(c.aliases || [])].map(norm).includes(h)));
      const missing = cols.filter((c, i) => mapped[i] < 0 && c.required).map(c => c.label);
      if (missing.length) { setError(`Missing columns in ${f.name}: ${missing.join(', ')}.`); return; }
      body = parsed.slice(1);
    } else if (matrix && parsed[0].length !== cols.length) {
      nextCols = Array.from({length: parsed[0].length}, (_, i) => cols[i] || {key: `c${i}`, label: newColumnLabel(i)});
      mapped = nextCols.map((_, i) => i);
    } else if (fixedRows) {
      // Read-only columns (e.g. the dataset's time axis) stay; the file fills the editable columns in order.
      const editable = cols.map((c, i) => c.readOnly ? -1 : i).filter(i => i >= 0);
      if (parsed[0].length !== editable.length) { setError(`${f.name} has ${parsed[0].length} columns; expected ${editable.length}.`); return; }
      if (parsed.length !== rows.length) { setError(`${f.name} has ${parsed.length} rows; expected ${rows.length} (one per time step).`); return; }
      setRows(rows.map((r, k) => r.map((v, i) => { const e = editable.indexOf(i); return e < 0 ? v : (parsed[k][e] ?? ''); })));
      setError(''); setInfo(`Imported ${parsed.length} rows from ${f.name}. Check them, then save.`); return;
    } else if (parsed[0].length !== cols.length) {
      setError(`${f.name} has ${parsed[0].length} columns; this table has ${cols.length}.`); return;
    }
    setCols(nextCols);
    setRows(body.map(r => nextCols.map((_, i) => mapped[i] >= 0 ? (r[mapped[i]] ?? '') : '')));
    setPage(0); setError(''); setInfo(`Imported ${body.length} rows from ${f.name}. Check them, then save.`);
  }
  async function save() {
    setBusy(true); setError('');
    const filled = fixedRows ? rows : rows.filter(r => r.some(v => v !== ''));
    try { await onSave(filled, cols); } catch (err) { setError(err.message); } finally { setBusy(false); }
  }
  const view = rows.slice(page * PAGE, page * PAGE + PAGE);
  return <Modal title={title} onClose={onClose} wide>
    {note && <p className="muted">{note}</p>}
    <div className="toolbar">
      <button className="secondary small" onClick={() => file.current.click()}>Import CSV into table</button>
      <input ref={file} type="file" accept=".csv,.txt,.tsv" hidden onChange={importFile}/>
      {!fixedRows && <button className="secondary small" onClick={() => setRows(rs => [...rs, ...Array.from({length: 10}, () => cols.map(() => ''))])}>+ 10 rows</button>}
      {matrix && <button className="secondary small" onClick={() => { const i = cols.length; setCols(c => [...c, {key: `c${i}`, label: newColumnLabel(i)}]); setRows(rs => rs.map(r => [...r, ''])); }}>+ Column</button>}
      {matrix && cols.length > 1 && <button className="secondary small" onClick={() => { setCols(c => c.slice(0, -1)); setRows(rs => rs.map(r => r.slice(0, -1))); }}>− Last column</button>}
      <button className="link small dangerText" onClick={() => setRows(fixedRows ? rows.map(r => r.map((v, i) => cols[i].readOnly ? v : '')) : Array.from({length: minRows}, () => cols.map(() => '')))}>Clear</button>
      <span className="spacer"/><span className="muted small">{rows.filter(r => r.some(v => v !== '')).length} filled rows{headerless ? ' · saved without a header row' : ''}</span>
    </div>
    {info && <p className="importReport">{info}</p>}
    <div className="scroll gridScroll"><table className="grid"><thead><tr><th className="rowNo">#</th>{cols.map((c, i) => <th key={i}>{c.label}{headerless && <small>column {i + 1}</small>}</th>)}<th/></tr></thead>
      <tbody>{view.map((r, k) => { const ri = page * PAGE + k; return <tr key={ri}><td className="rowNo">{ri + 1}</td>
        {r.map((v, ci) => <td key={ci}>{cols[ci].readOnly ? <span className="ro">{v}</span> : cols[ci].options ? <select value={v} onChange={e => setCell(ri, ci, e.target.value)}><option value=""/>{cols[ci].options.map(o => <option key={o}>{o}</option>)}</select>
          : <input value={v} onChange={e => setCell(ri, ci, e.target.value)} aria-label={`${cols[ci].label} row ${ri + 1}`}/>}</td>)}
        <td>{!fixedRows && <button className="iconBtn small" aria-label={`Delete row ${ri + 1}`} onClick={() => setRows(rs => rs.filter((_, i) => i !== ri))}>×</button>}</td></tr>; })}</tbody></table></div>
    {pages > 1 && <div className="pager"><span>Rows {page * PAGE + 1}–{Math.min(rows.length, (page + 1) * PAGE)} of {rows.length}</span>
      <button className="secondary small" disabled={page === 0} onClick={() => setPage(page - 1)}>Previous</button>
      <button className="secondary small" disabled={page >= pages - 1} onClick={() => setPage(page + 1)}>Next</button></div>}
    {error && <p className="formError" role="alert">{error}</p>}
    <div className="formActions"><button className="secondary" onClick={onClose}>Cancel</button><button disabled={busy} onClick={save}>{busy ? 'Saving…' : saveLabel}</button></div>
  </Modal>;
}
