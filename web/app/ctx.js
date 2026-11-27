'use client';
import { createContext, useCallback, useContext, useEffect, useRef, useState } from 'react';
import { api, qs } from './lib';

export const Ctx = createContext(null);
export const useCtx = () => useContext(Ctx);

const MODULES = ['crm', 'vfm', 'models', 'failure', 'facility', 'optimizer', 'results'];

function readHash() {
  if (typeof window === 'undefined') return {};
  const [path, query = ''] = window.location.hash.replace(/^#\/?/, '').split('?');
  const [module, sub] = path.split('/');
  const p = Object.fromEntries(new URLSearchParams(query));
  return {module: MODULES.includes(module) ? module : null, sub: sub || null, field: p.field, scenario: p.scenario, well: p.well, mode: p.mode, run: p.run};
}

/** Shared application state: selections persist in the URL so switching tabs keeps field, scenario and well. */
export function useApp() {
  const [schema, setSchema] = useState(null);
  const [fields, setFields] = useState(null);
  const [wells, setWells] = useState([]);
  const [scenarios, setScenarios] = useState([]);
  const [runs, setRuns] = useState([]);
  const [route, setRoute] = useState({module: null, sub: null, field: null, scenario: null, well: null, mode: 'edit', run: null});
  const [selection, setSelection] = useState(null);
  const [toasts, setToasts] = useState([]);
  const [error, setError] = useState('');
  const full = useRef(new Map());
  const [, bump] = useState(0);

  const notify = useCallback((text, tone = 'ok') => {
    const id = Math.random();
    setToasts(t => [...t, {id, text, tone}]);
    setTimeout(() => setToasts(t => t.filter(x => x.id !== id)), tone === 'error' ? 9000 : 5000);
  }, []);

  // URL <-> state
  useEffect(() => {
    const sync = () => { const h = readHash(); setRoute(r => ({...r, ...Object.fromEntries(Object.entries(h).filter(([, v]) => v != null)), module: h.module, sub: h.sub, run: h.run || null})); };
    let saved = {};
    try { saved = JSON.parse(localStorage.getItem('fc-route') || '{}'); } catch {}
    const h = readHash();
    setRoute(r => ({...r, ...saved, ...Object.fromEntries(Object.entries(h).filter(([, v]) => v != null))}));
    window.addEventListener('hashchange', sync);
    return () => window.removeEventListener('hashchange', sync);
  }, []);
  // The URL and saved selection are written outside the state updater: updating the router while React renders is not allowed.
  const routeRef = useRef(route);
  useEffect(() => { routeRef.current = route; }, [route]);
  const set = useCallback(patch => {
    const next = {...routeRef.current, ...patch};
    routeRef.current = next;
    setRoute(next);
    const hash = `#/${next.module || ''}${next.module && next.sub ? '/' + next.sub : ''}` + qs({field: next.field, scenario: next.scenario, well: next.well, mode: next.mode !== 'edit' ? next.mode : null, run: next.run});
    if (window.location.hash !== hash) window.history.replaceState(null, '', hash);
    try { localStorage.setItem('fc-route', JSON.stringify({field: next.field, scenario: next.scenario, well: next.well, mode: next.mode})); } catch {}
  }, []);
  const go = useCallback((module, sub = null, extra = {}) => { set({module, sub, run: null, ...extra}); window.scrollTo?.(0, 0); }, [set]);

  const loadFields = useCallback(() => api('/fields').then(f => { setFields(f); setError(''); return f; }).catch(e => { setError(e.message); return []; }), []);
  useEffect(() => { api('/schema').then(setSchema).catch(e => setError(e.message)); loadFields(); }, [loadFields]);

  // Choose a valid field once fields are known.
  useEffect(() => {
    if (!fields) return;
    if (route.field == null || (fields.length && !fields.some(f => f.name === route.field))) set({field: fields[0]?.name ?? ''});
  }, [fields, route.field, set]);

  const field = route.field ?? '';
  const reloadWells = useCallback(() => api('/wells?field=' + encodeURIComponent(field)).then(w => { setWells(w); return w; }).catch(e => notify(e.message, 'error')), [field, notify]);
  const reloadScenarios = useCallback(() => api('/scenarios' + qs({field})).then(s => { setScenarios(s); return s; }).catch(e => notify(e.message, 'error')), [field, notify]);
  const reloadRuns = useCallback(() => api('/runs' + qs({field, limit: 150})).then(r => { setRuns(r); return r; }).catch(() => {}), [field]);
  useEffect(() => { if (fields) { reloadWells(); reloadScenarios(); reloadRuns(); } }, [fields, field, reloadWells, reloadScenarios, reloadRuns]);

  useEffect(() => { if (scenarios.length && !scenarios.some(s => s.id === route.scenario)) set({scenario: scenarios[0].id}); }, [scenarios, route.scenario, set]);
  useEffect(() => { if (wells.length && route.well && !wells.some(w => w.id === route.well)) set({well: null}); }, [wells, route.well, set]);

  // Poll while runs are queued or running; refresh wells when one finishes (well cards cache latest results).
  const busy = runs.some(r => r.status === 'queued' || r.status === 'running');
  const prev = useRef(new Map());
  useEffect(() => {
    const timer = setInterval(reloadRuns, busy ? 2500 : 30000);
    return () => clearInterval(timer);
  }, [busy, reloadRuns]);
  useEffect(() => {
    let finished = false;
    for (const r of runs) {
      const before = prev.current.get(r.id);
      if (before && before !== r.status && (r.status === 'done' || r.status === 'failed')) {
        finished = true;
        full.current.delete(r.id);
        notify(r.status === 'done' ? `${r.name} finished` : `${r.name} failed: ${r.error}`, r.status === 'done' ? 'ok' : 'error');
      }
      prev.current.set(r.id, r.status);
    }
    if (finished) { reloadWells(); bump(n => n + 1); }
  }, [runs, notify, reloadWells]);

  const startRun = useCallback(async (kind, params, opts = {}) => {
    try {
      const run = await api('/runs', {method: 'POST', body: {kind, params, field, scenario_id: opts.scenario_id ?? null, name: opts.name || undefined}});
      setRuns(r => [run, ...r]);
      prev.current.set(run.id, run.status);
      notify(`Queued: ${run.name}`);
      return run;
    } catch (e) { notify(e.message, 'error'); return null; }
  }, [field, notify]);

  const getRun = useCallback(async id => {
    if (full.current.has(id)) return full.current.get(id);
    const run = await api(`/runs/${id}`);
    if (run.status === 'done' || run.status === 'failed') full.current.set(id, run);
    return run;
  }, []);

  const well = wells.find(w => w.id === route.well) || null;
  const scenario = scenarios.find(s => s.id === route.scenario) || null;
  return {schema, fields, loadFields, wells, reloadWells, scenarios, reloadScenarios, scenario, runs, reloadRuns, startRun, getRun,
          field, well, route, set, go, mode: route.mode || 'edit', selection, select: setSelection, toasts, notify, error};
}

/** Load a run's full outputs, re-fetching when it finishes. */
export function useRun(id) {
  const {getRun, runs} = useCtx();
  const [run, setRun] = useState(null);
  const status = runs.find(r => r.id === id)?.status;
  useEffect(() => {
    let live = true;
    if (!id) { setRun(null); return; }
    getRun(id).then(r => live && setRun(r)).catch(() => live && setRun(null));
    return () => { live = false; };
  }, [id, status, getRun]);
  return run;
}

/** Latest run of a kind matching a filter, from the recent-runs list. */
export function latest(runs, kind, filter = {}, statuses = ['done']) {
  return runs.find(r => r.kind === kind && statuses.includes(r.status) && Object.entries(filter).every(([k, v]) => r[k] === v || r.params?.[k] === v)) || null;
}
