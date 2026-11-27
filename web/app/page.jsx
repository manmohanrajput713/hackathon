'use client';
import { useEffect, useState } from 'react';
import { api, fmt, when, typeTone, roleOf, RUN_ICON } from './lib';
import { Ctx, useApp, useCtx } from './ctx';
import { Modal, Confirm, KV, Status } from './components';
import { WellForm, ImportWizard } from './data';
import { DeepCRM } from './crm';
import { VFM, WellModels, Failure, Facility, Optimizer } from './modules';
import { Results } from './results';
import { FieldCanvas } from './canvas';

const DATASET_MODULE = {rates: ['crm', 'history'], esp_fast: ['vfm'], gl_fast: ['vfm'], esp_daily: ['failure', 'esp'], gl_pressure: ['failure', 'gas_lift'], gl_curve: ['models'], events: null};

export default function Console() {
  const app = useApp();
  return <Ctx.Provider value={app}><Shell/></Ctx.Provider>;
}

// Module navigation, grouped the way the work flows: reservoir → wells → field → records.
const NAV = [
  {id: null, label: 'Field map', group: 'map'},
  {id: 'crm', label: 'Deep-CRM', group: 'reservoir'},
  {id: 'vfm', label: 'Flow meter', group: 'wells'},
  {id: 'models', label: 'Well models', group: 'wells'},
  {id: 'failure', label: 'Failure', group: 'wells'},
  {id: 'facility', label: 'Facility', group: 'field'},
  {id: 'optimizer', label: 'Optimizer', group: 'field'},
  {id: 'results', label: 'Results', group: 'records'},
];

const ICONS = {
  menu: <path d="M4 7h16M4 12h16M4 17h16"/>,
  panel: <><rect x="3.5" y="4.5" width="17" height="15" rx="2"/><path d="M14.5 4.5v15"/></>,
  search: <><circle cx="11" cy="11" r="6"/><path d="m20 20-4.2-4.2"/></>,
  plus: <path d="M12 5v14M5 12h14"/>,
  upload: <path d="M12 15V4m0 0L7.5 8.5M12 4l4.5 4.5M5 15v3.5A1.5 1.5 0 0 0 6.5 20h11a1.5 1.5 0 0 0 1.5-1.5V15"/>,
  close: <path d="M6 6l12 12M18 6 6 18"/>,
  more: <><circle cx="6" cy="12" r="1.2"/><circle cx="12" cy="12" r="1.2"/><circle cx="18" cy="12" r="1.2"/></>,
};
function Icon({name, size = 18}) {
  return <svg className="icon" width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">{ICONS[name]}</svg>;
}

function useMedia(query) {
  const [match, setMatch] = useState(false);
  useEffect(() => {
    const mq = window.matchMedia(query);
    const update = () => setMatch(mq.matches);
    update();
    mq.addEventListener('change', update);
    return () => mq.removeEventListener('change', update);
  }, [query]);
  return match;
}

/** Cards turn a few degrees toward the pointer (CSS reads --rx/--ry). Returns the cleanup. Skipped for touch and reduced motion. */
function startPointerTilt() {
  if (!window.matchMedia('(hover: hover)').matches || window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;
  const SEL = '.card, .metrics > div, .demoCard';
  let cur = null;
  const clear = el => { el?.style.removeProperty('--rx'); el?.style.removeProperty('--ry'); };
  const move = e => {
    const el = e.target.closest?.(SEL) || null;
    if (cur !== el) { clear(cur); cur = el; }
    if (!el) return;
    const r = el.getBoundingClientRect();
    const k = Math.min(5, 1100 / Math.max(r.width, r.height));        // large cards tilt less
    el.style.setProperty('--rx', `${(-((e.clientY - r.top) / r.height - .5) * k).toFixed(2)}deg`);
    el.style.setProperty('--ry', `${(((e.clientX - r.left) / r.width - .5) * k).toFixed(2)}deg`);
  };
  const leave = () => { clear(cur); cur = null; };
  document.addEventListener('pointermove', move, {passive: true});
  document.documentElement.addEventListener('pointerleave', leave);
  return () => { document.removeEventListener('pointermove', move); document.documentElement.removeEventListener('pointerleave', leave); leave(); };
}

function Shell() {
  const {schema, fields, error, route, mode, toasts, select, set, go, runs} = useCtx();
  const narrow = useMedia('(max-width: 1024px)');          // sidebar becomes a drawer, navigation moves into it
  const [docked, setDocked] = useState(true);              // desktop: sidebar shown or collapsed
  const [drawer, setDrawer] = useState(false);             // tablet/phone: drawer open
  const [inspector, setInspector] = useState(true);
  useEffect(() => { if (window.matchMedia('(max-width: 1360px)').matches) setInspector(false); }, []);
  useEffect(() => startPointerTilt(), []);
  useEffect(() => { setDrawer(false); }, [route.module, route.well]);
  useEffect(() => { select(null); }, [route.module, route.field, select]);
  useEffect(() => {
    const key = e => { if (e.key === 'Escape') { setDrawer(false); if (window.matchMedia('(max-width: 1360px)').matches) setInspector(false); } };
    window.addEventListener('keydown', key);
    return () => window.removeEventListener('keydown', key);
  }, []);
  const ready = schema && fields;
  const module = route.module;
  const View = module && {crm: DeepCRM, vfm: VFM, models: WellModels, failure: Failure, facility: Facility, optimizer: Optimizer, results: Results}[module];
  const busy = runs.filter(r => r.status === 'queued' || r.status === 'running').length;
  const open = id => id ? go(id) : set({module: null, sub: null, run: null});
  const toggleMenu = () => narrow ? setDrawer(!drawer) : setDocked(!docked);
  const menuOpen = narrow ? drawer : docked;
  return <div className={['shell', mode === 'review' && 'review', docked ? 'side-docked' : 'side-hidden', drawer && 'drawer-open', inspector ? 'insp-open' : 'insp-closed'].filter(Boolean).join(' ')}>
    <TopBar menuOpen={menuOpen} onMenu={toggleMenu} inspector={inspector} onInspector={() => setInspector(!inspector)}/>
    <nav className="moduleNav" aria-label="Modules">
      {NAV.map((n, i) => <span key={n.label} className="navSlot">
        {i > 0 && NAV[i - 1].group !== n.group && <i className="navSep" aria-hidden="true"/>}
        <button className="navTab" aria-current={module === n.id ? 'page' : undefined} onClick={() => open(n.id)}>
          {n.label}{n.id === 'results' && busy > 0 && <span className="count warn" title="Runs in progress">{busy}</span>}</button></span>)}
    </nav>
    <div className="layout3">
      <aside className="tree" aria-label="Project">
        <div className="drawerOnly">
          <div className="treeSection"><div className="treeTitle">Navigate</div>
            {NAV.map(n => <button key={n.label} className={'treeItem' + (module === n.id ? ' on' : '')} onClick={() => open(n.id)}>
              <span className="ellipsis">{n.label}</span>{n.id === 'results' && busy > 0 && <span className="count warn">{busy}</span>}</button>)}</div>
          <div className="treeSection"><div className="treeTitle">Context</div><ContextControls stacked/><ModeSwitch/></div>
        </div>
        {ready && <ProjectTree/>}
      </aside>
      <main className="canvasMain">
        {error && <div role="alert" className="notice">{error}</div>}
        {!ready && !error && <p className="loading">Loading the field…</p>}
        {ready && <FieldCanvas/>}
        {ready && View && <div className="panelOverlay"><div className="panel" role="region" aria-label={NAV.find(n => n.id === module)?.label}>
          <div className="panelBody"><View/></div></div></div>}
      </main>
      <aside className="inspector" aria-label="Inspector">
        <div className="inspectorTop"><span>Inspector</span><button className="iconBtn small" aria-label="Close inspector" onClick={() => setInspector(false)}><Icon name="close" size={16}/></button></div>
        {ready && <Inspector/>}
      </aside>
    </div>
    <div className="scrim forDrawer" onClick={() => setDrawer(false)}/>
    <div className="scrim forInsp" onClick={() => setInspector(false)}/>
    <div className="toasts" aria-live="polite">{toasts.map(t => <div key={t.id} className={'toast ' + t.tone}>{t.text}</div>)}</div>
  </div>;
}

function TopBar({menuOpen, onMenu, inspector, onInspector}) {
  return <header className="topbar">
    <button className="topIcon" aria-label={menuOpen ? 'Hide menu' : 'Show menu'} aria-expanded={menuOpen} onClick={onMenu}><Icon name="menu" size={20}/></button>
    <a className="brand" href="#/"><span className="mark" aria-hidden="true">≋</span><span className="brandText"><b>Hackathon</b><span>Field console</span></span></a>
    <i className="topDivider" aria-hidden="true"/>
    <ContextControls/>
    <div className="topRight">
      <ModeSwitch dark/>
      <button className={'topIcon' + (inspector ? ' on' : '')} aria-label={inspector ? 'Hide inspector' : 'Show inspector'} aria-pressed={inspector} onClick={onInspector}><Icon name="panel" size={19}/></button>
    </div>
  </header>;
}

/** Field, scenario and active well. Compact label-over-value fields in the top bar; stacked in the drawer. */
function ContextControls({stacked}) {
  const {fields, field, set, scenarios, scenario, wells, schema, mode, route} = useCtx();
  const [dialog, setDialog] = useState(null);
  const producers = wells.filter(w => schema && roleOf(schema, w) === 'producer'), injectors = wells.filter(w => schema && roleOf(schema, w) === 'injector');
  return <div className={'context' + (stacked ? ' stacked' : '')}>
    <label className="ctxField"><span>Field</span><select value={field} onChange={e => set({field: e.target.value, well: null, scenario: null, run: null})} disabled={!fields?.length}>
      {!fields?.length && <option value="">No wells yet</option>}
      {(fields || []).map(f => <option key={f.name} value={f.name}>{f.name || '(no field)'} · {f.wells} wells</option>)}</select></label>
    <div className="ctxGroup">
      <label className="ctxField"><span>Scenario</span><select value={scenario?.id || ''} onChange={e => e.target.value === '__new' ? setDialog('new') : set({scenario: e.target.value, run: null})}>
        {scenarios.map(s => <option key={s.id} value={s.id}>{s.name}</option>)}{mode === 'edit' && <option value="__new">+ New scenario…</option>}</select></label>
      {mode === 'edit' && scenario && <button className="ctxMore" aria-label="Scenario options" title="Rename or delete scenario" onClick={() => setDialog('edit')}><Icon name="more" size={16}/></button>}
    </div>
    <label className="ctxField"><span>Active well</span><select value={wells.some(w => w.id === route.well) ? route.well : ''} onChange={e => set({well: e.target.value || null})}>
      <option value="">None</option>
      {producers.length > 0 && <optgroup label="Producers">{producers.map(w => <option key={w.id} value={w.id}>{w.name}</option>)}</optgroup>}
      {injectors.length > 0 && <optgroup label="Injectors">{injectors.map(w => <option key={w.id} value={w.id}>{w.name}</option>)}</optgroup>}</select></label>
    {dialog && <ScenarioDialog mode={dialog} onClose={() => setDialog(null)}/>}
  </div>;
}

function ModeSwitch({dark}) {
  const {mode, set} = useCtx();
  return <div className={'seg' + (dark ? ' dark' : '')} role="group" aria-label="Mode">
    {[['edit', 'Edit'], ['review', 'Review']].map(([v, label]) => <button key={v} aria-pressed={mode === v} title={v === 'review' ? 'Read-only: hides every control that changes data' : 'Edit data and run models'} onClick={() => set({mode: v})}>{label}</button>)}
  </div>;
}

function ScenarioDialog({mode, onClose}) {
  const {scenario, field, set, reloadScenarios, notify} = useCtx();
  const [name, setName] = useState(mode === 'new' ? `${scenario?.name || 'Base'} copy` : scenario.name), [busy, setBusy] = useState(false), [remove, setRemove] = useState(false);
  async function save(e) {
    e.preventDefault(); setBusy(true);
    try {
      const s = mode === 'new' ? await api('/scenarios', {method: 'POST', body: {name, field, copy_from: scenario?.id}})
        : await api(`/scenarios/${scenario.id}`, {method: 'PUT', body: {name, facility: scenario.facility, policy: scenario.policy, notes: scenario.notes}});
      await reloadScenarios(); set({scenario: s.id}); notify(mode === 'new' ? `Created ${s.name}` : 'Scenario renamed'); onClose();
    } catch (err) { notify(err.message, 'error'); setBusy(false); }
  }
  return <Modal title={mode === 'new' ? 'New scenario' : `Scenario: ${scenario.name}`} onClose={onClose}>
    <form className="form" onSubmit={save}>
      <label>Name<input required maxLength={60} value={name} onChange={e => setName(e.target.value)} autoFocus/></label>
      {mode === 'new' && <small>Copies the facility limits and failure policy of “{scenario?.name}”. Change them on the Facility tab.</small>}
      <div className="formActions">{mode === 'edit' && <button type="button" className="secondary dangerText" onClick={() => setRemove(true)}>Delete scenario</button>}<span className="spacer"/>
        <button type="button" className="secondary" onClick={onClose}>Cancel</button><button disabled={busy}>{mode === 'new' ? 'Create' : 'Save'}</button></div>
    </form>
    {remove && <Confirm title={`Delete ${scenario.name}?`} message="Runs saved under this scenario are kept." onClose={() => setRemove(false)}
      onConfirm={async () => { try { await api(`/scenarios/${scenario.id}`, {method: 'DELETE'}); await reloadScenarios(); set({scenario: null}); onClose(); } catch (e) { notify(e.message, 'error'); } }}/>}
  </Modal>;
}


const TYPE_SHORT = {esp: 'ESP', gas_lift: 'Gas lift', producer: '', injector: ''};

function ProjectTree() {
  const {schema, wells, set, route, runs, go, field, reloadWells, loadFields, notify} = useCtx();
  const [query, setQuery] = useState(''), [form, setForm] = useState(false), [importing, setImporting] = useState(false), [demo, setDemo] = useState(null);
  const shown = wells.filter(w => !query || `${w.name} ${w.code}`.toLowerCase().includes(query.toLowerCase()));
  const groups = [['Producers', shown.filter(w => roleOf(schema, w) === 'producer')], ['Injectors', shown.filter(w => roleOf(schema, w) === 'injector')]];
  const active = wells.find(w => w.id === route.well);
  const refresh = () => { reloadWells(); loadFields(); };
  const alert = w => w.results.esp_risk?.status === 'alert' || w.results.valve_risk?.status === 'alert';
  return <div className="treeInner">
    <div className="searchBox"><Icon name="search" size={16}/><input type="search" placeholder="Search wells" value={query} onChange={e => setQuery(e.target.value)} aria-label="Search wells"/></div>
    <div className="treeActions editOnly">
      <button className="small" onClick={() => setForm(true)}><Icon name="plus" size={15}/>Add well</button>
      <button className="small secondary" onClick={() => setImporting(true)}><Icon name="upload" size={15}/>Import CSV</button></div>

    <div className="treeSection">
      <div className="treeTitle">Wells<span>{field || 'No field'}</span></div>
      {groups.map(([label, list]) => <div key={label} className="treeGroup">
        <div className="treeSub">{label}<span className="count">{list.length}</span></div>
        {!list.length && <p className="treeEmpty">{query ? 'No match' : `No ${label.toLowerCase()} yet`}</p>}
        {list.map(w => <button key={w.id} className={'treeItem' + (route.well === w.id ? ' on' : '')} onClick={() => set({well: w.id})} title={`${w.name} · ${schema.wellTypes[w.type].label}`}>
          <i className={'dot ' + typeTone[w.type]}/><span className="ellipsis">{w.name}</span>
          {alert(w) && <span className="alertDot" title="Risk alert"/>}
          {w.status !== 'active' ? <small>{w.status.replace('_', ' ')}</small> : TYPE_SHORT[w.type] ? <small>{TYPE_SHORT[w.type]}</small> : w.code !== w.name && <small>{w.code}</small>}</button>)}
      </div>)}
    </div>

    {active && <div className="treeSection">
      <div className="treeTitle">Datasets<span>{active.name}</span></div>
      {schema.wellTypes[active.type].datasets.length === 0 && <p className="treeEmpty">A flowing producer has no lift datasets. Right-click it on the map to add ESP or gas lift.</p>}
      {schema.wellTypes[active.type].datasets.map(d => { const c = active.counts[d]; const target = DATASET_MODULE[d] || (active.type === 'esp' ? ['failure', 'esp'] : ['failure', 'gas_lift']);
        return <button key={d} className="treeItem" onClick={() => go(target[0], target[1] || null, {well: active.id})}>
          <span className="ellipsis">{schema.datasets[d].label}</span><small className={c?.rows ? '' : 'faint'}>{c?.rows ? `${fmt(c.rows, 0)} rows` : 'empty'}</small></button>; })}
    </div>}

    <div className="treeSection">
      <div className="treeTitle">Saved runs{runs.length > 0 && <span>{runs.length}</span>}</div>
      {!runs.length && <p className="treeEmpty">Runs you start appear here with their inputs pinned.</p>}
      {runs.slice(0, 12).map(r => <button key={r.id} className={'treeItem run ' + r.status + (route.run === r.id && route.module === 'results' ? ' on' : '')} onClick={() => go('results', null, {run: r.id})} title={`${r.name} · ${r.status}`}>
        <i className={'runDot ' + r.status}/><span className="ellipsis">{r.name}</span>{r.stale?.length > 0 && <span className="alertDot warn" title="Inputs changed: needs review"/>}</button>)}
      {runs.length > 12 && <button className="link small treeMore" onClick={() => go('results')}>All {runs.length} runs →</button>}
    </div>

    <div className="demoCard editOnly">
      {field === 'Demo field'
        ? <><b>Demo field</b><p>Five wells built from the project's test datasets.</p>
          <div className="demoActions"><button className="link small" onClick={() => setDemo('create')}>Reset demo</button><button className="link small dangerText" onClick={() => setDemo('remove')}>Remove demo</button></div></>
        : <><b>Try the demo</b><p>Two injectors, an ESP and two gas-lift producers in a separate field.</p><button className="link small" onClick={() => setDemo('create')}>Load demo field →</button></>}
    </div>
    {form && <WellForm onClose={() => setForm(false)} onSaved={w => { setForm(false); refresh(); set({well: w.id, field: w.field}); }}/>}
    {importing && <ImportWizard kind="wells" onClose={() => setImporting(false)} onDone={refresh}/>}
    {demo === 'create' && <DemoDialog onClose={() => setDemo(null)}/>}
    {demo === 'remove' && <Confirm title="Remove the demo field?" message="Deletes the five demo wells, their data, scenarios and runs. Your other fields are not touched." action="Remove demo"
      onClose={() => setDemo(null)} onConfirm={async () => { try { await api('/demo', {method: 'DELETE'}); notify('Demo removed'); const f = await loadFields(); set({field: f[0]?.name ?? '', well: null, scenario: null}); } catch (e) { notify(e.message, 'error'); } }}/>}
  </div>;
}

function DemoDialog({onClose}) {
  const {loadFields, set, notify, fields} = useCtx();
  const [busy, setBusy] = useState(false);
  const exists = (fields || []).some(f => f.name === 'Demo field');
  return <Modal title={exists ? 'Reset the five-well demo' : 'Load the five-well demo'} onClose={onClose}>
    {exists && <div className="warning">The demo field already exists. Loading it again <b>resets</b> it: its wells, data, scenarios and runs are deleted and built fresh.
      Your other fields are not touched; if one of your wells already uses a demo name (such as I1), the demo well is named by its demo ID instead (DEMO-I1).</div>}
    <p>Creates a separate <b>Demo field</b> with two injectors, one ESP producer and two gas-lift producers, built from the project's own test datasets (pywaterflood, failure_1, failure_2, pinn and gas_pinc).
      Producer rates are generated from the real injection history with known connectivity, so you can check that a history match recovers it. It never mixes with your fields and can be removed in one click.</p>
    <p className="muted">Suggested path: Deep-CRM → History Match → Connectivity → Flow meter (GL1) → Well models → new scenario with less lift gas → Optimizer → Results compare.</p>
    <div className="formActions"><button className="secondary" onClick={onClose}>Cancel</button>
      <button disabled={busy} onClick={async () => { setBusy(true); try { await api('/demo', {method: 'POST'}); await loadFields(); set({field: 'Demo field', well: null, scenario: null, module: 'crm', sub: 'wells'}); notify(exists ? 'Demo field reset' : 'Demo field loaded'); onClose(); } catch (e) { notify(e.message, 'error'); setBusy(false); } }}>{busy ? 'Building… about 30 s' : exists ? 'Reset and load demo' : 'Load demo'}</button></div>
  </Modal>;
}


function Inspector() {
  const {well, schema, selection, select} = useCtx();
  const [quality, setQuality] = useState(null);
  const versionKey = well ? Object.values(well.counts).map(c => c.version).join(',') : '';
  useEffect(() => {
    if (!well) { setQuality(null); return; }
    let live = true;
    api(`/wells/${well.id}/quality`).then(q => live && setQuality(q)).catch(() => live && setQuality(null));
    return () => { live = false; };
  }, [well?.id, versionKey]); // eslint-disable-line react-hooks/exhaustive-deps
  const s = well?.settings || {};
  const warnings = well ? Object.values(well.results).flatMap(r => (r.warnings || []).map(w => `${r.label}: ${w}`)) : [];
  const datasets = well ? schema.wellTypes[well.type].datasets : [];
  return <div className="inspectorInner">
    {selection && <div className="card selection">
      <div className="cardHead"><b>{selection.title}</b><button className="iconBtn small" aria-label="Close" onClick={() => select(null)}><Icon name="close" size={15}/></button></div>
      {selection.items && <KV items={selection.items}/>}
      {selection.text && <p>{selection.text}</p>}
      {selection.links && <div className="links">{selection.links.map(([label, fn]) => <button key={label} className="link small" onClick={fn}>{label} →</button>)}</div>}
    </div>}
    {!well ? <div className="card emptyCard"><b>No well selected</b><p>Pick a well in the sidebar, on the map or in the Active well menu. Its inputs, data quality and warnings appear here.</p></div> : <>
      <div className="card">
        <div className="wellHead"><div><h3>{well.name}</h3><small>{well.code !== well.name ? `${well.code} · ` : ''}{well.field || 'No field'}</small></div>
          <span className={'tag ' + typeTone[well.type]}>{schema.wellTypes[well.type].label}</span></div>
        <KV items={[['Status', <span key="s" className={'statusText ' + well.status}>{well.status.replace('_', ' ')}</span>],
          well.type === 'esp' && ['Current frequency', s.current_hz != null ? fmt(s.current_hz) + ' Hz' : 'not set'],
          well.type === 'esp' && ['Envelope', s.env_min_hz != null || s.env_max_hz != null ? `${fmt(s.env_min_hz)}–${fmt(s.env_max_hz)} Hz` : 'not set'],
          well.type === 'gas_lift' && ['Current lift gas', s.current_gas_sm3d != null ? fmt(s.current_gas_sm3d, 0) + ' sm³/d' : 'not set'],
          well.type === 'gas_lift' && ['Envelope', s.env_min_gas != null || s.env_max_gas != null ? `${fmt(s.env_min_gas, 0)}–${fmt(s.env_max_gas, 0)} sm³/d` : 'not set']]}/>
        {Object.values(well.results).length > 0 && <div className="results">{Object.entries(well.results).map(([k, r]) => <div key={k} className="resultLine"><span>{r.label}</span>
          {r.status ? <Status value={r.status}/> : <b>{fmt(r.value)} {r.unit || ''}</b>}</div>)}</div>}
      </div>
      <div className="card"><div className="cardTitle">Input versions</div>
        {!datasets.length ? <p>No lift datasets for a flowing producer.</p>
          : <ul className="versionList">{datasets.map(d => { const c = well.counts[d]; return <li key={d}><span>{schema.datasets[d].label}</span>
            <small className={c?.rows ? '' : 'faint'}>{c?.rows ? `v${c.version} · ${fmt(c.rows, 0)} rows` : 'empty'}</small></li>; })}</ul>}</div>
      <div className="card"><div className="cardTitle">Data quality</div>
        {!quality ? <p>Checking…</p> : !quality.length ? <p>Nothing to check yet.</p> : <ul className="quality">{quality.map(q => <li key={q.dataset} className={q.issues.length ? 'issue' : q.rows ? 'ok' : 'none'}>
          <span>{q.label}</span>{q.issues.length ? q.issues.map(i => <small key={i}>{i}</small>) : <small>{q.rows ? `${q.first ?? ''}${q.last ? ' → ' + q.last : ''}` : 'No data'}</small>}
          {q.optional_coverage && <small>{Object.entries(q.optional_coverage).map(([k, v]) => `${k} ${Math.round(v * 100)}%`).join(', ')}</small>}</li>)}</ul>}</div>
      <div className="card"><div className="cardTitle">Warnings{warnings.length > 0 && <span className="count warn">{warnings.length}</span>}</div>
        {warnings.length ? <ul className="warnList">{warnings.map(w => <li key={w}>{w}</li>)}</ul> : <p>No model warnings for this well.</p>}</div>
    </>}
  </div>;
}
