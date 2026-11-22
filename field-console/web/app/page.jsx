'use client';
import { useEffect, useState } from 'react';
import { Field } from './field';

const number = (value, digits=2) => Number(value).toLocaleString('en-US',{maximumFractionDigits:digits});
const tabs = ['Well performance','Equipment health','Lift allocation'];

function Trend({well}) {
  const [cursor,setCursor]=useState(null);
  const rows=well.points, last=rows.length-1;
  const selected=rows[Math.min(cursor??last,last)];
  const values=rows.flatMap(r=>r.slice(1));
  const min=Math.min(...values), max=Math.max(...values);
  const x=t=>65+((t-rows[0][0])/(rows[last][0]-rows[0][0]||1))*890;
  const y=v=>174-((v-min)/(max-min||1))*146;
  const label=well.predLabel||'Model estimate', reference=well.refLabel||'Dataset reference';
  return <><div className="metrics">{[[label,selected[1]],[reference,selected[2]],['Difference',selected[1]-selected[2]]].map(([title,v])=><div key={title}><small>{title}</small><strong>{number(v)} <em>{well.unit}</em></strong></div>)}</div>
    <div className="chartHead"><h2>{well.quantity} history</h2><div className="legend"><i/>{label}<i className="ref"/>{reference}</div></div>
    <svg viewBox="0 0 985 204" className="chart" role="img" aria-label={well.quantity+' versus '+well.xLabel}>
    {[0,1,2,3,4].map(i=>{const v=min+(max-min)*i/4;return <g key={i}><line x1="65" x2="955" y1={y(v)} y2={y(v)} stroke="#d9e3df"/><text x="53" y={y(v)+4} textAnchor="end">{number(v,1)}</text></g>})}
    {[2,1].map(c=><polyline key={c} fill="none" stroke={c===1?'#007953':'#aa7551'} strokeWidth={c===1?2.8:2} strokeDasharray={c===2?'6 4':undefined} points={rows.map(r=>x(r[0])+','+y(r[c])).join(' ')}/>)}
    <line x1={x(selected[0])} x2={x(selected[0])} y1="20" y2="178" stroke="#849890" strokeDasharray="3 4"/><circle cx={x(selected[0])} cy={y(selected[1])} r="4" fill="#007953"/>
    <text x="65" y="198">{number(rows[0][0])}</text><text x="955" y="198" textAnchor="end">{number(rows[last][0])}</text></svg>
    <label className="slider">{well.xLabel} <b>{number(selected[0])}</b><input aria-label="Replay sample" type="range" min="0" max={last} value={cursor??last} onChange={e=>setCursor(Number(e.target.value))}/></label></>;
}
function Wells({data}) {
  const [index,setIndex]=useState(0);
  const w=data.wells[index];
  return <section><div className="sectionHead"><div><small>Well replay</small><h2>{w.name}</h2><p>{w.source}</p></div><label>Selected well<select value={index} onChange={e=>setIndex(Number(e.target.value))}>{data.wells.map((well,i)=><option value={i} key={well.name}>{well.name}</option>)}</select></label></div>
    <Trend well={w} key={w.name}/></section>;
}
function Health({rows}) {
  const [filter,setFilter]=useState('All equipment'),[status,setStatus]=useState('All statuses');
  const filtered=rows.filter(r=>(filter==='All equipment'||r.kind===filter)&&(status==='All statuses'||r.status===status));
  return <section><div className="sectionHead"><div><small>Maintenance replay</small><h2>Equipment health</h2><p>Latest available score per installation or valve.</p></div><div className="controls"><select aria-label="Equipment type" value={filter} onChange={e=>setFilter(e.target.value)}>{['All equipment','Pump','Valve'].map(x=><option key={x}>{x}</option>)}</select><select aria-label="Alert filter" value={status} onChange={e=>setStatus(e.target.value)}>{['All statuses','alert','no_alert','unavailable'].map(x=><option key={x} value={x}>{x.replace('_',' ')}</option>)}</select></div></div>
    <div className="scroll"><table><thead><tr>{['Equipment','Source','Observation','Risk score','Status'].map(k=><th key={k}>{k}</th>)}</tr></thead><tbody>{filtered.map(r=><tr key={r.kind+r.well}><td>{r.well}</td><td>{r.source}</td><td>{r.time}</td><td>{r.score==null?'—':number(r.score,3)}</td><td><span className={'status '+r.status}>{r.status.replace('_',' ')}</span></td></tr>)}</tbody></table>{!filtered.length&&<p>No matching scores.</p>}</div></section>;
}
function Allocation({budget}) {
  const [gas,setGas]=useState(((budget.min+budget.max)/2).toFixed(3)),[energy,setEnergy]=useState('0'),[carbon,setCarbon]=useState('0'),[result,setResult]=useState(null),[busy,setBusy]=useState(false);
  async function run(e){e.preventDefault();setBusy(true);setResult(null);
    try {const r=await fetch('/api/allocate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({gas:+gas,energy_weight:+energy,emissions_weight:+carbon})});if(!r.ok)throw new Error('Allocation service unavailable. Try again.');setResult(await r.json());}catch(err){setResult({message:err.message});}finally{setBusy(false);}}
  return <section><div className="sectionHead"><div><small>Simulation scenario</small><h2>Lift-gas allocation</h2><p>Compare feasible valve settings on two frozen, ten-minute well-response curves.</p></div></div>
  <form onSubmit={run} onChange={()=>setResult(null)}><fieldset disabled={busy}><label>Available gas (kg/s)<input required type="number" min="0" step="any" value={gas} onChange={e=>setGas(e.target.value)}/></label><label>Energy penalty (m³/kWh)<input required type="number" min="0" step="any" value={energy} onChange={e=>setEnergy(e.target.value)}/></label><label>Emissions penalty (m³/kg CO₂e)<input required type="number" min="0" step="any" value={carbon} onChange={e=>setCarbon(e.target.value)}/></label><button>{busy?'Solving…':'Run allocation'}</button></fieldset></form>
  <p>Model gas range {number(budget.min,3)}–{number(budget.max,3)} kg/s. Scenario assumptions are 25% water cut, 60 kW per kg/s of gas, and 0.4 kg CO₂e/kWh.</p>
  <div aria-live="polite">{result?.feasible?<><div className="metrics">{[['Estimated oil',result.totals.oil,'m³/day'],['Allocated gas',result.totals.gas,'kg/s'],['Energy',result.totals.energy,'kWh/day']].map(([s,n,u])=><div key={s}><small>{s}</small><strong>{number(n)} <em>{u}</em></strong></div>)}</div><p>Estimated emissions {number(result.totals.emissions)} kg CO₂e/day.</p><table><thead><tr><th>Well</th><th>Lift valve opening</th><th>Gas (kg/s)</th><th>Oil (m³/day)</th></tr></thead><tbody>{result.wells.map(w=><tr key={w.well}><td>{w.well}</td><td>{number(w.setting*100)}%</td><td>{number(w.gas,3)}</td><td>{number(w.oil)}</td></tr>)}</tbody></table></>:result&&<p role="status">{result.message}</p>}</div></section>;
}
export default function Dashboard(){
  const [data,setData]=useState(null),[error,setError]=useState(''),[tab,setTab]=useState(tabs[0]);
  async function refresh(){try{const r=await fetch('/api/overview',{cache:'no-store'});if(!r.ok)throw Error();setData(await r.json());setError('');}catch{setError('Replay data is unavailable. Check the model worker and refresh.');}}
  useEffect(()=>{refresh();const timer=setInterval(refresh,60000);return()=>clearInterval(timer);},[]);
  return <><header><div className="brand"><span className="mark" aria-hidden="true">≋</span> Strata</div><span className="headDetail">Field operations</span></header><main><div className="intro"><div><small>Production engineering</small><h1>Field overview</h1></div><span className="badge">Historical replay</span></div>
    <nav aria-label="Dashboard sections">{(data?.field?.length?[...tabs,'Client wells']:tabs).map(t=><button key={t} aria-current={tab===t?'page':undefined} onClick={()=>setTab(t)}>{t}</button>)}</nav>
    {error&&<div role="alert" className="notice">{error} <button onClick={refresh}>Refresh</button></div>}
    {!data&&!error&&<p>Loading model results…</p>}{data&&<>{tab===tabs[0]?<Wells data={data}/>:tab===tabs[1]?<Health rows={data.alerts}/>:tab===tabs[2]?<Allocation budget={data.budget}/>:<Field wells={data.field}/>}<footer>Replay generated {new Date(data.updated).toLocaleString()}</footer></>}</main></>;
}
