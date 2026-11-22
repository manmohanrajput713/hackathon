'use client';
import {useState} from 'react';

export function Field({wells}){
  const [id,select]=useState(0),[index,setIndex]=useState(null);
  const well=wells[id], last=well.records.length-1, row=well.records[index??last];
  const fmt=x=>x==null?'Unavailable':x.toLocaleString('en-US',{maximumFractionDigits:2});
  return <section><div className="sectionHead"><div><small>Matched field history</small><h2>{well.name}</h2><p>{row.timestamp}</p></div><label>Client well<select value={id} onChange={e=>{select(+e.target.value);setIndex(null)}}>{wells.map((w,i)=><option key={w.name} value={i}>{w.name}</option>)}</select></label></div>
  <div className="metrics">{[['Well test',row.test],['Virtual flow meter',row.vfm],['Deep-CRM',row.crm]].map(([k,v])=><div key={k}><small>{k}</small><strong>{fmt(v)} {v!=null&&<em>m³/h</em>}</strong></div>)}</div>
  <label className="slider">Observation {1+(index??last)} of {well.records.length}<input type="range" min="0" max={last} value={index??last} onChange={e=>setIndex(+e.target.value)}/></label>
  <p>Reservoir forecast time {row.crm_at||'unavailable'}</p></section>;
}

