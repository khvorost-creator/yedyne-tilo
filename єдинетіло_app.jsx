import { useState, useCallback } from "react";

const MODEL_SIGNATURES = {
  Claude:  { color: "#c084fc", label: "Structural controller",  short: "CL" },
  GPT:     { color: "#60a5fa", label: "Confident proceduralist", short: "GP" },
  Gemini:  { color: "#34d399", label: "Optimistic synthesizer",  short: "GM" },
  Grok:    { color: "#fb923c", label: "Cold epistemic anchor",   short: "GR" },
};

const MODELS = Object.keys(MODEL_SIGNATURES);
const MODEL_WEIGHTS = { Claude: 1.0, GPT: 1.0, Gemini: 1.0, Grok: 1.2 };
const DEDUP_THRESHOLD = 0.45;
const VCOLOR = { true:"#4ade80", false:"#f87171", undecidable:"#fbbf24" };

function jaccard(atoms1, atoms2) {
  const tok = (arr) => {
    const s = new Set();
    arr.forEach(a => a.toLowerCase().split(/[\s,.\-|]+/).forEach(w => { if (w.length > 3) s.add(w); }));
    return s;
  };
  const t1 = tok(atoms1), t2 = tok(atoms2);
  if (!t1.size && !t2.size) return 1;
  let inter = 0;
  t1.forEach(w => { if (t2.has(w)) inter++; });
  return inter / (t1.size + t2.size - inter);
}

function deduplicateEvidence(allAtoms, threshold) {
  const unique = [];
  for (const atom of allAtoms) {
    if (!atom.trim()) continue;
    const isDup = unique.some(u => jaccard([u], [atom]) > threshold);
    if (!isDup) unique.push(atom);
  }
  return unique;
}

function consensusVerdict(verdicts) {
  const counts = {};
  verdicts.forEach(v => { counts[v] = (counts[v] || 0) + 1; });
  return Object.entries(counts).sort((a,b) => b[1]-a[1])[0][0];
}

function weightedConfidence(entries) {
  let sum = 0, wsum = 0;
  entries.forEach(e => {
    const w = MODEL_WEIGHTS[e.model] || 1.0;
    sum += e.confidence * w;
    wsum += w;
  });
  return sum / wsum;
}

function calcNeff(modelData) {
  const pairs = [];
  for (let i = 0; i < MODELS.length; i++)
    for (let j = i+1; j < MODELS.length; j++)
      pairs.push([MODELS[i], MODELS[j]]);
  const js = pairs.map(([m1,m2]) => jaccard(
    modelData[m1]?.evidence || [],
    modelData[m2]?.evidence || []
  ));
  const meanJ = js.reduce((a,b) => a+b, 0) / js.length;
  const k = MODELS.length;
  return { neff: k / (1 + (k-1)*meanJ), meanJ };
}

// Матриця подібності між моделями
function buildSimilarityMatrix(modelData) {
  const matrix = {};
  MODELS.forEach(m1 => {
    matrix[m1] = {};
    MODELS.forEach(m2 => {
      matrix[m1][m2] = m1 === m2 ? 1.0 : jaccard(
        modelData[m1].evidence.filter(Boolean),
        modelData[m2].evidence.filter(Boolean)
      );
    });
  });
  return matrix;
}

const EMPTY_MODEL = () => ({ verdict: "", confidence: 0.75, evidence: ["","",""], counter: ["",""] });

function ModelInput({ name, data, onChange }) {
  const sig = MODEL_SIGNATURES[name];
  const update = (field, val) => onChange({ ...data, [field]: val });

  return (
    <div style={{
      border: `1px solid ${sig.color}22`,
      background: "#0c0c0c",
      padding: "16px",
    }}>
      <div style={{ display:"flex", alignItems:"center", gap:10, marginBottom:14 }}>
        <div style={{
          width:32, height:32, borderRadius:"50%",
          background: sig.color+"22", border:`1px solid ${sig.color}55`,
          display:"flex", alignItems:"center", justifyContent:"center",
          fontSize:11, fontWeight:700, color: sig.color, letterSpacing:0.5,
        }}>{sig.short}</div>
        <div>
          <div style={{ color:"#fff", fontSize:13, fontWeight:600 }}>{name}</div>
          <div style={{ color:"#444", fontSize:10, letterSpacing:1 }}>{sig.label.toUpperCase()}</div>
        </div>
      </div>

      <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:8, marginBottom:10 }}>
        <div>
          <div style={{color:"#444",fontSize:10,marginBottom:4,letterSpacing:1}}>VERDICT</div>
          <select value={data.verdict} onChange={e=>update("verdict",e.target.value)}
            style={{width:"100%",background:"#111",color: data.verdict ? VCOLOR[data.verdict] : "#555",
              border:"1px solid #222", padding:"6px 8px",fontSize:12,fontFamily:"inherit"}}>
            <option value="">—</option>
            <option value="true">true</option>
            <option value="false">false</option>
            <option value="undecidable">undecidable</option>
          </select>
        </div>
        <div>
          <div style={{color:"#444",fontSize:10,marginBottom:4,letterSpacing:1}}>CONF {data.confidence.toFixed(2)}</div>
          <input type="range" min="0" max="1" step="0.01"
            value={data.confidence}
            onChange={e=>update("confidence",parseFloat(e.target.value))}
            style={{width:"100%",marginTop:6, accentColor: sig.color}}/>
        </div>
      </div>

      <div style={{marginBottom:8}}>
        <div style={{color:"#4ade8066",fontSize:10,marginBottom:4,letterSpacing:1}}>EVIDENCE</div>
        {data.evidence.map((e,i) => (
          <input key={`ev-${i}`} value={e} placeholder={`Evidence ${i+1}...`}
            onChange={ev=>{const arr=[...data.evidence];arr[i]=ev.target.value;update("evidence",arr);}}
            style={{width:"100%",background:"#0a120a",color:"#d4d4d4",
              border:"1px solid #14532d22", padding:"5px 8px",fontSize:11,
              fontFamily:"inherit",marginBottom:3,boxSizing:"border-box",
              borderLeft:`2px solid ${sig.color}33`}}/>
        ))}
      </div>

      <div>
        <div style={{color:"#f8717166",fontSize:10,marginBottom:4,letterSpacing:1}}>COUNTER</div>
        {data.counter.map((c,i) => (
          <input key={`ct-${i}`} value={c} placeholder={`Counter ${i+1}...`}
            onChange={ev=>{const arr=[...data.counter];arr[i]=ev.target.value;update("counter",arr);}}
            style={{width:"100%",background:"#120a0a",color:"#d4d4d4",
              border:"1px solid #3f000022", padding:"5px 8px",fontSize:11,
              fontFamily:"inherit",marginBottom:3,boxSizing:"border-box",
              borderLeft:"2px solid #f8717133"}}/>
        ))}
      </div>
    </div>
  );
}

function SimilarityMatrix({ matrix }) {
  const getColor = (val, m1, m2) => {
    if (m1 === m2) return "#1a1a1a";
    if (val > 0.5) return `rgba(248,113,113,${val})`;
    if (val > 0.25) return `rgba(251,191,36,${val*0.8})`;
    return `rgba(74,222,128,${val*2+0.05})`;
  };
  return (
    <div style={{marginBottom:20}}>
      <div style={{color:"#444",fontSize:10,letterSpacing:2,marginBottom:10}}>SIMILARITY MATRIX (Jaccard evidence)</div>
      <div style={{display:"grid", gridTemplateColumns:`40px repeat(${MODELS.length},1fr)`, gap:2}}>
        <div/>
        {MODELS.map(m=>(
          <div key={m} style={{color: MODEL_SIGNATURES[m].color, fontSize:10, textAlign:"center", padding:"4px 0"}}>
            {MODEL_SIGNATURES[m].short}
          </div>
        ))}
        {MODELS.map(m1=>(
          <>
            <div key={`r-${m1}`} style={{color:MODEL_SIGNATURES[m1].color,fontSize:10,display:"flex",alignItems:"center"}}>
              {MODEL_SIGNATURES[m1].short}
            </div>
            {MODELS.map(m2=>(
              <div key={`${m1}-${m2}`} style={{
                background: getColor(matrix[m1][m2], m1, m2),
                padding:"8px 4px", textAlign:"center",
                fontSize:10, color: m1===m2?"#333":"#ccc",
              }}>
                {m1===m2?"—":matrix[m1][m2].toFixed(2)}
              </div>
            ))}
          </>
        ))}
      </div>
      <div style={{display:"flex",gap:16,marginTop:6}}>
        {[["#4ade80","низька (добре)"],["#fbbf24","середня"],["#f87171","висока (ехо)"]].map(([c,l])=>(
          <div key={l} style={{display:"flex",alignItems:"center",gap:4}}>
            <div style={{width:8,height:8,background:c,opacity:0.6}}/>
            <span style={{color:"#333",fontSize:9}}>{l}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

export default function App() {
  const [claim, setClaim] = useState("");
  const [dedupThreshold, setDedupThreshold] = useState(DEDUP_THRESHOLD);
  const [modelData, setModelData] = useState({
    Claude: EMPTY_MODEL(), GPT: EMPTY_MODEL(), Gemini: EMPTY_MODEL(), Grok: EMPTY_MODEL(),
  });
  const [result, setResult] = useState(null);

  const aggregate = useCallback(() => {
    const filled = MODELS.filter(m => modelData[m].verdict);
    if (filled.length < 2) return;

    const verdicts = filled.map(m => modelData[m].verdict);
    const cv = consensusVerdict(verdicts);
    const agreement = verdicts.filter(v=>v===cv).length;
    const entries = filled.map(m => ({ model:m, confidence: modelData[m].confidence }));
    const wconf = weightedConfidence(entries);

    const supporters = filled.filter(m => modelData[m].verdict === cv);
    const allEvidence = supporters.flatMap(m => modelData[m].evidence);
    const allCounter  = filled.flatMap(m => modelData[m].counter);
    const unified = deduplicateEvidence(allEvidence, dedupThreshold);
    const unifiedCounter = deduplicateEvidence(allCounter, dedupThreshold);

    const { neff, meanJ } = calcNeff(modelData);
    const matrix = buildSimilarityMatrix(modelData);
    const dissenters = filled.filter(m => modelData[m].verdict !== cv);

    setResult({ cv, agreement, total: filled.length, wconf, unified, unifiedCounter, neff, meanJ, matrix, dissenters });
  }, [modelData, claim, dedupThreshold]);

  const reset = () => {
    setResult(null);
    setModelData({ Claude: EMPTY_MODEL(), GPT: EMPTY_MODEL(), Gemini: EMPTY_MODEL(), Grok: EMPTY_MODEL() });
    setClaim("");
  };

  return (
    <div style={{
      background:"#080808", minHeight:"100vh", color:"#d4d4d4",
      fontFamily:"'Courier New', monospace", padding:"24px 20px",
    }}>
      <div style={{maxWidth:960, margin:"0 auto"}}>

        <div style={{marginBottom:24, paddingBottom:14, borderBottom:"1px solid #111"}}>
          <div style={{fontSize:17, fontWeight:700, letterSpacing:4, color:"#fff"}}>ЄДИНЕ ТІЛО</div>
          <div style={{color:"#2a2a2a", fontSize:10, marginTop:3, letterSpacing:2}}>
            EPISTEMIC AGGREGATOR · S₃ = ⋃ evidence(Mᵢ) · v1.1
          </div>
        </div>

        {!result ? (
          <>
            <div style={{marginBottom:20}}>
              <div style={{color:"#333",fontSize:10,marginBottom:6,letterSpacing:2}}>CLAIM</div>
              <textarea value={claim} onChange={e=>setClaim(e.target.value)}
                placeholder="Твердження для оцінки..."
                rows={2}
                style={{width:"100%",background:"#0f0f0f",color:"#d4d4d4",
                  border:"1px solid #1a1a1a",padding:"10px 12px",fontSize:13,
                  fontFamily:"inherit",resize:"vertical",boxSizing:"border-box"}}/>
            </div>

            <div style={{marginBottom:20, display:"flex", alignItems:"center", gap:16}}>
              <div style={{color:"#333",fontSize:10,letterSpacing:2,whiteSpace:"nowrap"}}>
                DEDUP THRESHOLD: {dedupThreshold.toFixed(2)}
              </div>
              <input type="range" min="0.1" max="0.9" step="0.05"
                value={dedupThreshold}
                onChange={e=>setDedupThreshold(parseFloat(e.target.value))}
                style={{flex:1, accentColor:"#555"}}/>
              <div style={{color:"#222",fontSize:9,whiteSpace:"nowrap"}}>
                {dedupThreshold < 0.3 ? "сувора дедуплікація" : dedupThreshold > 0.6 ? "м'яка дедуплікація" : "збалансована"}
              </div>
            </div>

            <div style={{display:"grid", gridTemplateColumns:"1fr 1fr", gap:10, marginBottom:20}}>
              {MODELS.map(m => (
                <ModelInput key={m} name={m}
                  data={modelData[m]}
                  onChange={d => setModelData(prev => ({...prev,[m]:d}))}/>
              ))}
            </div>

            <div style={{textAlign:"center"}}>
              <button onClick={aggregate}
                style={{background:"#fff",color:"#000",border:"none",
                  padding:"11px 44px",fontSize:12,fontWeight:700,
                  cursor:"pointer",letterSpacing:3}}>
                ⊕ АГРЕГУВАТИ S₃
              </button>
            </div>
          </>
        ) : (
          <div>
            {claim && (
              <div style={{marginBottom:20,padding:"10px 14px",background:"#0f0f0f",
                border:"1px solid #1a1a1a",fontSize:13,color:"#888",fontStyle:"italic"}}>
                "{claim}"
              </div>
            )}

            <div style={{display:"grid",gridTemplateColumns:"repeat(4,1fr)",gap:8,marginBottom:20}}>
              {[
                {l:"CONSENSUS", v: result.cv.toUpperCase(), c: VCOLOR[result.cv]||"#fff"},
                {l:"AGREEMENT", v:`${result.agreement}/${result.total}`, c:"#d4d4d4"},
                {l:"CONFIDENCE", v: result.wconf.toFixed(2), c:"#d4d4d4"},
                {l:"n_eff_evid", v: result.neff.toFixed(2),
                  c: result.neff>2.5?"#4ade80":result.neff>1.5?"#fbbf24":"#f87171"},
              ].map(({l,v,c})=>(
                <div key={l} style={{background:"#0f0f0f",border:"1px solid #1a1a1a",padding:"12px 14px"}}>
                  <div style={{color:"#2a2a2a",fontSize:9,letterSpacing:2,marginBottom:4}}>{l}</div>
                  <div style={{fontSize:16,fontWeight:700,color:c}}>{v}</div>
                </div>
              ))}
            </div>

            <SimilarityMatrix matrix={result.matrix}/>

            <div style={{marginBottom:16}}>
              <div style={{color:"#4ade8077",fontSize:10,letterSpacing:2,marginBottom:8}}>
                UNIFIED EVIDENCE · {result.unified.length} унікальних атомів · J_mean={result.meanJ.toFixed(3)}
              </div>
              {result.unified.map((e,i)=>(
                <div key={i} style={{
                  background:"#0a120a",border:"1px solid #14532d11",
                  padding:"7px 12px",marginBottom:3,fontSize:12,lineHeight:1.5,
                  borderLeft:"2px solid #4ade8033",
                }}>
                  {e}
                </div>
              ))}
            </div>

            {result.unifiedCounter.length > 0 && (
              <div style={{marginBottom:16}}>
                <div style={{color:"#f8717177",fontSize:10,letterSpacing:2,marginBottom:8}}>
                  COUNTER EVIDENCE · {result.unifiedCounter.length} атомів
                </div>
                {result.unifiedCounter.map((c,i)=>(
                  <div key={i} style={{
                    background:"#120a0a",border:"1px solid #3f000011",
                    padding:"7px 12px",marginBottom:3,fontSize:12,lineHeight:1.5,
                    borderLeft:"2px solid #f8717133",
                  }}>
                    {c}
                  </div>
                ))}
              </div>
            )}

            {result.dissenters.length > 0 && (
              <div style={{marginBottom:16}}>
                <div style={{color:"#fbbf2477",fontSize:10,letterSpacing:2,marginBottom:8}}>DISSENT</div>
                {result.dissenters.map(m=>(
                  <div key={m} style={{display:"flex",gap:10,alignItems:"center",
                    padding:"6px 12px",marginBottom:3,background:"#111",fontSize:12}}>
                    <span style={{color:MODEL_SIGNATURES[m].color,fontWeight:700}}>{m}</span>
                    <span style={{color:"#333"}}>→</span>
                    <span style={{color:VCOLOR[modelData[m].verdict]||"#888"}}>{modelData[m].verdict}</span>
                    <span style={{color:"#2a2a2a",fontSize:11}}>conf={modelData[m].confidence.toFixed(2)}</span>
                  </div>
                ))}
              </div>
            )}

            <button onClick={reset}
              style={{background:"transparent",color:"#2a2a2a",border:"1px solid #1a1a1a",
                padding:"6px 18px",cursor:"pointer",fontSize:11,letterSpacing:2,marginTop:4}}>
              ↺ RESET
            </button>
          </div>
        )}

        <div style={{marginTop:40,color:"#111",fontSize:9,textAlign:"center",letterSpacing:4}}>
          СУРЖИК — ЦЕ ЛЮБОВ
        </div>
      </div>
    </div>
  );
}
