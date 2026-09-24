import { useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import "./styles.css";
import "./charts.css";
import "./auth.css";

const API = import.meta.env.VITE_FORESTWATCH_API || "http://127.0.0.1:8000";
const pct = (value) => `${Number(value || 0).toFixed(2)}%`;

function Dashboard({account, onLogout}) {
  const [section, setSection] = useState("GIS");
  const [dates, setDates] = useState([]);
  const [series, setSeries] = useState([]);
  const [earlier, setEarlier] = useState("");
  const [later, setLater] = useState("");
  const [summary, setSummary] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    fetch(`${API}/api/ndvi/dates`).then((r) => r.json()).then((payload) => {
      const unique = payload.dates || [];
      setDates(unique); setEarlier(unique[0] || ""); setLater(unique.at(-1) || "");
    }).catch(() => setError("Start the FastAPI service on port 8000 to load GIS dates."));
    // The historical chart is optional and can load after the date controls.
    fetch(`${API}/api/ndvi/timeseries`).then((r) => r.ok ? r.json() : []).then((rows) => setSeries(rows.filter((row) => Number.isFinite(row.ndvi_mean)))).catch(() => {});
  }, []);
  useEffect(() => {
    fetch(`${API}/api/alerts/check`, {method:"POST", headers:{Authorization:`Bearer ${account.token}`}})
      .then((r) => r.json()).then((result) => { if (result.email_delivery?.message) setError(result.email_delivery.message); })
      .catch(() => {});
  }, [account.token]);

  const load = async () => {
    if (!earlier || !later || earlier === later) return setError("Choose two different observation dates.");
    setLoading(true); setError("");
    try {
      const response = await fetch(`${API}/api/gis/dashboard?earlier_date=${earlier}&later_date=${later}`);
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail || "Unable to calculate the selected period.");
      setSummary(payload);
    } catch (e) { setError(e.message); } finally { setLoading(false); }
  };
  const refreshDashboard = async () => {
    setLoading(true); setError("");
    try {
      const [dateResponse, seriesResponse] = await Promise.all([
        fetch(`${API}/api/ndvi/dates`),
        fetch(`${API}/api/ndvi/timeseries`),
      ]);
      const datePayload = await dateResponse.json();
      const refreshedDates = datePayload.dates || [];
      setDates(refreshedDates);
      setEarlier((current) => refreshedDates.includes(current) ? current : (refreshedDates[0] || ""));
      setLater((current) => refreshedDates.includes(current) ? current : (refreshedDates.at(-1) || ""));
      if (seriesResponse.ok) {
        const rows = await seriesResponse.json();
        setSeries(rows.filter((row) => Number.isFinite(row.ndvi_mean)));
      }
    } catch { setError("Unable to refresh the satellite dataset. Check that the API is running."); }
    finally { setLoading(false); }
  };

  return <main className="app-shell">
    <header><div><span className="eyebrow">FORESTWATCH</span><h1>Forest Monitoring</h1><p>Signed in as {account.email}</p></div><div className="header-actions"><button className="secondary" onClick={refreshDashboard} disabled={loading}>↻ Refresh data</button><button className="secondary" onClick={onLogout}>Sign out</button></div></header>
    <nav className="section-nav">{["GIS","Upload images","Predictions","Alerts","Reports"].map(item=><button key={item} className={section===item?"active":"secondary"} onClick={()=>setSection(item)}>{item}</button>)}</nav>
    {section === "GIS" && <>
    <section className="controls"><label>Earlier observation<select value={earlier} onChange={(e) => setEarlier(e.target.value)}>{dates.map((d) => <option key={d}>{d}</option>)}</select></label><span className="arrow">→</span><label>Later observation<select value={later} onChange={(e) => setLater(e.target.value)}>{dates.map((d) => <option key={d}>{d}</option>)}</select></label><button onClick={load} disabled={loading}>{loading ? "Updating…" : "Update dashboard"}</button><div className="period">Analysis period<br/><strong>{summary?.period || `${earlier || "—"} to ${later || "—"}`}</strong></div></section>
    {error && <div className="notice">{error}</div>}
    <section className="metrics">
      <Metric title="Vegetation decrease" value={pct(summary?.decrease_pixel_percentage)} tone="red" detail="of Gorewada boundary" />
      <Metric title="Vegetation increase" value={pct(summary?.increase_pixel_percentage)} tone="green" detail="of Gorewada boundary" />
      <Metric title="Stable area" value={pct(summary?.stable_pixel_percentage)} tone="blue" detail="no detected change" />
      <Metric title="Change threshold" value="0.15" tone="yellow" detail="NDVI-index difference" />
    </section>
    <section className="dashboard-grid">
      <article className="panel map-panel"><div className="panel-head"><div><span className="eyebrow">ACTUAL SATELLITE IMAGERY</span><h2>Gorewada true-colour scenes</h2></div><span className="live">● Selected dates</span></div><div className="true-scenes"><figure><img src={earlier ? `${API}/api/gis/true-colour/${earlier}` : ""} alt={`True-colour Gorewada scene from ${earlier}`} /><figcaption>Earlier: {earlier || "—"}</figcaption></figure><figure><img src={later ? `${API}/api/gis/true-colour/${later}` : ""} alt={`True-colour Gorewada scene from ${later}`} /><figcaption>Later: {later || "—"}</figcaption></figure></div><p className="caption">These are the real true-colour satellite scenes selected above. Calculated NDVI-index change is reported in the adjacent panels; no schematic map or artificial imagery is used.</p></article>
      <article className="panel"><span className="eyebrow">CHANGE BREAKDOWN</span><h2>Selected-period distribution</h2><div className="bars"><Bar label="Decrease" value={summary?.decrease_pixel_percentage} color="#ef5e68"/><Bar label="Stable" value={summary?.stable_pixel_percentage} color="#6c8cff"/><Bar label="Increase" value={summary?.increase_pixel_percentage} color="#40c98a"/></div><p className="caption">{summary?.interpretation || "Choose dates and update the dashboard to calculate the distribution."}</p></article>
    </section>
    <section className="dashboard-grid charts"><article className="panel chart-panel"><span className="eyebrow">HISTORICAL OBSERVATIONS</span><h2>NDVI-index trend</h2><TrendChart values={series}/></article><article className="panel chart-panel"><span className="eyebrow">CURRENT PERIOD</span><h2>Change distribution</h2><Donut summary={summary}/></article></section>
    </>}
    {section === "Upload images" && <UploadSection />}
    {section === "Predictions" && <PredictionSection />}
    {section === "Alerts" && <AlertsSection token={account.token} />}
    {section === "Reports" && <ReportsSection />}
  </main>;
}
function App() {
  const [account, setAccount] = useState(() => { try { return JSON.parse(localStorage.getItem("forestwatch-user")); } catch { return null; } });
  return account ? <Dashboard account={account} onLogout={() => { localStorage.removeItem("forestwatch-user"); setAccount(null); }} /> : <Login onLogin={setAccount}/>;
}
function Login({onLogin}) {
  const [email, setEmail] = useState(""); const [password, setPassword] = useState(""); const [isRegister, setRegister] = useState(false); const [message, setMessage] = useState(""); const [busy, setBusy] = useState(false);
  const submit = async (event) => { event.preventDefault(); setBusy(true); setMessage(""); try { const r=await fetch(`${API}/api/auth/${isRegister ? "register" : "login"}`, {method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({email,password})}); const p=await r.json(); if(!r.ok) throw new Error(p.detail || "Unable to sign in."); localStorage.setItem("forestwatch-user",JSON.stringify(p)); onLogin(p); } catch(e) { setMessage(e.message); } finally { setBusy(false); } };
  return <main className="login-page"><form className="login-card" onSubmit={submit}><span className="eyebrow">FORESTWATCH</span><h1>Forest Monitoring</h1><p>Sign in to access the GIS dashboard and receive health alerts at this email address.</p><label>Email address<input type="email" value={email} onChange={e=>setEmail(e.target.value)} required /></label><label>Password<input type="password" value={password} onChange={e=>setPassword(e.target.value)} minLength="8" required /></label>{message && <div className="notice">{message}</div>}<button type="submit">{busy ? "Please wait…" : isRegister ? "Create account" : "Sign in"}</button><button className="link-button" type="button" onClick={()=>setRegister(!isRegister)}>{isRegister ? "Already have an account? Sign in" : "New user? Create an account"}</button></form></main>;
}
function Metric({ title, value, tone, detail }) { return <article className={`metric ${tone}`}><span>{title}</span><strong>{value}</strong><small>{detail}</small></article>; }
function Bar({ label, value, color }) { const safe = Number(value || 0); return <div className="bar-row"><span>{label}</span><div className="track"><i style={{width: `${Math.min(100, safe)}%`, background: color}} /></div><b>{pct(safe)}</b></div>; }
function TrendChart({values}) { if (!values.length) return <p className="caption">Historical NDVI observations will appear here.</p>; const ys=values.map(v=>v.ndvi_mean), lo=Math.min(...ys), hi=Math.max(...ys), pad=Math.max(.03,(hi-lo)*.12); const y=v=>145-((v-(lo-pad))/((hi+pad)-(lo-pad)||1))*120; const x=i=>32+i*(336/Math.max(1,values.length-1)); const points=values.map((v,i)=>`${x(i)},${y(v.ndvi_mean)}`).join(" "); return <svg className="trend-chart" viewBox="0 0 390 180" role="img" aria-label="Historical NDVI index trend"><line x1="32" y1="15" x2="32" y2="145"/><line x1="32" y1="145" x2="368" y2="145"/><polyline points={points}/><text x="4" y="20">{hi.toFixed(2)}</text><text x="4" y="148">{lo.toFixed(2)}</text><text x="32" y="170">{values[0].date}</text><text x="292" y="170">{values.at(-1).date}</text></svg>; }
function Donut({summary}) { const d=Number(summary?.decrease_pixel_percentage||0), s=Number(summary?.stable_pixel_percentage||0), i=Number(summary?.increase_pixel_percentage||0); const style={background:`conic-gradient(#ef5e68 0 ${d}%, #6c8cff ${d}% ${d+s}%, #40c98a ${d+s}% 100%)`}; return <div className="donut-wrap"><div className="donut" style={style}><div>{(d+s+i).toFixed(0)}%<small>boundary</small></div></div><div className="donut-key"><span>■ Decrease {pct(d)}</span><span>■ Stable {pct(s)}</span><span>■ Increase {pct(i)}</span></div></div>; }
function UploadSection(){const [files,setFiles]=useState(null),[type,setType]=useState("Sentinel-2 NDVI"),[date,setDate]=useState(""),[message,setMessage]=useState("");const submit=async e=>{e.preventDefault();if(!files?.length)return setMessage("Choose one or more image files.");const body=new FormData();[...files].forEach(f=>body.append("files",f));body.append("image_type",type);body.append("acquisition_date",date);const r=await fetch(`${API}/api/uploads`,{method:"POST",body});const p=await r.json();setMessage(r.ok?`Added ${p.added} image(s) dated ${p.date}.`:p.detail)};return <section className="feature-panel"><h2>Satellite image upload</h2><p>Upload dated Sentinel imagery into the local ForestWatch dataset.</p><form onSubmit={submit}><label>Images<input type="file" multiple accept="image/*,.tif,.tiff,.jp2" onChange={e=>setFiles(e.target.files)}/></label><label>Image type<select value={type} onChange={e=>setType(e.target.value)}>{["Sentinel-1 VV","Sentinel-1 VH","Sentinel-2 NDVI","Sentinel-2 True Color"].map(x=><option key={x}>{x}</option>)}</select></label><label>Acquisition date<input type="date" required value={date} onChange={e=>setDate(e.target.value)}/></label><button>Add images to dataset</button></form><p className="caption">{message}</p></section>}
function PredictionSection(){const [months,setMonths]=useState(6),[rows,setRows]=useState([]),[message,setMessage]=useState("");const run=async()=>{setMessage("Training local forecast model…");const r=await fetch(`${API}/api/predictions/forecast?horizon=${months}`);const p=await r.json();if(!r.ok)return setMessage(p.detail);setRows(p);setMessage(`Generated ${p.length} future prediction(s).`)};return <section className="feature-panel"><h2>Future NDVI prediction</h2><p>The backend trains a local Random Forest from historical observations before forecasting.</p><label>Months to predict<input type="number" min="1" max="24" value={months} onChange={e=>setMonths(e.target.value)}/></label><button onClick={run}>Generate future prediction</button><p className="caption">{message}</p><PredictionChart rows={rows}/><SimpleTable rows={rows} columns={["forecast_date","predicted_ndvi_percent"]}/></section>}
function PredictionChart({rows}){const points=rows.map(row=>({date:String(row.forecast_date||""),value:Number(row.predicted_ndvi_percent)})).filter(row=>Number.isFinite(row.value));if(!points.length)return <div className="prediction-empty">Generate a forecast to view its NDVI trend.</div>;const values=points.map(point=>point.value),low=Math.max(0,Math.floor(Math.min(...values)-5)),high=Math.min(100,Math.ceil(Math.max(...values)+5)),range=Math.max(1,high-low),x=index=>68+index*(690/Math.max(1,points.length-1)),y=value=>242-((value-low)/range)*184,polyline=points.map((point,index)=>`${x(index)},${y(point.value)}`).join(" ");const labels=[0,Math.floor((points.length-1)/2),points.length-1].filter((value,index,array)=>array.indexOf(value)===index);return <figure className="prediction-figure"><figcaption><strong>Predicted vegetation trend</strong><span>NDVI (%)</span></figcaption><svg className="prediction-chart" viewBox="0 0 800 300" role="img" aria-label="Line chart of future predicted NDVI percentage"><title>Future predicted NDVI</title><line className="prediction-grid" x1="68" y1="58" x2="758" y2="58"/><line className="prediction-grid" x1="68" y1="150" x2="758" y2="150"/><line className="prediction-grid" x1="68" y1="242" x2="758" y2="242"/><line className="prediction-axis" x1="68" y1="42" x2="68" y2="242"/><line className="prediction-axis" x1="68" y1="242" x2="758" y2="242"/><text x="12" y="62">{high.toFixed(0)}%</text><text x="12" y="154">{((low+high)/2).toFixed(0)}%</text><text x="12" y="246">{low.toFixed(0)}%</text><polyline className="prediction-line" points={polyline}/>{points.map((point,index)=><g key={`${point.date}-${index}`}><circle className="prediction-point" cx={x(index)} cy={y(point.value)} r="5"/><title>{`${point.date}: ${point.value.toFixed(2)}%`}</title></g>)}{labels.map(index=><text className="prediction-date" key={index} x={x(index)} y="278" textAnchor={index===0?"start":index===points.length-1?"end":"middle"}>{points[index].date}</text>)}</svg><div className="prediction-summary"><span>Lowest: <b>{Math.min(...values).toFixed(2)}%</b></span><span>Highest: <b>{Math.max(...values).toFixed(2)}%</b></span></div></figure>}
function AlertsSection({token}){const [rows,setRows]=useState([]),[message,setMessage]=useState("");const load=async()=>{try{const check=await fetch(`${API}/api/alerts/check`,{method:"POST",headers:{Authorization:`Bearer ${token}`}});const checkData=await check.json();const r=await fetch(`${API}/api/alerts`);setRows(await r.json());setMessage(checkData.email_delivery?.message||checkData.message||checkData.detail||"Unable to check email delivery.")}catch{setMessage("Unable to contact the alerts service.")}};useEffect(()=>{load()},[]);return <section className="feature-panel"><h2>Vegetation health alerts</h2><p>Alert threshold: health score below 35. Alerts are sent to the signed-in email when SMTP is configured.</p><button onClick={load}>Refresh alerts</button><p className="caption">{message}</p><SimpleTable rows={rows} columns={["date","severity","reason","status","recommended_action"]}/></section>}
function ReportsSection(){const [period,setPeriod]=useState("monthly"),[available,setAvailable]=useState({months:[],years:[]}),[selected,setSelected]=useState(""),[message,setMessage]=useState("");const loadPeriods=async()=>{try{setMessage("Loading available report dates…");const r=await fetch(`${API}/api/reports/available-periods`);const p=await r.json();if(!r.ok)throw new Error(p.detail||"Unable to load report dates.");const months=p.months||[],years=p.years||[];setAvailable({months,years});const choices=period==="monthly"?months:years;setSelected(choices.at(-1)||"");setMessage(choices.length?"":"No NDVI observations are available for this report type.")}catch(e){setMessage(e.message||"Start the backend to load available report dates.")}};useEffect(()=>{loadPeriods()},[]);const choices=period==="monthly"?available.months:available.years;const changePeriod=value=>{setPeriod(value);const next=value==="monthly"?available.months:available.years;setSelected(next.at(-1)||"")};const download=format=>{if(!selected)return setMessage("Choose a report date first.");window.open(`${API}/api/reports/${period}/download?format=${format}&report_date=${encodeURIComponent(selected)}`,format==="html"?"_blank":"_self")};return <section className="feature-panel"><h2>Monitoring reports</h2><p>Choose the month or year before downloading a report. Only periods that contain NDVI observations are shown.</p><div className="report-picker"><label>Report type<select value={period} onChange={e=>changePeriod(e.target.value)}><option value="monthly">Monthly</option><option value="yearly">Yearly</option></select></label><label>{period==="monthly"?"Month":"Year"}<select value={selected} onChange={e=>setSelected(e.target.value)}><option value="">Choose {period==="monthly"?"month":"year"}</option>{choices.map(value=><option key={value} value={value}>{value}</option>)}</select></label><button className="secondary" onClick={loadPeriods}>↻ Load dates</button></div><div className="report-grid"><article><h3>{period==="monthly"?"Monthly":"Yearly"} report: {selected||"—"}</h3><button onClick={()=>download("html")}>Download HTML report</button><button className="secondary" onClick={()=>download("csv")}>Download CSV metrics</button></article></div><p className="caption">{message}</p></section>}
function SimpleTable({rows,columns}){if(!rows?.length)return null;return <div className="table-wrap"><table><thead><tr>{columns.map(c=><th key={c}>{c.replaceAll("_"," ")}</th>)}</tr></thead><tbody>{rows.map((row,i)=><tr key={i}>{columns.map(c=><td key={c}>{row[c] ?? "—"}</td>)}</tr>)}</tbody></table></div>}
createRoot(document.getElementById("root")).render(<App />);
