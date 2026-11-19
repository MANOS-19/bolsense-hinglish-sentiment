import React,{ useState } from "react";
import { ArrowUpRight, BrainCircuit, Sparkles, RotateCcw, Clipboard, Check, LoaderCircle, MessageCircle, Languages } from "lucide-react";
import "./style.css";

const API = "http://localhost:8000";
const examples = [
  "Yeh movie bahut achi hai, I really loved it!",
  "Mujhe yeh service bilkul pasand nahi aayi.",
  "Theek tha, nothing special honestly."
];
export default function App() {
  const [text,setText] = useState("");
  const [result,setResult] = useState(null);
  const [loading,setLoading] = useState(false);
  const [error,setError] = useState("");
  const [copied,setCopied] = useState(false);

  async function analyze(value=text) {
    const input=value.trim();
    if(input.length<2){setError("Write at least two characters before analyzing.");return;}
    setLoading(true);setError("");setResult(null);
    try{
      const response=await fetch(`${API}/api/analyze`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({text:input})});
      const data=await response.json();
      if(!response.ok)throw new Error(data.detail||"Analysis failed. Is the backend running?");
      setResult(data);
    }catch(e){setError(e.message);}
    finally{setLoading(false);}
  }
  async function copyResult(){
    if(!result)return;
    await navigator.clipboard.writeText(`${result.sentiment} (${result.confidence}% confidence)\n${result.text}`);
    setCopied(true);setTimeout(()=>setCopied(false),1600);
  }
  const tone=result?.sentiment.toLowerCase();
  return <div className="app-shell">
    <header className="nav"><a className="wordmark" href="#"><span className="mark"><BrainCircuit size={20}/></span> bol<span>sense.</span></a><div className="nav-caption">LANGUAGE · CONTEXT · SENTIMENT</div><span className="version">NLP LAB / 01</span></header>
    <main className="layout">
      <section className="hero"><div className="kicker"><Sparkles size={14}/> UNDERSTAND THE FEELING BEHIND THE WORDS</div><h1>Hinglish has a<br/><em>feeling of its own.</em></h1><p>Analyze the sentiment of Hindi-English mixed text with a multilingual AI model. Write naturally — just the way you speak.</p></section>
      <section className="workspace">
        <div className="input-panel">
          <div className="panel-head"><div><span className="step">01 / INPUT</span><h2>Your text, your words.</h2></div><Languages size={22}/></div>
          <textarea value={text} onChange={e=>{setText(e.target.value);setError("");}} maxLength={2000} placeholder="Example: Aaj ka din bahut amazing tha, totally made me happy..."/>
          <div className="input-meta"><span>{text.length}/2000 characters</span><span>Hindi + English supported</span></div>
          <div className="examples"><span>TRY AN EXAMPLE</span>{examples.map(x=><button key={x} onClick={()=>{setText(x);setResult(null);setError("");}}>{x}</button>)}</div>
          {error&&<div className="error">{error}</div>}
          <button className="analyze" onClick={()=>analyze()} disabled={loading}>{loading?<><LoaderCircle className="spin"/> Analyzing...</>:<>Analyze sentiment <ArrowUpRight size={18}/></>}</button>
        </div>
        <div className="result-panel">
          <div className="panel-head"><div><span className="step">02 / INSIGHT</span><h2>What does it feel like?</h2></div><MessageCircle size={22}/></div>
          {!result&&!loading&&<div className="empty"><div className="empty-orbit"><BrainCircuit size={42}/></div><h3>Your insight will appear here.</h3><p>Enter a Hinglish sentence and run the analysis to reveal its predicted sentiment.</p></div>}
          {loading&&<div className="empty"><LoaderCircle className="spin" size={38}/><h3>Reading between the words...</h3><p>The model is looking at the overall tone of your text.</p></div>}
          {result&&<div className={`result-content ${tone}`}><span className="result-label">PREDICTED SENTIMENT</span><div className="sentiment">{result.sentiment}</div><p className="result-copy">The model reads this text as <b>{result.sentiment.toLowerCase()}</b>.</p><div className="confidence"><div className="confidence-head"><span>MODEL CONFIDENCE</span><b>{result.confidence}%</b></div><div className="track"><i style={{width:`${result.confidence}%`}}/></div></div><div className="analyzed-text"><span>ANALYZED TEXT</span><p>{result.text}</p></div><button className="copy" onClick={copyResult}>{copied?<Check size={15}/>:<Clipboard size={15}/>} {copied?"Copied":"Copy result"}</button><p className="disclaimer">{result.disclaimer}</p></div>}
        </div>
      </section>
      <div className="bottom-note"><span>BUILT FOR THE WAY WE COMMUNICATE.</span><span>POWERED BY MULTILINGUAL NLP <ArrowUpRight size={13}/></span></div>
    </main>
  </div>;
}
