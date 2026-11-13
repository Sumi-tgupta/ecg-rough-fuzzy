// Browser port of the ecg-rough-fuzzy pipeline: preprocess -> R-peaks -> beat features -> rough-set reduct -> fuzzy rules.
(function(G){
const FS=360,PRE=90,POST=144,L=PRE+POST,CLS=['N','S','V','F','Q'],K=5,TN=['Low','Med','High'];
const CN={N:'Normal / bundle-branch block',S:'Supraventricular ectopic',V:'Ventricular ectopic',F:'Fusion',Q:'Paced / unknown'};
const FN=['rr_prev','rr_next','rr_mean','rr_ratio_prev','rr_ratio_next','r_amp','qrs_dur','energy','variance','skewness','kurtosis','ptp','area','mean_pre_r','mean_post_r','t_wave_amp','dom_freq','psd_0p5_5','psd_5_15','psd_15_40','spec_entropy','wav_a4','wav_d4','wav_d3','wav_d2'];
const DOM=n=>/^rr_/.test(n)?'RR / rhythm':/psd|freq|entropy/.test(n)?'Frequency':/^wav/.test(n)?'Wavelet (Haar)':/variance|skew|kurt/.test(n)?'Statistical':'Time / morphology';
const rng=s=>()=>{s|=0;s=s+0x6D2B79F5|0;let t=Math.imul(s^s>>>15,1|s);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296};
const gs=r=>Math.sqrt(-2*Math.log(r()+1e-12))*Math.cos(6.2832*r());
const med=a=>{const s=Float64Array.from(a).sort();return s[s.length>>1]||0};
// ---------- synthetic ECG (used for in-browser training + demo) ----------
const MORPH={N:{w:[[-.16,.15,.025],[-.03,-.1,.01],[0,1,.012],[.03,-.2,.01],[.22,.3,.045]],rr:1},
S:{w:[[-.12,.06,.03],[-.03,-.08,.01],[0,.95,.013],[.03,-.18,.01],[.2,.28,.045]],rr:.68,next:1.2},
V:{w:[[0,1.25,.034],[.07,-.5,.03],[.26,-.5,.07]],rr:.68,next:1.35},
F:{w:[[-.12,.08,.03],[0,1.1,.024],[.05,-.25,.02],[.24,.1,.05]],rr:.86,next:1.1},
Q:{w:[[-.045,.6,.004],[0,.9,.03],[.08,-.2,.03],[.25,-.2,.06]],rr:1}};
const SCEN={normal:{N:1,S:0,V:0,F:0,Q:0},pvc:{N:.65,S:0,V:.35,F:0,Q:0},apc:{N:.7,S:.3,V:0,F:0,Q:0},paced:{N:.3,S:0,V:0,F:0,Q:.7},mixed:{N:.4,S:.2,V:.22,F:.1,Q:.08}};
function synth(seed,dur,mix,noise){const r=rng(seed),n=Math.round(dur*FS),x=new Float64Array(n),ann=[],base=.62+.35*r(),amp=.8+.5*r(),cum=[];let s=0;
 for(const c of CLS){s+=mix[c];cum.push(s)}
 let t=.2,pend=1;
 for(;;){const u=r()*s,c=CLS[cum.findIndex(v=>u<=v)]||'N',m=MORPH[c];t+=base*m.rr*pend*(1+.05*gs(r));pend=m.next||1;if(t>dur-.5)break;
  ann.push({i:Math.round(t*FS),c});const lo=Math.max(0,Math.round((t-.5)*FS)),hi=Math.min(n,Math.round((t+.55)*FS));
  const ws=m.w.map(([d,a,w])=>[d+.008*gs(r),a*(1+.12*gs(r)),w*(1+.1*gs(r))]);
  for(let i=lo;i<hi;i++){const tt=i/FS-t;let v=0;for(const[d,a,w]of ws)v+=a*Math.exp(-(tt-d)*(tt-d)/(2*w*w));x[i]+=amp*v}}
 const ph=r()*6.28,A=.25+.3*r();
 for(let i=0;i<n;i++){const tt=i/FS;x[i]+=A*Math.sin(6.2832*.25*tt+ph)+.15*Math.sin(6.2832*.05*tt)+noise*gs(r)+.03*Math.sin(6.2832*50*tt)}
 return{x,ann,fs:FS}}
// ---------- preprocessing ----------
function bq(type,f0,fs,Q=.7071){const w=6.2832*f0/fs,c=Math.cos(w),a=Math.sin(w)/(2*Q);let b0,b1;if(type==='lp'){b0=(1-c)/2;b1=1-c}else{b0=(1+c)/2;b1=-(1+c)}const a0=1+a;return[b0/a0,b1/a0,b0/a0,-2*c/a0,(1-a)/a0]}
function biq(x,[b0,b1,b2,a1,a2]){const y=new Float64Array(x.length);let x1=0,x2=0,y1=0,y2=0;for(let i=0;i<x.length;i++){const v=b0*x[i]+b1*x1+b2*x2-a1*y1-a2*y2;y[i]=v;x2=x1;x1=x[i];y2=y1;y1=v}return y}
function filtfilt(x,cs,fs){const n=x.length,p=Math.min(n-1,2*fs|0),e=new Float64Array(n+2*p);
 for(let i=0;i<p;i++){e[i]=2*x[0]-x[p-i];e[n+p+i]=2*x[n-1]-x[n-2-i]}e.set(x,p);
 let y=e;for(const c of cs)y=biq(y,c);y=y.slice().reverse();for(const c of cs)y=biq(y,c);y=y.reverse();return y.slice(p,p+n)}
function medfilt(x,k){const n=x.length,h=k>>1,o=new Float64Array(n);for(let i=0;i<n;i++){const a=x.slice(Math.max(0,i-h),Math.min(n,i+h+1)).sort();o[i]=a[a.length>>1]}return o}
function removeBaseline(x,fs){const d=4,m=Math.ceil(x.length/d),xd=new Float64Array(m);for(let i=0;i<m;i++)xd[i]=x[i*d];
 const b=medfilt(medfilt(xd,(.2*fs/d|0)|1),(.6*fs/d|0)|1),o=new Float64Array(x.length);
 for(let i=0;i<x.length;i++){const j=Math.min(i/d|0,m-2),f=i/d-j;o[i]=x[i]-(m>1?b[j]*(1-f)+b[j+1]*f:b[0])}return o}
function resample(x,fi){if(Math.abs(fi-FS)<1e-6)return x;const n=Math.round(x.length*FS/fi),o=new Float64Array(n);for(let i=0;i<n;i++){const p=i*fi/FS,j=Math.min(Math.floor(p),x.length-2),f=p-j;o[i]=x[j]*(1-f)+x[j+1]*f}return o}
function clean(a){const x=Float64Array.from(a);let last=0;for(let i=0;i<x.length;i++){if(Number.isFinite(x[i]))last=x[i];else x[i]=last}return x}
// ---------- R-peak detection (Pan-Tompkins style) ----------
function detectR(x,fs){const f=filtfilt(x,[bq('hp',5,fs),bq('lp',15,fs)],fs),n=x.length,sq=new Float64Array(n);
 for(let i=1;i<n-1;i++){const d=(f[i+1]-f[i-1])/2;sq[i]=d*d}
 const w=Math.max(1,.12*fs|0),cs=new Float64Array(n+1);for(let i=0;i<n;i++)cs[i+1]=cs[i]+sq[i];
 const ig=new Float64Array(n);for(let i=0;i<n;i++){const a=Math.max(0,i-(w>>1)),b=Math.min(n,i+(w>>1)+1);ig[i]=(cs[b]-cs[a])/(b-a)}
 const thr=.15*Float64Array.from(ig).sort()[Math.floor(.995*(n-1))],dist=.25*fs|0,pk=[];
 for(let i=1;i<n-1;i++)if(ig[i]>thr&&ig[i]>=ig[i-1]&&ig[i]>ig[i+1]){if(pk.length&&i-pk[pk.length-1]<dist){if(ig[i]>ig[pk[pk.length-1]])pk[pk.length-1]=i}else pk.push(i)}
 const hw=.05*fs|0,out=new Set();for(const p of pk){let b=Math.max(0,p-hw),bv=-1;for(let i=b;i<Math.min(n,p+hw+1);i++)if(Math.abs(x[i])>bv){bv=Math.abs(x[i]);b=i}out.add(b)}
 return[...out].sort((a,b)=>a-b)}
// ---------- features ----------
const HAN=Array.from({length:L},(_,i)=>.5-.5*Math.cos(6.2832*i/(L-1))),FQ=Array.from({length:80},(_,k)=>.5+.5*k);
const CT=FQ.map(f=>Float64Array.from({length:L},(_,j)=>Math.cos(6.2832*f*j/FS))),ST=FQ.map(f=>Float64Array.from({length:L},(_,j)=>Math.sin(6.2832*f*j/FS)));
function bfeat(b){const r0=PRE,o={};o.r_amp=b[r0];
 const d=[];for(let i=0;i<L-1;i++)d.push(Math.abs(b[i+1]-b[i]));const w=.14*FS|0,lo=Math.max(r0-w,0),hi=Math.min(r0+w,d.length),seg=d.slice(lo,hi),th=.2*Math.max(...seg);
 let fi=seg.findIndex(v=>v>th),la=seg.length-1;while(la>0&&seg[la]<=th)la--;o.qrs_dur=(la-fi+1)/FS;
 const mean=b.reduce((a,v)=>a+v,0)/L;let m2=0,m3=0,m4=0,en=0,ar=0,mn=1e9,mx=-1e9;
 for(const v of b){const e=v-mean;m2+=e*e;m3+=e**3;m4+=e**4;en+=v*v;ar+=Math.abs(v);mn=Math.min(mn,v);mx=Math.max(mx,v)}
 m2/=L;m3/=L;m4/=L;const sd=Math.sqrt(m2)||1;o.energy=en/L;o.variance=m2;o.skewness=m3/sd**3;o.kurtosis=m4/sd**4-3;o.ptp=mx-mn;o.area=ar/FS;
 const av=a=>a.reduce((s,v)=>s+v,0)/a.length;o.mean_pre_r=av(b.slice(0,r0-(.05*FS|0)));o.mean_post_r=av(b.slice(r0+(.05*FS|0)));
 const ts=b.slice(r0+(.12*FS|0),r0+(.38*FS|0));let ti=0;ts.forEach((v,i)=>{if(Math.abs(v)>Math.abs(ts[ti]))ti=i});o.t_wave_amp=ts[ti];
 const bw=b.map((v,i)=>(v-mean)*HAN[i]),P=FQ.map((_,k)=>{let c=0,s=0;for(let j=0;j<L;j++){c+=bw[j]*CT[k][j];s+=bw[j]*ST[k][j]}return(c*c+s*s)/L});
 const tot=P.reduce((a,v)=>a+v,0)+1e-12,bd=(a,z)=>FQ.reduce((s,f,k)=>f>=a&&f<z?s+P[k]:s,0)/tot;
 o.dom_freq=FQ[P.indexOf(Math.max(...P))];o.psd_0p5_5=bd(0,5);o.psd_5_15=bd(5,15);o.psd_15_40=bd(15,41);o.spec_entropy=-P.reduce((s,v)=>s+(v/tot)*Math.log(v/tot+1e-12),0);
 let a=Array.from(b);const E=[];for(let l=0;l<4;l++){const h=a.length>>1,na=[],dd=[];for(let i=0;i<h;i++){na.push((a[2*i]+a[2*i+1])/1.4142);dd.push((a[2*i]-a[2*i+1])/1.4142)}E.push(dd.reduce((s,v)=>s+v*v,0));a=na}
 const ea=a.reduce((s,v)=>s+v*v,0),T=ea+E.reduce((s,v)=>s+v,0)+1e-12;o.wav_a4=ea/T;o.wav_d4=E[3]/T;o.wav_d3=E[2]/T;o.wav_d2=E[1]/T;return o}
function process(sig,fs0){let x=resample(clean(sig),fs0);const m=med(x);x=x.map(v=>v-m);
 const base=removeBaseline(x,FS),filt=filtfilt(base,[bq('hp',.5,FS),bq('lp',40,FS)],FS),rp=detectR(filt,FS);
 const sc=med(rp.map(i=>Math.abs(filt[i])))||1,norm=filt.map(v=>v/sc),rr=[];for(let i=1;i<rp.length;i++)rr.push((rp[i]-rp[i-1])/FS);
 const prev=rp.map((_,i)=>i?rr[i-1]:(rr[0]||.8)),next=rp.map((_,i)=>i<rr.length?rr[i]:(rr[rr.length-1]||.8)),rows=[];
 rp.forEach((r,i)=>{if(r-PRE<0||r+POST>norm.length)return;const w=prev.slice(Math.max(i-4,0),i+6),mu=w.reduce((a,v)=>a+v,0)/w.length;
  const o=bfeat(norm.slice(r-PRE,r+POST));Object.assign(o,{rr_prev:prev[i],rr_next:next[i],rr_mean:mu,rr_ratio_prev:prev[i]/mu,rr_ratio_next:next[i]/mu});
  rows.push({r,f:FN.map(k=>o[k]),beat:norm.slice(r-PRE,r+POST)})});
 return{x,base,filt,norm,rp,rows,scale:sc}}
function detStats(rp,ann,tol=.075*FS){let tp=0;for(const a of ann)if(rp.some(p=>Math.abs(p-a.i)<=tol))tp++;return{tp,se:tp/Math.max(ann.length,1),ppv:tp/Math.max(rp.length,1),n:ann.length}}
// ---------- rough set (QuickReduct) ----------
function roughSel(X,y,{bins=4,tol=0,maxPer=250}={}){const r=rng(7),idx=[];
 for(let k=0;k<K;k++){const a=[];y.forEach((c,i)=>{if(c===k)a.push(i)});for(let i=a.length-1;i>0;i--){const j=Math.floor(r()*(i+1));[a[i],a[j]]=[a[j],a[i]]}idx.push(...a.slice(0,maxPer))}
 const n=idx.length,nf=FN.length,Y=idx.map(i=>y[i]),D=[],cuts=[];
 for(let f=0;f<nf;f++){const v=idx.map(i=>X[i][f]).sort((a,b)=>a-b),c=[];for(let b=1;b<bins;b++)c.push(v[Math.floor(b*n/bins)]);cuts.push(c);D.push(Uint8Array.from(idx,i=>c.filter(t=>X[i][f]>t).length))}
 const gamma=B=>{if(!B.length)return 0;const m=new Map(),ks=[];for(let i=0;i<n;i++){let k='';for(const f of B)k+=D[f][i];ks.push(k);const v=m.get(k);if(v===undefined)m.set(k,Y[i]);else if(v!==Y[i]&&v!==-1)m.set(k,-1)}
  let p=0;for(const k of ks)if(m.get(k)>=0)p++;return p/n};
 const all=[...Array(nf).keys()],gC=gamma(all),steps=[];let R=[],g=0;
 while(g<gC-tol-1e-12&&R.length<nf){let bf=-1,bg=-1;for(const f of all)if(!R.includes(f)){const v=gamma([...R,f]);if(v>bg){bg=v;bf=f}}R.push(bf);g=bg;steps.push({f:bf,g})}
 for(const f of [...R])if(R.length>1&&gamma(R.filter(z=>z!==f))>=gC-tol-1e-12)R=R.filter(z=>z!==f);
 return{reduct:R,gammaAll:gC,gammaR:gamma(R),core:all.filter(f=>gamma(all.filter(z=>z!==f))<gC-1e-12),steps,single:all.map(f=>gamma([f])),n,bins,tol}}
// ---------- fuzzy rule classifier ----------
function mu(c,t,v){const[a,b,d]=c;if(t===0)return v<=a?1:v>=b?0:(b-v)/(b-a||1);if(t===2)return v>=d?1:v<=b?0:(v-b)/(d-b||1);return v<=a||v>=d?0:v<b?(v-a)/(b-a||1):(d-v)/(d-b||1)}
function fitFuzzy(X,y,cols,{beam=30,rpc=12}={}){const n=X.length,nk=Array(K).fill(0);y.forEach(c=>nk[c]++);
 const cen=cols.map(c=>{const v=X.map(r=>r[c]).sort((a,b)=>a-b),q=p=>v[Math.min(n-1,Math.floor(p*n))];return[q(.15),q(.5),q(.85)]});
 const terms=[];cols.forEach((c,ci)=>{for(let t=0;t<3;t++)terms.push({ci,t,m:Float32Array.from(X,r=>mu(cen[ci],t,r[c]))})});
 const fv=ts=>{const f=new Float32Array(n).fill(1);for(const j of ts)for(let i=0;i<n;i++)f[i]*=terms[j].m[i];return f};
 const mk=(ts,f)=>{const s=Array(K).fill(0);for(let i=0;i<n;i++)s[y[i]]+=f[i];for(let k=0;k<K;k++)s[k]=nk[k]?s[k]/nk[k]:0;const tot=s.reduce((a,b)=>a+b,0)||1;return{ts,s,conf:s.map(v=>v/tot)}};
 let lvl=terms.map((_,i)=>mk([i],fv([i]))),pool=[...lvl];
 for(let d=2;d<=3;d++){const top=new Set(),seen=new Set(),nx=[];for(let k=0;k<K;k++)[...lvl].sort((a,b)=>b.s[k]*b.conf[k]-a.s[k]*a.conf[k]).slice(0,beam).forEach(r=>top.add(r));
  for(const r of top){const pf=fv(r.ts);for(let j=0;j<terms.length;j++){if(r.ts.some(i=>terms[i].ci===terms[j].ci))continue;const ts=[...r.ts,j].sort((a,b)=>a-b),key=ts.join();if(seen.has(key))continue;seen.add(key);
   const f=new Float32Array(n);for(let i=0;i<n;i++)f[i]=pf[i]*terms[j].m[i];nx.push(mk(ts,f))}}
  pool.push(...nx);lvl=nx}
 const rules=[];
 for(let k=0;k<K;k++){const cand=pool.filter(r=>r.s[k]>=.03).sort((a,b)=>b.s[k]*b.conf[k]-a.s[k]*a.conf[k]).slice(0,300).map(r=>({r,f:fv(r.ts)})),cov=new Float32Array(n),used=new Set();
  for(let q=0;q<rpc&&cand.length;q++){let bi=-1,bs=0;cand.forEach((c,ci)=>{if(used.has(ci))return;let s=0;for(let i=0;i<n;i++)if(y[i]===k)s+=c.f[i]/(1+cov[i]);s=s/nk[k]*c.r.conf[k];if(s>bs){bs=s;bi=ci}});
   if(bi<0)break;used.add(bi);const c=cand[bi];for(let i=0;i<n;i++)cov[i]+=c.f[i];rules.push({ts:c.r.ts.map(j=>[terms[j].ci,terms[j].t]),cls:k,w:c.r.conf[k]})}}
 return{cols,cen,rules}}
function predict(m,row){const pn=Array(K).fill(1),fired=[];
 for(const r of m.rules){let f=1;for(const[ci,t]of r.ts){f*=mu(m.cen[ci],t,row[m.cols[ci]]);if(!f)break}if(f>1e-9){pn[r.cls]*=1-r.w*f;fired.push({r,f})}}
 const sc=pn.map(p=>1-p),o=sc.map((v,i)=>[v,i]).sort((a,b)=>b[0]-a[0]),none=o[0][0]<1e-9;
 return{k:none?0:o[0][1],sc,margin:o[0][0]-o[1][0],none,fired:fired.sort((a,b)=>b.r.w*b.f-a.r.w*a.f)}}
function evaluate(m,X,y){const cm=Array.from({length:K},()=>Array(K).fill(0));X.forEach((r,i)=>cm[y[i]][predict(m,r).k]++);
 const pc=CLS.map((_,k)=>{const tp=cm[k][k],fn=cm[k].reduce((a,b)=>a+b,0)-tp,fp=cm.reduce((a,r)=>a+r[k],0)-tp,p=tp/Math.max(tp+fp,1),rc=tp/Math.max(tp+fn,1);return{n:tp+fn,p,r:rc,f1:p+rc?2*p*rc/(p+rc):0}});
 const sup=pc.filter(c=>c.n>0);return{cm,pc,acc:cm.reduce((a,r,k)=>a+r[k],0)/X.length,macroF1:sup.reduce((a,c)=>a+c.f1,0)/sup.length,n:X.length}}
// ---------- dataset + training ----------
function dataset(seed0,nRec,dur){const X=[],y=[];for(let k=0;k<nRec;k++){const r=rng(seed0+k*7919),mix={N:.3+.3*r(),S:.1+.15*r(),V:.1+.2*r(),F:.05+.1*r(),Q:.05+.1*r()},rec=synth(seed0+k,dur,mix,.03+.04*r()),p=process(rec.x,FS);
 for(const row of p.rows){let b=-1,bd=1e9;for(const a of rec.ann){const d=Math.abs(a.i-row.r);if(d<bd){bd=d;b=a}}if(bd<=.1*FS){X.push(row.f);y.push(CLS.indexOf(b.c))}}}return{X,y}}
function train({tol=0,bins=4}={}){const tr=dataset(100,40,24),te=dataset(9000,14,24),rough=roughSel(tr.X,tr.y,{bins,tol}),
 mB=fitFuzzy(tr.X,tr.y,rough.reduct),mA=fitFuzzy(tr.X,tr.y,FN.map((_,i)=>i));
 return{rough,mA,mB,evA:evaluate(mA,te.X,te.y),evB:evaluate(mB,te.X,te.y),nTrain:tr.X.length,nTest:te.X.length}}
G.ECG={FS,PRE,POST,CLS,CN,FN,DOM,TN,SCEN,synth,process,detStats,roughSel,fitFuzzy,predict,evaluate,train,mu};
if(typeof module!=='undefined')module.exports=G.ECG;
})(typeof window!=='undefined'?window:globalThis);
