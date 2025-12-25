function linspace(a, b, n){
  const arr = new Array(n);
  const step = (b - a) / (n - 1);
  for(let i=0;i<n;i++) arr[i] = a + step * i;
  return arr;
}
function fmt(n, d=3){ return Number(n).toFixed(d); }

function matToText(A, name){
  let s = `${name} =\n`;
  for(const r of A){
    s += "  [" + r.map(v => v.toFixed(3).padStart(8)).join("  ") + "]\n";
  }
  return s;
}
function nearestIndex(arr, x){
  let bestI = 0, bestD = Infinity;
  for(let i=0;i<arr.length;i++){
    const d = Math.abs(arr[i] - x);
    if(d < bestD){ bestD = d; bestI = i; }
  }
  return bestI;
}

// 4x4 Gaussian elimination (no libs)
function solveLinear4(A, b){
  const M = A.map(r => r.slice());
  const y = b.slice();
  const n = 4;

  for(let k=0;k<n;k++){
    let piv = k;
    for(let i=k+1;i<n;i++){
      if(Math.abs(M[i][k]) > Math.abs(M[piv][k])) piv = i;
    }
    if(piv !== k){
      [M[k], M[piv]] = [M[piv], M[k]];
      [y[k], y[piv]] = [y[piv], y[k]];
    }
    const diag = M[k][k];
    if(Math.abs(diag) < 1e-12) throw new Error("Singular system");

    for(let j=k;j<n;j++) M[k][j] /= diag;
    y[k] /= diag;

    for(let i=0;i<n;i++){
      if(i === k) continue;
      const f = M[i][k];
      for(let j=k;j<n;j++) M[i][j] -= f * M[k][j];
      y[i] -= f * y[k];
    }
  }
  return y;
}

function makeField(top, bottom, sigma, nx, ny, nodes){
  const xs = linspace(0, 4, nx);
  const ys = linspace(0, 2, ny);

  const baseAt = (y) => bottom - (bottom - top) * (y / 2.0);
  const gauss = (dx, dy) => Math.exp(-(dx*dx + dy*dy) / (2*sigma*sigma));

  // Build K and rhs to match node temperatures
  const K = Array.from({length:4}, ()=> Array(4).fill(0));
  const rhs = new Array(4).fill(0);

  for(let i=0;i<4;i++){
    rhs[i] = nodes[i].T - baseAt(nodes[i].y);
    for(let j=0;j<4;j++){
      K[i][j] = gauss(nodes[i].x - nodes[j].x, nodes[i].y - nodes[j].y);
    }
    K[i][i] += 1e-9;
  }

  const coeff = solveLinear4(K, rhs);

  const Z = Array.from({length:ny}, ()=> new Array(nx).fill(0));
  for(let iy=0; iy<ny; iy++){
    const y = ys[iy];
    const base = baseAt(y);
    for(let ix=0; ix<nx; ix++){
      const x = xs[ix];
      let val = base;
      for(let k=0;k<4;k++){
        val += coeff[k] * gauss(x - nodes[k].x, y - nodes[k].y);
      }
      Z[iy][ix] = val;
    }
  }
  return {xs, ys, Z};
}

let lastPayload = null;

function renderPlots(payload){
  const heat = {
    type: 'heatmap',
    x: payload.x,
    y: payload.y,
    z: payload.z,
    colorbar: { title: '°C' },
    hovertemplate: 'x=%{x:.2f} cm<br>y=%{y:.2f} cm<br>T=%{z:.2f} °C<extra></extra>'
  };

  const contour = {
    type: 'contour',
    x: payload.x,
    y: payload.y,
    z: payload.z,
    contours: { coloring: 'none', showlabels: true },
    line: { width: 1 },
    hoverinfo: 'skip'
  };

  const nodes = {
    type: 'scatter',
    mode: 'markers+text',
    x: payload.nodes.x,
    y: payload.nodes.y,
    text: payload.nodes.labels,
    textposition: 'top center',
    marker: { size: 10 },
    hovertemplate: '%{text}<extra></extra>'
  };

  const layout = {
    title: { text: 'میدان دما — تعاملی', font: { size: 18 } },
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    margin: { l: 55, r: 18, t: 55, b: 45 },
    xaxis: { title: 'x (cm)', gridcolor: 'rgba(255,255,255,0.08)' },
    yaxis: { title: 'y (cm)', gridcolor: 'rgba(255,255,255,0.08)' },
    annotations: [
      { x: 2, y: 2.03, text: `Top=${payload.meta.top.toFixed(0)}°C`, showarrow: false },
      { x: 2, y: -0.08, text: `Bottom=${payload.meta.bottom.toFixed(0)}°C`, showarrow: false },
    ]
  };

  const config = {
    responsive: true,
    displaylogo: false,
    toImageButtonOptions: { filename: 'P5-52_Mehranjafari', format: 'png' }
  };

  Plotly.newPlot('plot', [heat, contour, nodes], layout, config);

  // Profile plot
  const xSlider = document.getElementById('xSlider');
  const xVal = document.getElementById('xVal');

  function drawProfile(xcut){
    xVal.textContent = xcut.toFixed(2);
    const xi = nearestIndex(payload.x, xcut);
    const profY = payload.y;
    const profT = payload.z.map(row => row[xi]);

    const trace = {
      type:'scatter',
      mode:'lines+markers',
      x: profT,
      y: profY,
      hovertemplate: 'y=%{y:.2f} cm<br>T=%{x:.2f} °C<extra></extra>'
    };

    const lay = {
      title: { text: `پروفایل دما در x≈${payload.x[xi].toFixed(2)} cm`, font:{size:16} },
      paper_bgcolor:'rgba(0,0,0,0)',
      plot_bgcolor:'rgba(0,0,0,0)',
      margin:{l:55,r:18,t:55,b:45},
      xaxis:{ title:'T (°C)', gridcolor:'rgba(255,255,255,0.08)' },
      yaxis:{ title:'y (cm)', gridcolor:'rgba(255,255,255,0.08)' }
    };

    Plotly.newPlot('profile', [trace], lay, {responsive:true, displaylogo:false});
  }

  drawProfile(parseFloat(xSlider.value));
  xSlider.oninput = (e)=> drawProfile(parseFloat(e.target.value));
}

function simulate(){
  const top = parseFloat(document.getElementById('top').value);
  const bottom = parseFloat(document.getElementById('bottom').value);
  const vnode = parseFloat(document.getElementById('vnode').value);
  const sigma = Math.max(0.35, Math.min(1.20, parseFloat(document.getElementById('sigma').value)));
  const nx = Math.max(80, Math.min(320, parseInt(document.getElementById('nx').value, 10)));
  const ny = Math.max(60, Math.min(260, parseInt(document.getElementById('ny').value, 10)));

  const A = [
    [ 4.0, -2.0,  0.0,  0.0],
    [-1.0,  4.0,  0.0,  0.0],
    [ 0.0, -1.0,  1.0,  0.0],
    [ 1.0,  0.0,  0.0, -1.0],
  ];
  const b = [
    top + bottom,
    top + vnode + bottom,
    0.0,
    0.0
  ];
  const x = solveLinear4(A, b);
  const [T1, T2, T3, T4] = x;

  document.getElementById('T1').textContent = `${fmt(T1)} °C`;
  document.getElementById('T2').textContent = `${fmt(T2)} °C`;
  document.getElementById('T3').textContent = `${fmt(T3)} °C`;
  document.getElementById('T4').textContent = `${fmt(T4)} °C`;

  document.getElementById('detailsText').textContent =
`معادلات (FDM + شرط عایق + تقارن):
1) 4T1 - 2T2 = Top + Bottom
2) -T1 + 4T2 = Top + Vnode + Bottom
3) -T2 + T3  = 0   (T3 = T2)
4)  T1 - T4  = 0   (T4 = T1)

Top=${top.toFixed(2)}°C , Bottom=${bottom.toFixed(2)}°C , Vnode=${vnode.toFixed(2)}°C , sigma=${sigma.toFixed(2)}
نتیجه: T1=${T1.toFixed(3)}, T2=${T2.toFixed(3)}, T3=${T3.toFixed(3)}, T4=${T4.toFixed(3)} (°C)`;

  document.getElementById('matrixText').textContent =
    matToText(A, "A") + "\n" +
    `b = [${b.map(v => v.toFixed(3).padStart(8)).join("  ")}]\n` +
    `x = [T1 T2 T3 T4]^T = [${x.map(v => v.toFixed(3).padStart(8)).join("  ")}]\n`;

  const nodes = [
    {x:1.0, y:1.0, T:T1},
    {x:2.0, y:1.0, T:T2},
    {x:3.0, y:1.0, T:T3},
    {x:4.0, y:1.0, T:T4},
  ];

  const {xs, ys, Z} = makeField(top, bottom, sigma, nx, ny, nodes);

  const plotPayload = {
    x: xs,
    y: ys,
    z: Z,
    nodes: {
      x: nodes.map(p=>p.x),
      y: nodes.map(p=>p.y),
      labels: [
        `T1=${T1.toFixed(1)}°C`,
        `T2=${T2.toFixed(1)}°C`,
        `T3=${T3.toFixed(1)}°C`,
        `T4=${T4.toFixed(1)}°C`,
      ]
    },
    meta: {top, bottom}
  };

  lastPayload = {
    title: "Heat Conduction Simulation — Problem 5-52",
    author: "مهران جعفری",
    professor: "دکتر محمدجواد نایری",
    inputs: {top, bottom, vnode, sigma, nx, ny},
    A, b, x,
    T: {T1, T2, T3, T4},
    plotPayload
  };

  renderPlots(plotPayload);
}

function downloadJSON(){
  if(!lastPayload) return;
  const blob = new Blob([JSON.stringify(lastPayload, null, 2)], {type:"application/json"});
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "P5-52_Mehranjafari.json";
  a.click();
  URL.revokeObjectURL(url);
}

async function downloadPNG(){
  const gd = document.getElementById("plot");
  const dataUrl = await Plotly.toImage(gd, {format:"png", height:700, width:1200});
  const a = document.createElement("a");
  a.href = dataUrl;
  a.download = "P5-52_Mehranjafari.png";
  a.click();
}

document.getElementById("form").addEventListener("submit", (e)=>{
  e.preventDefault();
  simulate();
});
document.getElementById("btnJson").addEventListener("click", (e)=>{
  e.preventDefault();
  downloadJSON();
});
document.getElementById("btnPng").addEventListener("click", (e)=>{
  e.preventDefault();
  downloadPNG();
});

// initial render
simulate();
