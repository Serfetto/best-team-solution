// ===== helpers.js =====
// DOM helpers
window.$  = (sel) => document.querySelector(sel);
window.$$ = (sel) => Array.from(document.querySelectorAll(sel));

// fetch helpers
window.check_status = function(r){
  if (!r.ok) throw new Error(`Ошибка загрузки: ${r.status} ${r.statusText} ${r.message}`);
  return r.json();
};
window.check_status_for_export = function(r){
  if (!r.ok) throw new Error(`Ошибка загрузки: ${r.status} ${r.statusText}`);
  return r;
};

// utils
window.parseCSV = function(text){
  const lines = text.split(/\r?\n/).filter(Boolean);
  if (!lines.length) return { data: [] };
  const header = lines.shift().split(",").map(s=>s.trim().toLowerCase());
  const idIdx = header.indexOf("id");
  const textIdx = header.indexOf("text");
  const data = lines.map(row=>{
    const parts = row.split(",");
    return { id: parts[idIdx], text: parts[textIdx] };
  });
  return { data };
};

window.downloadToFile = async function(resp, filename, statusEl){
  if(!resp || !resp.ok){ statusEl.textContent = "Ошибка скачивания."; return; }
  const blob = await resp.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url; a.download = filename || "dataset.json";
  document.body.appendChild(a); a.click();
  a.remove(); URL.revokeObjectURL(url);
  statusEl.textContent = "Файл скачан.";
};

window.redirectOverallAnalysis = function (e) {
  e && e.preventDefault && e.preventDefault();

  const portMeta = document.querySelector('meta[name="analysis-port"]');
  const hostMeta = document.querySelector('meta[name="analysis-host"]');
  const pathMeta = document.querySelector('meta[name="analysis-path"]');

  const port = (portMeta?.content || '8000').replace(':','').trim();
  const host = (hostMeta?.content || window.location.hostname).trim();
  let path = (pathMeta?.content || '/').trim();
  if (!path.startsWith('/')) path = '/' + path;

  const params = new URLSearchParams();
  if (dateFrom) params.set("start_date", dateFrom);
  if (dateTo) params.set("end_date", dateTo);

  // Жёстко ставим http, как вы и просили
  const target = `http://${host}:${port}${path}?${params.toString()}`;
  window.location.href = target;

  return false;
};