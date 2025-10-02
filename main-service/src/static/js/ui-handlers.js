// ===== ui-handlers.js =====

function doSomethingCool(event) {
  event.preventDefault(); // Предотвращаем перезагрузку страницы
  alert('Привет от JavaScript!');
}

// Глобальная функция для остановки всех таймеров
window.stopAllTimers = function() {
  // Останавливаем таймер парсинга
  if (window.parsingTimer) {
    clearInterval(window.parsingTimer);
    window.parsingTimer = null;
    console.log('All timers stopped');
  }
};

// Останавливаем таймеры при уходе со страницы
window.addEventListener('beforeunload', window.stopAllTimers);

//===============parsing=================
(function () {
  const elFreq   = $("#parsing-frequency");
  const elMode   = $("#parsing-mode");
  const elSource = $("#parsing-source");
  const btnStart = $("#btn-parsing-start");
  const btnEdit  = $("#btn-parsing-edit");
  const statusEl = $("#parsing-status");

  const POLL_MS = 5000;
  let timer = null;
  let isEditing = false; // флаг: редактируем или стартуем
  let originalValues = {}; // сохраняем исходные значения полей

  function syncStartBtn(){
    if (!btnStart) return;
    if (isEditing) {
      btnStart.textContent = "Сохранить изменения";
      return;
    }
    const hasFreq = !!(elFreq?.value);
    btnStart.textContent = hasFreq ? 'Начать' : 'Немедленно начать';
  }

  // Функция для проверки изменений
  function hasChanges() {
    return elFreq?.value !== originalValues.freq ||
           elMode?.value !== originalValues.mode ||
           elSource?.value !== originalValues.source;
  }

  // Функция для сохранения текущих значений
  function saveOriginalValues() {
    originalValues = {
      freq: elFreq?.value,
      mode: elMode?.value,
      source: elSource?.value
    };
  }

  elFreq?.addEventListener('change', syncStartBtn);
  syncStartBtn();

  // Collapsible description in Parsing
  try {
    const block = document.getElementById('parsing-description');
    const toggle = document.getElementById('parsing-desc-toggle');
    const content = document.getElementById('parsing-desc-content');
    if (block && toggle && content) {
      // initial state: open
      block.classList.add('open');
      block.classList.remove('closed');
      toggle.setAttribute('aria-expanded', 'true');
      toggle.addEventListener('click', () => {
        const expanded = toggle.getAttribute('aria-expanded') === 'true';
        const next = !expanded;
        toggle.setAttribute('aria-expanded', String(next));
        if (next) {
          block.classList.add('open');
          block.classList.remove('closed');
          toggle.querySelector('.collapsible__icon')?.replaceChildren(document.createTextNode('▲'));
        } else {
          block.classList.remove('open');
          block.classList.add('closed');
          toggle.querySelector('.collapsible__icon')?.replaceChildren(document.createTextNode('▼'));
        }
      });
    }
  } catch(_){}

  function renderParsingStatus(state) {
    const el = document.getElementById("parsing-status");
    if (!el) return;
    if (!state || !state.running) {
      el.innerHTML = "<p>Парсинг не запущен.</p>" + 
        (state?.last_parsing_time ? 
          `<p><b>Последний запуск:</b> ${state.last_parsing_time}</p>` : 
          "");
      return;
    }
    el.innerHTML = `
      <p><b>Статус:</b> ${state.running ? "Запущен" : "Остановлен"}</p>
      <p><b>ID задачи:</b> ${state.job_id || "-"}</p>
      <p><b>Источник:</b> ${state.source || "-"}</p>
      <p><b>Частота:</b> ${state.every || "-"}</p>
      <p><b>Начато:</b> ${state.started_at || "-"}</p>
      <p><b>Следующий запуск:</b> ${state.next_started_at || "-"}</p>
      ${state.last_parsing_time ? `<p><b>Последний запуск:</b> ${state.last_parsing_time}</p>` : ''}
    `;
    btnStart.disabled = state.running && !isEditing;
  }

  async function start(){
    // Если мы в режиме редактирования, но изменений нет - выходим из режима редактирования
    if (isEditing && !hasChanges()) {
      isEditing = false;
      statusEl.dataset.state = "info";
      statusEl.textContent = "Изменений не обнаружено. Режим редактирования отключен.";
      syncStartBtn();
      return;
    }

    const every  = (elFreq?.value || '').trim() || null;
    
    const mode   = elMode?.value || 'incremental';
    const source = elSource?.value || 'full';

    const payload = { mode, source };
    if (every) payload.every = every;

    statusEl.dataset.state = 'info';
    statusEl.textContent = isEditing
      ? 'Отправляем изменения задачи...'
      : 'Отправляем запрос на запуск парсинга...';

    try {
      let res;
      if (isEditing) {
        res = await API.parsingEdit(payload);
        isEditing = false; // после сохранения изменений возвращаемся в обычный режим
        saveOriginalValues(); // сохраняем новые значения как исходные
      } else {
        res = await API.parsingStart(payload);
        saveOriginalValues(); // сохраняем значения при старте
      }

      ["parsing-frequency", "parsing-mode", "parsing-source"].forEach((id) => {
        document.getElementById(id).disabled = res.running;
      });

      renderParsingStatus(res?.state || res);
      syncStartBtn();
    } catch (e) {
      statusEl.dataset.state = 'error';
      statusEl.textContent = e?.message || 'Ошибка при запуске/изменении парсинга.';
    }
  }

  function edit() {
    // Сохраняем текущие значения как исходные
    saveOriginalValues();
    isEditing = true;
    
    ["parsing-frequency", "parsing-mode", "parsing-source"].forEach((id) => {
      document.getElementById(id).disabled = false;
    });
    
    statusEl.dataset.state = "info";
    statusEl.textContent = "Измените параметры и нажмите «Сохранить изменения». Кнопка станет активной только при изменении полей.";
    
    // Блокируем кнопку сохранения до тех пор, пока не будет изменений
    btnStart.disabled = true;
    
    // Добавляем обработчики изменений для полей
    const fields = [elFreq, elMode, elSource];
    fields.forEach(field => {
      if (field) {
        field.addEventListener('input', handleFieldChange);
        field.addEventListener('change', handleFieldChange);
      }
    });
    
    syncStartBtn();
  }

  function handleFieldChange() {
    // Активируем кнопку сохранения только если есть изменения
    btnStart.disabled = !hasChanges();
    
    if (hasChanges()) {
      statusEl.dataset.state = "info";
      statusEl.textContent = "Обнаружены изменения. Нажмите «Сохранить изменения» для применения.";
    } else {
      statusEl.dataset.state = "info";
      statusEl.textContent = "Измените параметры и нажмите «Сохранить изменения».";
    }
  }

  async function refresh() {
    try {
      const s = await API.parsingStatus();
      
      // Если не в режиме редактирования, обновляем интерфейс как обычно
      if (!isEditing) {
        if (s.running) {
          ["parsing-frequency", "parsing-mode", "parsing-source"].forEach((id) => {
            document.getElementById(id).disabled = true;
          });
        } else {
          ["parsing-frequency", "parsing-mode", "parsing-source"].forEach((id) => {
            document.getElementById(id).disabled = false;
          });
        }
        btnStart.disabled = s.running;
        if (s.running && s.every === null){
          btnEdit.disabled = s.running;
        }
        else{
          btnEdit.disabled = false
        }
          
        
        renderParsingStatus(s);
        
        // Сохраняем текущие значения после обновления статуса
        saveOriginalValues();
      } else {
        // В режиме редактирования показываем специальный статус
        statusEl.dataset.state = "info";
        statusEl.textContent = "Вы редактируете задачу. Текущий статус задачи: " +
          (s.running ? "Запущен" : "Остановлен") + 
          (hasChanges() ? ". Обнаружены изменения." : ". Изменений нет.");
      }
    } catch (e) {
      if (!isEditing) { // не сбивать интерфейс редактирования
        statusEl.dataset.state = "error";
        statusEl.textContent = "Не удалось получить статус парсинга.";
      }
    }
  }

  btnStart?.addEventListener("click", start);
  btnEdit?.addEventListener("click", edit);

  // Expose polling controls to other modules (e.g., sidebar/nav)
  window.startParsingPolling = function(){
    if (timer) clearInterval(timer);
    refresh();
    timer = setInterval(refresh, POLL_MS);
    window.parsingTimer = timer;
  };
  window.stopParsingPolling = function(){
    if (timer) clearInterval(timer);
    timer = null;
    window.parsingTimer = null;
  };

  document
    .querySelectorAll('.vnav a[data-target="parsing"], .tab[data-target="parsing"]')
    .forEach((node) => {
      node.addEventListener("click", () => {
        window.startParsingPolling?.();
      });
    });

  saveOriginalValues();

  if (location.hash === "#parsing") {
    window.startParsingPolling?.();
  }
})();


//===============sidebar=================
(() => {
  const sidebar  = document.getElementById('left-sidebar');
  const tabs     = document.querySelectorAll('.tab');
  const sections = document.querySelectorAll('.section');

  function showSection(id) {
    sections.forEach(s => s.classList.toggle('visible', s.id === id));
    tabs.forEach(b => b.classList.toggle('active', b.dataset.target === id));

    history.replaceState(null, '', `#${id}`);

    sidebar?.classList.remove('open');
    document.body.classList.remove('sidebar-open');

    // Start/stop polling based on active section
    if (id === 'parsing') {
      window.startParsingPolling?.();
    } else {
      window.stopParsingPolling?.();
    }
  }

  // клики по табам
  tabs.forEach(btn => {
    btn.addEventListener('click', (e) => {
      const id = btn.dataset.target;

      showSection(id);
    });
  });


  document.querySelectorAll('.vnav a[data-target]').forEach(a => {
    a.addEventListener('click', (e) => {
      const id = a.dataset.target;
      e.preventDefault();
      if (id && document.getElementById(id)) {
        showSection(id);
        if (id === 'taxonomy') {
          window.loadServerTaxonomy?.();
        }
      }
    });
  });

  // открыть секцию при загрузке (по хэшу), иначе остаёмся на текущей
  const initial = location.hash ? location.hash.slice(1) : null;
  if (initial && document.getElementById(initial)) {
    showSection(initial);
    if (initial === 'taxonomy') {
      window.loadServerTaxonomy?.();
    }
  }

  // Handle browser navigation (back/forward) updating hash
  window.addEventListener('hashchange', () => {
    const id = location.hash ? location.hash.slice(1) : '';
    if (id && document.getElementById(id)) {
      showSection(id);
    }
  });

  
})();

(function(){
  const sidebar   = document.getElementById('left-sidebar');
  const toggleBtn = document.getElementById('sidebar-toggle');

  function syncState(){
    const opened = sidebar.classList.contains('open');
    document.body.classList.toggle('sidebar-open', opened);
    toggleBtn.setAttribute('aria-expanded', opened ? 'true' : 'false');
    const label = opened ? 'Закрыть боковую панель' : 'Открыть боковую панель';
    toggleBtn.setAttribute('aria-label', label);
    toggleBtn.dataset.tooltip = label;
    // при открытии уберём подсказку, чтобы не заслоняла пункты
    if (opened) toggleBtn.blur();
  }

  toggleBtn.addEventListener('click', () => {
    sidebar.classList.toggle('open');
    syncState();
  });

  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && sidebar.classList.contains('open')) {
      sidebar.classList.remove('open');
      syncState();
    }
  });

  syncState();
})();

// tabs
$$(".tab").forEach((btn) => {
  btn.addEventListener("click", () => {
    $$(".tab").forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");
    $$(".section").forEach((section) => section.classList.remove("visible"));
    $(`#${btn.dataset.target}`).classList.add("visible");
    if (btn.dataset.target === "taxonomy") {
      window.loadServerTaxonomy?.();
    }
  });
});

$("#btn-dl-enriched-xlsx")?.addEventListener("click", async () => {
  $("#dl-enriched-xlsx-status").textContent = "Подготовка загрузки:";
  try {
    const resp = await API.exportEnrichedCsv();
    await downloadToFile(resp, "enriched_combined.csv", $("#dl-enriched-xlsx-status"));
  } catch (e) {
    $("#dl-enriched-xlsx-status").textContent = "Ошибка загрузки";
  }
});


// taxonomy interactions
const taxonomyInput = $("#taxonomy-input");
const addTaxonomyBtn = $("#btn-add-taxonomy");
const saveTaxonomyBtn = $("#btn-save-taxonomy");
const refreshTaxonomyBtn = $("#btn-refresh-taxonomy");
const serverTaxonomyList = $("#taxonomy-server-list");
const derivedTaxonomyList = $("#taxonomy-derived-list");

const arraysEqual = (a, b) => {
  if (a === b) return true;
  if (!Array.isArray(a) || !Array.isArray(b)) return false;
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i += 1) {
    if (a[i] !== b[i]) return false;
  }
  return true;
};

addTaxonomyBtn?.addEventListener("click", () => {
  const value = taxonomyInput?.value || "";
  window.addTaxonomyItem?.(value);
  if (taxonomyInput) {
    taxonomyInput.value = "";
    taxonomyInput.focus();
  }
});

taxonomyInput?.addEventListener("keydown", (event) => {
  if (event.key === "Enter") {
    event.preventDefault();
    addTaxonomyBtn?.click();
  }
});

refreshTaxonomyBtn?.addEventListener("click", () => {
  window.loadServerTaxonomy?.(true);
});

saveTaxonomyBtn?.addEventListener("click", async () => {
  const currentBase = Array.isArray(window.serverTaxonomy) ? window.serverTaxonomy : [];
  const currentDerived = Array.isArray(window.derivedTaxonomy) ? window.derivedTaxonomy : [];
  const initialBase = Array.isArray(window.taxonomyInitialState?.server)
    ? window.taxonomyInitialState.server
    : [];
  const initialDerived = Array.isArray(window.taxonomyInitialState?.derived)
    ? window.taxonomyInitialState.derived
    : [];

  const hasBaseChanges = !arraysEqual(currentBase, initialBase);
  const hasDerivedChanges = !arraysEqual(currentDerived, initialDerived);

  if (!hasBaseChanges && !hasDerivedChanges) {
    window.setTaxonomyStatus?.("No changes to save", "info");
    return;
  }

  window.setTaxonomyStatus?.("Saving taxonomy...", "info");

  try {
    await API.taxonomyEdit({ show: currentBase, hide: currentDerived });
    window.setTaxonomyStatus?.("Changes saved", "success");
    await window.loadServerTaxonomy?.(true);
  } catch (error) {
    console.error(error);
    window.setTaxonomyStatus?.("Failed to save taxonomy", "error");
  }
});
serverTaxonomyList?.addEventListener("click", (event) => {
  const target = event.target.closest("button[data-action]");
  if (!target) return;
  const index = Number(target.dataset.index ?? "-1");
  if (Number.isNaN(index) || index < 0) return;
  const action = target.dataset.action;
  if (action === "edit") {
    window.editTaxonomyItem?.(index);
  } else if (action === "remove") {
    window.removeTaxonomyItem?.(index);
  }
});

derivedTaxonomyList?.addEventListener("click", (event) => {
  const target = event.target.closest("button[data-action]");
  if (!target) return;
  const index = Number(target.dataset.index ?? "-1");
  if (Number.isNaN(index) || index < 0) return;
  const action = target.dataset.action;
  if (action === "promote") {
    window.promoteDerivedTaxonomy?.(index);
  } else if (action === "remove") {
    window.removeDerivedTaxonomyItem?.(index);
  }
});

// export
$("#btn-dl-combined")?.addEventListener("click", async () => {
  $("#dl-combined-status").textContent = "Готовим выгрузку…";
  try {
    const resp = await API.exportCombined();
    await downloadToFile(resp, "combined.json", $("#dl-combined-status"));
  } catch (e) {
    $("#dl-combined-status").textContent = "Ошибка выгрузки";
  }
});

$("#btn-dl-source")?.addEventListener("click", async () => {
  const src = $("#export-source").value;
  $("#dl-source-status").textContent = "Готовим выгрузку…";
  try {
    const resp = await API.exportByName(src);
    await downloadToFile(resp, `${src}.json`, $("#dl-source-status"));
  } catch (e) {
    $("#dl-source-status").textContent = "Ошибка выгрузки";
  }
});


// ======== Analize / predict================

$("#btn-upload")?.addEventListener("click", async () => {
  const file = $("#file-input").files[0];
  const status = $("#upload-status");
  const result_block = $("#result_block");
  if (!file) {
    status.textContent = "Выберите файл .csv или .json";
    return;
  }

  status.textContent = "Читаю файл…";
  const text = await file.text();
  let payload;
  try {
    if (file.name.endsWith(".csv")) payload = parseCSV(text);
    else payload = JSON.parse(text);
  } catch (e) {
    status.textContent = "Ошибка: не удалось обработать файл";
    return;
  }

  status.textContent = "Отправляю на /analysis/predict…";
  try {
    const res = await API.predict(payload);
    const preds = res?.predictions || [];
    result_block.style.display = "block";
    window.lastPredictions = preds;
    window.lastAnalysisMeta = res;
    fillPredictionsTable(preds);
    renderAnalysisArea(preds, res);
    status.textContent = `Готово. Получено ${preds.length} предсказаний.`;
  } catch (e) {
    status.textContent = "Ошибка при отправке";
  }
});

function redirectOverallAnalysis(event) {
  event.preventDefault();
  const dateFrom = document.getElementById("products-date-from")?.min;
  const dateTo = document.getElementById("products-date-to")?.max;

  // формируем URL с параметрами
  const params = new URLSearchParams();
  if (dateFrom) params.set("start_date", dateFrom);
  if (dateTo) params.set("end_date", dateTo);

  // переходим на дашборд (порт 8000)
  window.location.href = `http://dashboard:8000/?${params.toString()}`;
  return false;
}

let currentPage = 1;
const cardsPerPage = 10;
let predictions = [];

function renderCards(data) {
  const container = document.getElementById("cards_container");
  container.innerHTML = "";

  const start = (currentPage - 1) * cardsPerPage;
  const end = start + cardsPerPage;
  const pageData = data.slice(start, end);

  pageData.forEach(item => {
    const card = document.createElement("div");
    card.classList.add("card");

    const topicsHtml = item.topics.map(t => `<span class="topic">${t}</span>`).join("");
    const sentimentsHtml = item.sentiments.map(s => {
      const cls = s.toLowerCase().includes("положительно") ? "positive" : "negative";
      return `<span class="sentiment ${cls}">${s}</span>`;
    }).join("");

    card.innerHTML = `
      <h4>ID: ${item.id}</h4>
      <div class="topics">${topicsHtml}</div>
      <div class="sentiments">${sentimentsHtml}</div>
    `;
    container.appendChild(card);
  });

  renderPagination(data.length);
}

function renderPagination(totalItems) {
  const pages = Math.ceil(totalItems / cardsPerPage);
  const pagination = document.getElementById("pagination");
  pagination.innerHTML = "";

  const prevBtn = document.createElement("button");
  prevBtn.textContent = "« Предыдущая";
  prevBtn.disabled = currentPage === 1;
  prevBtn.classList.toggle("disabled", currentPage === 1);
  prevBtn.onclick = () => { currentPage--; renderCards(predictions); };
  pagination.appendChild(prevBtn);

  const nextBtn = document.createElement("button");
  nextBtn.textContent = "Следующая »";
  nextBtn.disabled = currentPage === pages;
  nextBtn.classList.toggle("disabled", currentPage === pages);
  nextBtn.onclick = () => { currentPage++; renderCards(predictions); };
  pagination.appendChild(nextBtn);
}

// Используется после получения ответа с сервера
function displayPredictions(response) {
  predictions = response.predictions || [];
  currentPage = 1;
  if(predictions.length === 0) {
    document.getElementById("cards_container").innerHTML = "<p>Нет данных для отображения.</p>";
    document.getElementById("pagination").innerHTML = "";
    return;
  }
  renderCards(predictions);
}


const dropContainer = document.getElementById("dropcontainer");
const fileInput = document.getElementById("json_file");
const resultBlock = document.getElementById("result_block");
const uploadStatus = document.getElementById("upload_status");

function setStatus(message, type = "info") {
  uploadStatus.textContent = message;
  uploadStatus.className = `status ${type}`;
}

// Drag & Drop стили
dropContainer.addEventListener("dragover", (e) => e.preventDefault());
dropContainer.addEventListener("dragenter", () => dropContainer.classList.add("drag-active"));
dropContainer.addEventListener("dragleave", () => dropContainer.classList.remove("drag-active"));
dropContainer.addEventListener("drop", (e) => {
  e.preventDefault();
  dropContainer.classList.remove("drag-active");
  fileInput.files = e.dataTransfer.files;
  handleFile(fileInput.files[0]);
});

// Обработка выбора через input
fileInput.addEventListener("change", () => {
  if (fileInput.files.length) {
    handleFile(fileInput.files[0]);
  }
});

async function handleFile(file) {
  const download_json = document.getElementById("download-json");
  if (!file) return;

  // Проверяем расширение
  if (!file.name.endsWith(".json")) {
    setStatus("Пожалуйста, выберите JSON файл!", "error");
    return;
  }

  const reader = new FileReader();
  reader.onload = async (e) => {
    try {
      const jsonData = JSON.parse(e.target.result);

      setStatus("Отправка данных на сервер...", "info");
      resultBlock.style.display = "none";

      // Отправка на сервер
      const response = await window.API.predict(jsonData);
      window.lastUploadedJson = response;

      setStatus("Данные успешно обработаны!", "success");
      resultBlock.style.display = "block";
      download_json.style.display = "block";
      displayPredictions(response); // Вывод карточек с пагинацией

    } catch (err) {
      setStatus(`Ошибка при чтении или обработке JSON файла!(${err.message})`, "error");
      resultBlock.style.display = "none";
    } finally {
      // Сброс input, чтобы можно было повторно выбрать тот же файл
      fileInput.value = "";
    }
  };
  reader.readAsText(file);
}


// Download back uploaded JSON
try { document.getElementById('btn-dl-upload-json')?.addEventListener('click', () => {
  try {
    const data = window.lastUploadedJson;
    if (!data) { setStatus('Нет данных для скачивания', 'warning'); return; }
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = 'analysis_response.json';
    document.body.appendChild(a); a.click(); a.remove();
    URL.revokeObjectURL(url);
  } catch (err) {
    setStatus('Ошибка скачивания JSON', 'error');
  }
}); } catch (e) {}




