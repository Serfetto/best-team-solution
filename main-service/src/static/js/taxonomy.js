// ===== taxonomy.js =====
const TAXONOMY_STATUS_TIMEOUT = 4000;

const taxonomyEnsureArray = (value) => (Array.isArray(value) ? value : []);

const taxonomyClean = (items) => {
  const result = [];
  const seen = new Set();
  taxonomyEnsureArray(items).forEach((raw) => {
    const value = String(raw ?? '').trim();
    if (!value) return;
    const key = value.toLocaleLowerCase();
    if (seen.has(key)) return;
    seen.add(key);
    result.push(value);
  });
  return result;
};

const taxonomyArraysEqual = (a, b) => {
  if (a === b) return true;
  if (!Array.isArray(a) || !Array.isArray(b)) return false;
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i += 1) {
    if (a[i] !== b[i]) return false;
  }
  return true;
};

const serverList = document.getElementById('taxonomy-server-list');
const derivedList = document.getElementById('taxonomy-derived-list');
const serverEmptyHint = document.getElementById('taxonomy-server-empty');
const derivedEmptyHint = document.getElementById('taxonomy-derived-empty');

let sortableInstances = [];

const updateEmptyHints = () => {
  if (serverEmptyHint) {
    serverEmptyHint.classList.toggle('hidden', Boolean(window.serverTaxonomy?.length));
  }
  if (derivedEmptyHint) {
    derivedEmptyHint.classList.toggle('hidden', Boolean(window.derivedTaxonomy?.length));
  }
};

const setTaxonomyStatus = (message, state = '') => {
  const status = document.getElementById('taxonomy-status');
  if (!status) return;
  if (window.taxonomyStatusTimer) {
    clearTimeout(window.taxonomyStatusTimer);
    window.taxonomyStatusTimer = null;
  }
  if (!message) {
    status.textContent = '';
    delete status.dataset.state;
    return;
  }
  status.textContent = message;
  if (state) status.dataset.state = state;
  else delete status.dataset.state;
  window.taxonomyStatusTimer = setTimeout(() => {
    status.textContent = '';
    delete status.dataset.state;
    window.taxonomyStatusTimer = null;
  }, TAXONOMY_STATUS_TIMEOUT);
};

const createActionButton = (action, label, index) => {
  const button = document.createElement('button');
  button.type = 'button';
  button.className = 'btn small';
  button.dataset.action = action;
  button.dataset.index = String(index);
  button.textContent = label;
  return button;
};

const createItem = (value, type, index) => {
  const li = document.createElement('li');
  li.className = 'taxonomy-item';
  li.dataset.type = type;
  li.dataset.value = value;

  const drag = document.createElement('span');
  drag.className = 'drag-handle';
  drag.setAttribute('aria-hidden', 'true');
  drag.textContent = '⋮⋮';

  const text = document.createElement('span');
  text.className = 'text';
  text.textContent = value;

  const actions = document.createElement('div');
  actions.className = 'actions';

  if (type === 'server') {
    actions.append(createActionButton('edit', 'Изменить', index));
    actions.append(createActionButton('remove', 'Удалить', index));
  } else {
    actions.append(createActionButton('promote', 'Добавить', index));
    actions.append(createActionButton('remove', 'Удалить', index));
  }

  li.append(drag, text, actions);
  return li;
};

const renderList = (target, items, type) => {
  if (!target) return;
  target.innerHTML = '';
  taxonomyEnsureArray(items).forEach((value, index) => {
    target.appendChild(createItem(value, type, index));
  });
};

const destroySortables = () => {
  sortableInstances.forEach((instance) => instance.destroy());
  sortableInstances = [];
};

const setSaveButtonState = (state) => {
  const button = document.getElementById('btn-save-taxonomy');
  if (!button) return;
  if (state === 'dirty') {
    button.dataset.state = 'dirty';
  } else {
    delete button.dataset.state;
  }
};

const syncTaxonomyDirtyState = () => {
  const currentBase = Array.isArray(window.serverTaxonomy) ? window.serverTaxonomy : [];
  const currentDerived = Array.isArray(window.derivedTaxonomy) ? window.derivedTaxonomy : [];
  const initialBase = Array.isArray(window.taxonomyInitialState?.server)
    ? window.taxonomyInitialState.server
    : [];
  const initialDerived = Array.isArray(window.taxonomyInitialState?.derived)
    ? window.taxonomyInitialState.derived
    : [];
  const dirty =
    !taxonomyArraysEqual(currentBase, initialBase) ||
    !taxonomyArraysEqual(currentDerived, initialDerived);
  window.taxonomyDirty = dirty;
  setSaveButtonState(dirty ? 'dirty' : undefined);
  return dirty;
};

const markButtonAction = () => {
  syncTaxonomyDirtyState();
};

const resetSaveButtonState = () => {
  window.taxonomyDirty = false;
  setSaveButtonState();
};

const commitStateFromDom = () => {
  if (!serverList || !derivedList) return;
  const valuesFrom = (listEl) =>
    Array.from(listEl.querySelectorAll('.taxonomy-item .text')).map((node) => node.textContent.trim());

  const serverValues = taxonomyClean(valuesFrom(serverList));
  const derivedValues = taxonomyClean(valuesFrom(derivedList));
  const serverKeys = new Set(serverValues.map((value) => value.toLocaleLowerCase()));
  const filteredDerived = derivedValues.filter((value) => !serverKeys.has(value.toLocaleLowerCase()));

  const serverChanged = !taxonomyArraysEqual(serverValues, window.serverTaxonomy);
  const derivedChanged = !taxonomyArraysEqual(filteredDerived, window.derivedTaxonomy);

  window.serverTaxonomy = serverValues;
  window.derivedTaxonomy = filteredDerived;

  if (serverChanged || derivedChanged || filteredDerived.length !== derivedValues.length) {
    renderTaxonomyLists();
  } else {
    updateEmptyHints();
  }
  syncTaxonomyDirtyState();
};

const initSortables = () => {
  if (!serverList || !derivedList || typeof Sortable === 'undefined') return;
  destroySortables();
  const options = {
    group: 'taxonomy-board',
    animation: 160,
    handle: '.drag-handle',
    ghostClass: 'taxonomy-ghost',
    dragClass: 'taxonomy-drag',
    fallbackOnBody: true,
    swapThreshold: 0.2,
    onEnd: commitStateFromDom,
    onAdd: commitStateFromDom,
    onSort: commitStateFromDom,
  };
  sortableInstances = [
    new Sortable(serverList, options),
    new Sortable(derivedList, options),
  ];
};

const renderTaxonomyLists = () => {
  renderList(serverList, window.serverTaxonomy, 'server');
  renderList(derivedList, window.derivedTaxonomy, 'derived');
  updateEmptyHints();
  initSortables();
};

const setServerTaxonomy = (items = []) => {
  window.serverTaxonomy = taxonomyClean(items);
  renderTaxonomyLists();
};

const setDerivedTaxonomy = (items = [], options = {}) => {
  const append = Boolean(options.append);
  const cleaned = taxonomyClean(items);
  const serverKeys = new Set(window.serverTaxonomy.map((value) => value.toLocaleLowerCase()));
  const base = append ? window.derivedTaxonomy.slice() : [];
  const seen = new Set(base.map((value) => value.toLocaleLowerCase()));
  cleaned.forEach((value) => {
    const key = value.toLocaleLowerCase();
    if (serverKeys.has(key) || seen.has(key)) return;
    seen.add(key);
    base.push(value);
  });
  if (!taxonomyArraysEqual(base, window.derivedTaxonomy)) {
    window.derivedTaxonomy = base;
    renderTaxonomyLists();
    return true;
  }
  updateEmptyHints();
  return false;
};

const addTaxonomyItem = (raw) => {
  const [value] = taxonomyClean([raw]);
  if (!value) {
    setTaxonomyStatus('Введите название категории', 'warning');
    return;
  }
  const key = value.toLocaleLowerCase();
  const exists = window.serverTaxonomy.some((item) => item.toLocaleLowerCase() === key);
  if (exists) {
    setTaxonomyStatus('Такая категория уже есть', 'warning');
    return;
  }
  window.serverTaxonomy.push(value);
  window.derivedTaxonomy = window.derivedTaxonomy.filter((item) => item.toLocaleLowerCase() !== key);
  renderTaxonomyLists();
  markButtonAction();
  setTaxonomyStatus('Категория добавлена', 'success');
};

const editTaxonomyItem = (index) => {
  const current = window.serverTaxonomy[index];
  if (typeof current !== 'string') return;
  const next = prompt('Изменить категорию', current);
  if (next === null) return;
  const [value] = taxonomyClean([next]);
  if (!value) {
    setTaxonomyStatus('Введите корректное название', 'warning');
    return;
  }
  const key = value.toLocaleLowerCase();
  const duplicate = window.serverTaxonomy.some((item, idx) => idx !== index && item.toLocaleLowerCase() === key);
  if (duplicate) {
    setTaxonomyStatus('Такая категория уже есть', 'warning');
    return;
  }
  window.serverTaxonomy[index] = value;
  window.derivedTaxonomy = window.derivedTaxonomy.filter((item) => item.toLocaleLowerCase() !== key);
  renderTaxonomyLists();
  markButtonAction();
  setTaxonomyStatus('Категория обновлена', 'success');
};

const removeTaxonomyItem = (index) => {
  if (index < 0 || index >= window.serverTaxonomy.length) return;
  const [removed] = window.serverTaxonomy.splice(index, 1);
  if (removed) {
    const appended = setDerivedTaxonomy([removed], { append: true });
    if (!appended) {
      renderTaxonomyLists();
    }
    markButtonAction();
    setTaxonomyStatus('Категория перемещена в предложения', 'info');
  } else {
    renderTaxonomyLists();
  }
};



const promoteDerivedTaxonomy = (index) => {
  if (index < 0 || index >= window.derivedTaxonomy.length) return;
  const [value] = window.derivedTaxonomy.splice(index, 1);
  if (!value) {
    renderTaxonomyLists();
    return;
  }
  const key = value.toLocaleLowerCase();
  const exists = window.serverTaxonomy.some((item) => item.toLocaleLowerCase() === key);
  if (!exists) {
    window.serverTaxonomy.push(value);
    setTaxonomyStatus('Категория добавлена в базовую таксономию', 'success');
  } else {
    setTaxonomyStatus('Категория уже есть в базовой таксономии', 'warning');
  }
  renderTaxonomyLists();
  markButtonAction();
};

const removeDerivedTaxonomyItem = (index) => {
  if (index < 0 || index >= window.derivedTaxonomy.length) return;
  window.derivedTaxonomy.splice(index, 1);
  renderTaxonomyLists();
  markButtonAction();
  setTaxonomyStatus('Категория удалена из предложений', 'info');
};

const loadServerTaxonomy = async function (force = false) {
  if (!force && window.taxonomyLoaded && window.serverTaxonomy.length) return;
  try {
    setTaxonomyStatus('Loading taxonomy...', 'info');
    const data = await API.taxonomyGet();
    const baseItems = taxonomyClean(taxonomyEnsureArray(data?.base ?? data?.items));
    const derivedItems = taxonomyClean(taxonomyEnsureArray(data?.full));
    window.derivedTaxonomy = [];
    setServerTaxonomy(baseItems);
    setDerivedTaxonomy(derivedItems);
    window.taxonomyInitialState = {
      server: window.serverTaxonomy.slice(),
      derived: window.derivedTaxonomy.slice(),
    };
    resetSaveButtonState();
    const summaryParts = [`Base categories: ${window.serverTaxonomy.length}`];
    if (window.derivedTaxonomy.length) {
      summaryParts.push(`New categories: ${window.derivedTaxonomy.length}`);
    }
    window.taxonomyLoaded = true;
    setTaxonomyStatus(`Loaded. ${summaryParts.join(' | ')}`, 'success');
  } catch (error) {
    console.error(error);
    setTaxonomyStatus('Failed to load taxonomy', 'error');
  }
};

const TAXONOMY_CANDIDATE_PATHS = [
  ['derived_taxonomy'],
  ['mapping', 'derived_taxonomy'],
  ['report', 'derived_taxonomy'],
  ['report', 'mapping', 'derived_taxonomy'],
  ['analysis', 'derived_taxonomy'],
  ['new_taxonomy'],
];

const extractByPath = (payload, path) =>
  path.reduce((acc, key) => (acc && acc[key] !== undefined ? acc[key] : undefined), payload);

const extractDerivedTaxonomy = (payload) => {
  if (!payload) return [];
  const collected = [];
  TAXONOMY_CANDIDATE_PATHS.forEach((path) => {
    const candidate = extractByPath(payload, path);
    if (Array.isArray(candidate)) {
      collected.push(...candidate);
    }
  });
  return taxonomyClean(collected);
};

const registerDerivedTaxonomy = (payload, options = {}) => {
  const derived = extractDerivedTaxonomy(payload);
  if (!derived.length) return;
  const changed = setDerivedTaxonomy(derived, options);
  if (changed) {
    setTaxonomyStatus('Найдены новые предложения таксономии', 'success');
  }
};



window.setTaxonomyStatus = setTaxonomyStatus;
window.renderTaxonomyLists = renderTaxonomyLists;
window.setServerTaxonomy = setServerTaxonomy;
window.setDerivedTaxonomy = setDerivedTaxonomy;
window.addTaxonomyItem = addTaxonomyItem;
window.editTaxonomyItem = editTaxonomyItem;
window.removeTaxonomyItem = removeTaxonomyItem;
window.promoteDerivedTaxonomy = promoteDerivedTaxonomy;
window.removeDerivedTaxonomyItem = removeDerivedTaxonomyItem;
window.loadServerTaxonomy = loadServerTaxonomy;
window.registerDerivedTaxonomy = registerDerivedTaxonomy;
