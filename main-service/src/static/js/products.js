// ===== products.js =====
(function () {
  const carouselInner = document.getElementById('products-carousel');
  const carouselPrev = document.getElementById('carousel-prev');
  const carouselNext = document.getElementById('carousel-next');
  const carouselIndicators = document.getElementById('carousel-indicators');
  const statusEl = document.getElementById('products-status');
  const btnRefresh = document.getElementById('btn-refresh-products');
  const btnApplyDates = document.getElementById('btn-apply-dates');
  const dateFromInput = document.getElementById('products-date-from');
  const dateToInput = document.getElementById('products-date-to');

  let loaded = false;
  let products = [];
  let currentIndex = 0;

  const productDateRanges = new Map();
  const urlDatesInitial = getDatesFromURL();
  const dateBoundsDefault = {
    dateFrom: urlDatesInitial?.dateFrom || '2024-01-01',
    dateTo: urlDatesInitial?.dateTo || '2025-05-31',
  };
  const initialProductKey = getProductFromURL();
  let isSyncingDates = false;

  function getProductFromURL() {
    try {
      const urlParams = new URLSearchParams(window.location.search);
      const product = urlParams.get('product');
      return (product || '').trim().toLowerCase() || null;
    } catch (_) {
      return null;
    }
  }

  // Функция для установки статуса
  function setStatus(msg, state = '') {
    if (!statusEl) return;
    statusEl.textContent = msg || '';
    if (state) statusEl.dataset.state = state;
    else delete statusEl.dataset.state;
  }

  function getDatesFromURL() {
      const urlParams = new URLSearchParams(window.location.search);
      const startDate = urlParams.get('start_date');
      const endDate = urlParams.get('end_date');
      
      if (startDate && endDate) {
          return { 
              dateFrom: startDate, 
              dateTo: endDate 
          };
      }
      return null;
  }

  function normalizeProductKey(value) {
    if (!value && value !== 0) return null;
    const asString = String(value).trim();
    return asString ? asString.toLowerCase() : null;
  }

  function getCurrentProductKey() {
    const current = products[currentIndex];
    if (!current) return null;
    return normalizeProductKey(current.title || current.id);
  }

  function ensureProductDateRange(productKey) {
    if (!productKey) {
      return { ...dateBoundsDefault };
    }
    if (!productDateRanges.has(productKey)) {
      const baseRange = { ...dateBoundsDefault };
      if (initialProductKey && productKey === initialProductKey && urlDatesInitial) {
        baseRange.dateFrom = urlDatesInitial.dateFrom || baseRange.dateFrom;
        baseRange.dateTo = urlDatesInitial.dateTo || baseRange.dateTo;
      }
      productDateRanges.set(productKey, baseRange);
    }
    const stored = productDateRanges.get(productKey);
    return stored ? { ...stored } : { ...dateBoundsDefault };
  }

  function syncInputsWithCurrentProduct() {
    const key = getCurrentProductKey();
    const range = ensureProductDateRange(key);
    setSelectedDates(range.dateFrom, range.dateTo);
  }

  function persistCurrentProductDates() {
    if (isSyncingDates) return;
    const key = getCurrentProductKey();
    if (!key) return;
    const { dateFrom, dateTo } = getSelectedDates();
    productDateRanges.set(key, {
      dateFrom: dateFrom || '',
      dateTo: dateTo || '',
    });
  }

  // Функция для получения дат из полей ввода
  function getSelectedDates() {
    return {
      dateFrom: dateFromInput?.value || '',
      dateTo: dateToInput?.value || ''
    };
  }

  // Функция для установки дат в поля ввода
  function setSelectedDates(dateFrom, dateTo) {
    isSyncingDates = true;
    try {
      if (dateFromInput) {
        dateFromInput.value = dateFrom ?? '';
      }
      if (dateToInput) {
        dateToInput.value = dateTo ?? '';
      }
    } finally {
      isSyncingDates = false;
    }
  }

  // Функция для загрузки продуктов с учетом выбранных дат
  async function loadProducts(force = false) {
    if (loaded && !force) return;
    try {
      setStatus('Загружаю продукты…', 'info');
      const { dateFrom, dateTo } = getSelectedDates();
      
      const payload = { 
        date_from: dateFrom || undefined,
        date_to: dateTo || undefined
      };
      
      const result = await API.InsightsProduct(payload);
      products = normalizePayload(result);
      // Если в URL указан продукт — переключаемся на него
      const urlProduct = getProductFromURL();
      if (urlProduct) {
        const idx = products.findIndex(p => (p.title || '').toString().trim().toLowerCase() === urlProduct);
        currentIndex = idx >= 0 ? idx : 0;
      } else {
        currentIndex = 0;
      }
      loaded = true;
      renderCarousel();
      setStatus(`Загружено ${products.length} продуктов`, 'success');
    } catch (e) {
      setStatus('Не удалось загрузить продукты', 'error');
    }
  }
  function capitalizeFirst(str) {
    if (!str) return '';
    return str.charAt(0).toUpperCase() + str.slice(1).toLowerCase();
  }
  btnApplyDates?.addEventListener('click', async () => {
    try {
      btnApplyDates.disabled = true;
      setStatus('Обновляю по выбранным датам…', 'info');

      const { dateFrom, dateTo } = getSelectedDates();
      persistCurrentProductDates();
      const product_name = (products[currentIndex]?.title || '').toLowerCase();

      const payload = { 
        product_name,
        date_from: dateFrom,
        date_to: dateTo
      };


      const res = await API.InsightsDescriptionProducts(payload);
      const item = normalizePayload(res)[0];
      if (item) {
        // Обновляем ТОЛЬКО выбранную карточку нужными полями
        const prev = products[currentIndex] || {};
        products[currentIndex] = {
          ...prev,
          title: capitalizeFirst(item.title),
          description: item.description,
          basis: item.basis,
        };
        renderCarousel();
        setStatus('Данные обновились.', 'info');
      } else {
        setStatus('Нет данных по выбранному периоду для этого продукта', 'warning');
      }
    } catch (e) {
      setStatus('Не удалось обновить по датам', 'error');
    } finally {
      btnApplyDates.disabled = false;
    }
  });

  function normalizeOne(raw, idx = 0) {
    const title = raw?.product_name || raw?.title || `Продукт ${idx + 1}`;
    const desc = raw?.description || '';
    const basis = Array.isArray(raw?.examples) ? raw.examples : [];
    const id = raw?.id || raw?.slug || String(idx);
    const strengths = raw?.strengths_summary || '';
    const weaknesses = raw?.weaknesses_summary || '';
    return {
      id,
      title,
      description: String(desc || ''),
      basis,
      strengths: String(strengths || ''),
      weaknesses: String(weaknesses || ''),
    };
  }

  function normalizePayload(payload) {
    if (Array.isArray(payload)) return payload.map((r, i) => normalizeOne(r, i));
    if (payload && typeof payload === 'object') return [normalizeOne(payload, 0)];
    return [];
  }

  function createProductCard(product) {
    const card = document.createElement('div');
    card.className = 'product-card';
    
    const title = document.createElement('h4');
    title.textContent = product.title;

    const desc = document.createElement('p');
    desc.className = 'statement';
    desc.textContent = product.description || '—';

    // Сильные стороны (зеленый)
    const strengthsSection = document.createElement('div');
    strengthsSection.className = 'product-section';
    
    const strengthsTitle = document.createElement('div');
    strengthsTitle.className = 'section-title';
    strengthsTitle.textContent = 'Сильные стороны:';
    
    const strengthsContent = document.createElement('div');
    strengthsContent.className = 'section-content strengths';
    strengthsContent.textContent = product.strengths || '—';
    
    strengthsSection.append(strengthsTitle, strengthsContent);

    // Слабые стороны (красный)
    const weaknessesSection = document.createElement('div');
    weaknessesSection.className = 'product-section';
    
    const weaknessesTitle = document.createElement('div');
    weaknessesTitle.className = 'section-title';
    weaknessesTitle.textContent = 'Слабые стороны:';
    
    const weaknessesContent = document.createElement('div');
    weaknessesContent.className = 'section-content weaknesses';
    weaknessesContent.textContent = product.weaknesses || '—';
    
    weaknessesSection.append(weaknessesTitle, weaknessesContent);

    const basisTitle = document.createElement('div');
    basisTitle.className = 'section-title';
    basisTitle.textContent = 'На основе чего сформирован:';

    const basisList = document.createElement('ul');
    basisList.className = 'section-list';

    if (product.basis && product.basis.length) {
      // всегда берём только первые 5 элементов
      product.basis.slice(0, 5).forEach((b) => {
        const li = document.createElement('li');
        li.textContent = typeof b === 'string' ? b : JSON.stringify(b);
        basisList.appendChild(li);
      });

      // и всегда добавляем "…и т.п."
      const liEtc = document.createElement('li');
      liEtc.textContent = '…и т.п.';
      basisList.appendChild(liEtc);
    } else {
      const li = document.createElement('li');
      li.textContent = '—';
      basisList.appendChild(li);
    }

    const extra = document.createElement('div');
    extra.id = 'product-extra';

    const metrics = document.createElement('div');
    metrics.id = 'product-metrics';

    const graph1 = document.createElement('div');
    graph1.id = 'product-graph-reviews';

    const graph2 = document.createElement('div');
    graph2.id = 'product-graph-mean';

    const sentimentGraph = document.createElement('div');
    sentimentGraph.id = 'product-sentiment';

    const cloudGraph = document.createElement('div');
    cloudGraph.id = 'product-cloud';

    const cloudWrapper = document.createElement('div');
    cloudWrapper.className = 'cloud-wrapper';

    cloudWrapper.appendChild(cloudGraph);

    extra.append(metrics, sentimentGraph, graph1, graph2, cloudWrapper);

    card.append(
      title,
      desc,
      strengthsSection,
      weaknessesSection,
      basisTitle,
      basisList,
      extra   // <-- добавляем в карточку
    );

    return card;
  }

  function renderCarousel() {
    if (!carouselInner) return;
    
    carouselInner.innerHTML = '';
    carouselIndicators.innerHTML = '';

    if (!products.length) {
      carouselInner.innerHTML = '<p class="hint">Нет данных для отображения.</p>';
      return;
    }

    // Создаем карточку для текущего продукта
    const card = createProductCard(products[currentIndex]);
    carouselInner.appendChild(card);

    syncInputsWithCurrentProduct();
    loadProductGraphs(products[currentIndex]);

    // Создаем индикаторы
    products.forEach((_, index) => {
      const indicator = document.createElement('button');
      indicator.className = `carousel-indicator ${index === currentIndex ? 'active' : ''}`;
      indicator.addEventListener('click', () => {
        persistCurrentProductDates();
        currentIndex = index;
        renderCarousel();
        updateNavigation();
      });
      carouselIndicators.appendChild(indicator);
    });

    updateNavigation();
  }

  function updateNavigation() {
    if (carouselPrev) {
      carouselPrev.disabled = currentIndex === 0;
    }
    if (carouselNext) {
      carouselNext.disabled = currentIndex === products.length - 1;
    }
  }

  function nextProduct() {
    if (currentIndex < products.length - 1) {
      persistCurrentProductDates();
      currentIndex++;
      renderCarousel();
    }
  }

  function prevProduct() {
    if (currentIndex > 0) {
      persistCurrentProductDates();
      currentIndex--;
      renderCarousel();
    }
  }
  // Инициализация дат по умолчанию (последние 30 дней)
  function initializeDefaultDates() {
    if (dateFromInput) {
      dateFromInput.min = dateBoundsDefault.dateFrom;
      dateFromInput.max = dateBoundsDefault.dateTo;
    }
    if (dateToInput) {
      dateToInput.min = dateBoundsDefault.dateFrom;
      dateToInput.max = dateBoundsDefault.dateTo;
    }
    syncInputsWithCurrentProduct();
  }

  async function loadProductGraphs(product) {
    try {
      const { dateFrom, dateTo } = getSelectedDates();
      const payload = {
        product_name: product.title.toLowerCase(),
        date_from: dateFrom,
        date_to: dateTo
      };
      const res = await API.ProductGraphs(payload);

      // Находим текущую карточку и её контейнеры
      const card = document.querySelector('.product-card');
      if (!card) return;

      const metricsEl = card.querySelector('#product-metrics');
      const reviewsEl = card.querySelector('#product-graph-reviews');
      const meanEl = card.querySelector('#product-graph-mean');
      const sentimentEl = card.querySelector('#product-sentiment');
      const cloudEl = card.querySelector('#product-cloud');

      if (metricsEl) {
        metricsEl.innerHTML = `
          <div class="metric">
            <div class="metric-value">${res.total_reviews ?? '—'}</div>
            <div class="metric-label">Кол-во упоминаний</div>
          </div>
          <div class="metric">
            <div class="metric-value">${res.average_rating?.toFixed(2) ?? '—'}</div>
            <div class="metric-label">Средний рейтинг</div>
          </div>
        `;
      }

      if (reviewsEl && res.fig_reviews) {
        Plotly.newPlot(reviewsEl, res.fig_reviews.data, res.fig_reviews.layout, res.config);
      }

      if (meanEl && res.fig_mean) {
        Plotly.newPlot(meanEl, res.fig_mean.data, res.fig_mean.layout, res.config);
      }

      if (sentimentEl && res.fig_sentiment) {
        Plotly.newPlot(sentimentEl, res.fig_sentiment.data, res.fig_sentiment.layout, res.config);
      }
      if (cloudEl && res.fig_cloud) {

        Plotly.purge(cloudEl);
        cloudEl.innerHTML = '';

        Plotly.newPlot(
          cloudEl,
          res.fig_cloud.data,
          res.fig_cloud.layout,
          {
            ...(res.config || {}),
            displaylogo: false,
            scrollZoom: false,
            modeBarButtonsToRemove: [
              "zoom2d", "pan2d", "select2d", "lasso2d",
              "zoomIn2d", "zoomOut2d", "autoScale2d", "resetScale2d"
            ]
          }
        );
      }
    } catch (e) {
      console.error('Ошибка при загрузке графиков продукта:', e);
    }
  }

  // Обработчики событий
  carouselPrev?.addEventListener('click', prevProduct);
  carouselNext?.addEventListener('click', nextProduct);
  btnRefresh?.addEventListener('click', () => {
    persistCurrentProductDates();
    currentIndex = 0;
    loadProducts(true);
  });

  dateFromInput?.addEventListener('change', persistCurrentProductDates);
  dateToInput?.addEventListener('change', persistCurrentProductDates);

  // Навигация с клавиатуры
  document.addEventListener('keydown', (e) => {
    if (document.getElementById('products').classList.contains('visible')) {
      if (e.key === 'ArrowLeft') prevProduct();
      if (e.key === 'ArrowRight') nextProduct();
    }
  });

  // Запуск при входе на вкладку
  document
    .querySelectorAll('.vnav a[data-target="products"], .tab[data-target="products"]')
    .forEach((node) => node.addEventListener('click', () => {
      initializeDefaultDates();
      loadProducts(false);
    }));

  // Инициализация при загрузке страницы, если активна вкладка продуктов
  if (location.hash === '#products') {
    initializeDefaultDates();
    loadProducts(false);
  }
})();
