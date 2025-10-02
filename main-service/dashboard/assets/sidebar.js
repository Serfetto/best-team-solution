(function initSidebar(){
  // ждём, пока компоненты Dash появятся в DOM
  const sidebar   = document.getElementById('left-sidebar');
  const toggleBtn = document.getElementById('sidebar-toggle');

  if (!sidebar || !toggleBtn) {
    return setTimeout(initSidebar, 50);
  }

  function syncState(){
    const opened = sidebar.classList.contains('open');
    document.body.classList.toggle('sidebar-open', opened);
    toggleBtn.setAttribute('aria-expanded', opened ? 'true' : 'false');
    const label = opened ? 'Закрыть боковую панель' : 'Открыть боковую панель';
    toggleBtn.setAttribute('aria-label', label);
    toggleBtn.dataset.tooltip = label;
    
    // Перепозиционировать DatePicker при открытии/закрытии сайдбара
    setTimeout(() => {
      window.dispatchEvent(new Event('resize'));
    }, 300);
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

  // начальная синхронизация
  syncState();
})();