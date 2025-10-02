// Simple toast notifications driven by #status_erorrs content
(function initErrorNotifications(){
  let lastMsg = '';
  let lastAt = 0;
  const ID = 'status_erorrs';

  function ensureContainer(){
    let el = document.getElementById('toast-container');
    if (!el){
      el = document.createElement('div');
      el.id = 'toast-container';
      document.body.appendChild(el);
    }
    return el;
  }

  function showToast(message, opts){
    const { type = 'error', timeout = 5000 } = opts || {};
    const container = ensureContainer();
    const toast = document.createElement('div');
    toast.className = `toast toast--${type}`;
    toast.setAttribute('role', 'status');
    toast.setAttribute('aria-live', 'polite');

    const text = document.createElement('div');
    text.className = 'toast__text';
    text.textContent = message;

    const close = document.createElement('button');
    close.className = 'toast__close';
    close.type = 'button';
    close.setAttribute('aria-label', 'Close notification');
    close.textContent = '×';
    close.addEventListener('click', () => {
      toast.classList.add('is-hiding');
      setTimeout(() => toast.remove(), 180);
    });

    toast.appendChild(text);
    toast.appendChild(close);
    container.appendChild(toast);

    // Auto-dismiss
    if (timeout > 0){
      setTimeout(() => {
        if (toast.isConnected){
          toast.classList.add('is-hiding');
          setTimeout(() => toast.remove(), 180);
        }
      }, timeout);
    }
  }

  function initObserver(target){
    const obs = new MutationObserver(() => {
      let msg = target.textContent.trim();
      // срежем « 12:34:56.123456» в конце, если вдруг появился
      //msg = msg.replace(/\s\d{2}:\d{2}:\d{2}\.\d{3,6}\s*$/, '');
      if (!msg) return;
      if (msg.toLowerCase() === 'success') return; // don't notify on success
      const now = Date.now();
      // Avoid spamming the same error repeatedly (interval refreshes)
      //if (msg === lastMsg && now - lastAt < 8000) return;
      lastMsg = msg; lastAt = now;
      showToast(msg, { type: 'error', timeout: 6000 });
    });
    obs.observe(target, { childList: true, characterData: true, subtree: true });
  }

  function waitForTarget(attempts){
    const el = document.getElementById(ID);
    if (el){
      initObserver(el);
      return;
    }
    if ((attempts || 0) > 200) return; // ~10s max
    setTimeout(() => waitForTarget((attempts || 0) + 1), 50);
  }

  if (document.readyState === 'loading'){
    document.addEventListener('DOMContentLoaded', () => waitForTarget(0));
  } else {
    waitForTarget(0);
  }
})();
