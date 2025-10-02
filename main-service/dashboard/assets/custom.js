// Отслеживание Ctrl + ЛКМ / Ctrl + ПКМ по барам в графике "popular-products"
document.addEventListener("DOMContentLoaded", function () {
  function setupPopularProductsHandlers() {
    var container = document.getElementById("popular-products");
    if (!container) return;

    // В некоторых версиях Dash/Plotly нужный div уже является js-plotly-plot,
    // но на всякий случай поищем вложенный .js-plotly-plot.
    var gd = document.getElementById("popular-products").querySelector('.js-plotly-plot');
    if (!gd) return;

    var lastPoint = null;

    if (typeof gd.on === 'function') {
      // Храним последнюю наведенную точку, чтобы уметь определить бар при ПКМ
      gd.on('plotly_hover', function (d) {
        lastPoint = d && d.points && d.points[0] ? d.points[0] : null;
      });
      gd.on('plotly_unhover', function () {
        lastPoint = null;
      });

      // ЛКМ: событие plotly_click отдаёт MouseEvent в data.event
      // Заменить участок внутри gd.on('plotly_click', ...)
      gd.on('plotly_click', function (data) {
        var evt = data && data.event;
        var isLeft = evt && evt.button === 0;
        if (!isLeft || !data.points || !data.points[0]) return;

        var p = data.points[0];
        var rawLabel = p.label || p.y || p.x;
        let slug = (rawLabel || '').toString().trim().toLowerCase();

        if (evt.ctrlKey || evt.metaKey) {
          return;
        }

        // ✅ обычный ЛКМ → переход
        try {
          var link = document.getElementById('products-link');
          var href = link && link.href ? link.href : 'http://localhost:8003/#products';
          var target = new URL(href);
          target.searchParams.set('product', slug);
          window.location.href = target.toString();
        } catch (e) {
          console.warn('Не удалось перейти на вкладку продуктов', e);
        }
      });


    }
  }

  // Дожидаемся, когда график будет отрисован
  if (document.readyState === 'complete') {
    setTimeout(setupPopularProductsHandlers, 0);
  } else {
    window.addEventListener('load', setupPopularProductsHandlers);
    // подстраховка, если отрисовка чуть позже
    setTimeout(setupPopularProductsHandlers, 800);
  }
});
