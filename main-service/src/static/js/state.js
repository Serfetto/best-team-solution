// ===== state.js =====
window.lastPredictions = [];
window.baseSummary = null;

window.reviewsBrief = $("#reviews-brief");

window.serverTaxonomy = [];
window.derivedTaxonomy = [];
window.taxonomyInitialState = { server: [], derived: [] };
window.taxonomyDirty = false;
window.taxonomyLoaded = false;
window.taxonomyStatusTimer = null;
window.lastAnalysisMeta = null;

// polling
window.pollingPaused = false;
window.POLL_MS_DEFAULT = 3600 * 1000;
window.pollTimer = null;
