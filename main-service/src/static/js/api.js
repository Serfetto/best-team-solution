// ===== api.js =====
window.API = {
  taxonomyGet: async () => {
    const r = await fetch("/taxonomy/");
    return check_status(r);
  },
  taxonomyPut: async (payload) => {
    const r = await fetch("/taxonomy/", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    return check_status(r);
  },
  taxonomyEdit: async (payload) => {
    const r = await fetch("/taxonomy/edit", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    return check_status(r);
  },
  taxonomyShuffle: async (payload) => {
    const r = await fetch("/taxonomy/shuffle", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    return check_status(r);
  },
  exportCombined: async () => {
    const r = await fetch("/export/combined");
    return check_status_for_export(r);
  },
  exportByName: async (name) => {
    const r = await fetch(`/export/dataset/${name}`);
    return check_status_for_export(r);
  },
  exportEnrichedCsv: async () => {
    const r = await fetch("/export/enriched_combined.csv");
    return check_status_for_export(r);
  },
  predict: async (payload) => {
    const r = await fetch("/analysis/predict/", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    return check_status(r);
  },
  parsingStatus: async () => {
    const r = await fetch("/parsing/status/");
    return check_status(r);
  },
  parsingStart: async (payload) => {
    const r = await fetch("/parsing/start/", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    return check_status(r);
  },
  parsingEdit: async (payload) => {
    const r = await fetch("/parsing/edit/", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    return check_status(r);
  },
  parsingStop: async () => {
    const r = await fetch("/parsing/stop/", { method: "POST" });
    return check_status(r);
  },
  products: async (payload) => {
    const r = await fetch("/analysis/insights/all-product-description-summary/", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    return check_status(r);
  },
  InsightsDescriptionProducts: async (payload) => {
    const r = await fetch("/analysis/insights/product-description/", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    return check_status(r);
  },
  InsightsSummaryProducts: async (payload) => {
    const r = await fetch("/analysis/insights/product-summary/", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    return check_status(r);
  },
  InsightsProduct: async () => {
    const r = await fetch("/analysis/insights/products/");
    return check_status(r);
  },
  ProductGraphs: async (payload) => {
    const r = await fetch("/analysis/insights/product-graphs/", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    return check_status(r);
  },
};



