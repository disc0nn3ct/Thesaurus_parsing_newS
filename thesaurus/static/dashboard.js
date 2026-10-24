"use strict";
// Charts use JSON and the installed Plotly package, never external CDNs.
document.querySelectorAll("[data-chart-url]").forEach(async (element) => {
  try {
    const response = await fetch(element.dataset.chartUrl, {credentials: "same-origin"});
    if (!response.ok) throw new Error("Chart unavailable");
    const data = await response.json();
    if (!data.traces.length) {
      element.className = "empty";
      element.textContent = data.message || "Нет наблюдений для выбранного периода и фильтров.";
      return;
    }
    const traces = data.traces.map((trace) => ({
      x: trace.dates, y: trace.values, name: trace.name,
      type: "scatter", mode: "lines+markers", marker: {size: 4}, line: {width: 2},
      hovertemplate: "%{x}<br>%{y}<extra>%{fullData.name}</extra>"
    }));
    element.replaceChildren();
    await Plotly.newPlot(element, traces, {
      font: {family: "Segoe UI, sans-serif", color: "#18333d"},
      paper_bgcolor: "transparent", plot_bgcolor: "transparent",
      margin: {l: 55, r: 20, t: 25, b: 60},
      xaxis: {title: {text: "Дата"}, tickformat: "%d.%m.%Y", gridcolor: "#edf1f0"},
      yaxis: {title: {text: ({count: "Число запросов", "%": "Проценты"})[data.unit] || data.unit || "Значение"}, gridcolor: "#edf1f0"},
      legend: {orientation: "h", y: -0.23}, hovermode: "x unified",
      colorway: ["#126a5b", "#c48b30", "#456ead", "#ad635d", "#7b729b", "#5899a0"]
    }, {responsive: true, displaylogo: false, modeBarButtonsToRemove: ["sendDataToCloud"]});
  } catch (_) {
    element.className = "empty";
    element.textContent = "Не удалось построить график. Обновите страницу или проверьте состояние сервиса.";
  }
});
