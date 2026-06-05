/* search.js — text search + type filter (discovery, read-only cards). */

const state = { type: "all", q: "" };
let debounce;

async function runSearch() {
  const params = new URLSearchParams({ type: state.type, q: state.q });
  const { attractions } = await getJSON(`/api/attractions/?${params.toString()}`);
  const grid = document.getElementById("grid");
  const count = document.getElementById("resultCount");
  const empty = document.getElementById("emptyState");
  grid.innerHTML = "";
  attractions.forEach(att => grid.appendChild(buildAttractionCard(att, { rateable: false })));
  count.textContent = `${attractions.length} place${attractions.length === 1 ? "" : "s"} found`;
  empty.hidden = attractions.length !== 0;
  initReveal();
}

function init() {
  document.getElementById("searchInput").addEventListener("input", (e) => {
    state.q = e.target.value.trim();
    clearTimeout(debounce);
    debounce = setTimeout(runSearch, 200);
  });
  document.getElementById("typeFilter").addEventListener("click", (e) => {
    const btn = e.target.closest(".fchip");
    if (!btn) return;
    document.querySelectorAll(".fchip").forEach(c => c.classList.remove("active"));
    btn.classList.add("active");
    state.type = btn.dataset.type;
    runSearch();
  });
  runSearch();
}
init();
