/* places.js — full catalogue with type filter, star rating,
   and a live "Your recommendations" strip that refreshes on rating. */

const state = { type: "all", ratings: {} };

function renderRecommendations(recs) {
  const track = document.getElementById("recTrack");
  if (!track) return;
  track.innerHTML = "";
  recs.forEach((r, i) => track.appendChild(buildRecCard(r, i)));
}

async function loadRecommendations() {
  if (!window.IS_AUTHENTICATED) return;
  const { recommendations } = await getJSON("/api/recommendations/?n=6");
  renderRecommendations(recommendations);
}

async function loadMyRatings() {
  if (!window.IS_AUTHENTICATED) return;
  const { ratings } = await getJSON("/api/my-ratings/");
  state.ratings = {};
  for (const [aid, r] of Object.entries(ratings)) state.ratings[parseInt(aid, 10)] = r;
}

async function loadGrid() {
  const { attractions } = await getJSON(`/api/attractions/?type=${encodeURIComponent(state.type)}`);
  const grid = document.getElementById("grid");
  grid.innerHTML = "";
  attractions.forEach(att => {
    grid.appendChild(buildAttractionCard(att, {
      rateable: true,
      rated: state.ratings[att.id],
    }));
  });
  initReveal();
}

function initFilter() {
  document.getElementById("typeFilter").addEventListener("click", (e) => {
    const btn = e.target.closest(".fchip");
    if (!btn) return;
    document.querySelectorAll(".fchip").forEach(c => c.classList.remove("active"));
    btn.classList.add("active");
    state.type = btn.dataset.type;
    loadGrid();
  });
}

function applyTypeFromURL() {
  const wanted = new URLSearchParams(location.search).get("type");
  if (!wanted) return;
  const chips = document.querySelectorAll(".fchip");
  for (const c of chips) {
    if (c.dataset.type.toLowerCase() === wanted.toLowerCase()) {
      chips.forEach(x => x.classList.remove("active"));
      c.classList.add("active");
      state.type = c.dataset.type;
      break;
    }
  }
}

(async function init() {
  initFilter();
  applyTypeFromURL();
  await loadMyRatings();
  await Promise.all([loadGrid(), loadRecommendations()]);
})();
