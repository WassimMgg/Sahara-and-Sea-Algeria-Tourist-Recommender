/* places.js — full catalogue with type filter, star rating,
   live recommendations strip, skeleton loading states, and pagination. */

const state = {
  type: "all",
  ratings: {},
  limit: 20,
  offset: 0,
  total: 0,
};

function renderRecommendations(recs) {
  const track = document.getElementById("recTrack");
  if (!track) return;
  track.innerHTML = "";
  recs.forEach((r, i) => track.appendChild(buildRecCard(r, i)));
  initReveal();
}

async function loadRecommendations() {
  if (!window.IS_AUTHENTICATED) return;
  const track = document.getElementById("recTrack");
  if (track) track.innerHTML = renderRecSkeletons(4);
  try {
    const { recommendations } = await getJSON("/api/recommendations/?n=6");
    renderRecommendations(recommendations);
  } catch (_) {
    if (track) track.innerHTML = "";
  }
}

async function loadMyRatings() {
  if (!window.IS_AUTHENTICATED) return;
  const { ratings } = await getJSON("/api/my-ratings/");
  state.ratings = {};
  for (const [aid, r] of Object.entries(ratings)) state.ratings[parseInt(aid, 10)] = r;
}

async function loadGrid(append = false) {
  const grid = document.getElementById("grid");
  if (!append) {
    state.offset = 0;
    grid.innerHTML = renderSkeletons(state.limit);
  }

  const url = `/api/attractions/?type=${encodeURIComponent(state.type)}&limit=${state.limit}&offset=${state.offset}`;
  const { attractions, total } = await getJSON(url);
  state.total = total;

  if (!append) grid.innerHTML = "";
  attractions.forEach(att => {
    grid.appendChild(buildAttractionCard(att, {
      rateable: true,
      rated: state.ratings[att.id],
    }));
  });
  initReveal();

  // update "Load more" button
  const btn = document.getElementById("loadMoreBtn");
  if (btn) {
    const shown = state.offset + attractions.length;
    const remaining = total - shown;
    if (remaining > 0) {
      btn.textContent = `Load more (${remaining} remaining)`;
      btn.style.display = "";
    } else {
      btn.style.display = "none";
    }
  }
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

function initLoadMore() {
  const btn = document.getElementById("loadMoreBtn");
  if (!btn) return;
  btn.addEventListener("click", () => {
    state.offset += state.limit;
    loadGrid(true);
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
  initLoadMore();
  applyTypeFromURL();
  await loadMyRatings();
  await Promise.all([loadGrid(), loadRecommendations()]);
})();
