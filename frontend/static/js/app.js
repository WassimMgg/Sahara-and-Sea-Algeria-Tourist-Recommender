/* ============================================================
   Sahara & Sea — application logic (vanilla JS)
   Structured as small "component" functions (loadX / renderX)
   so it maps cleanly onto Angular components if ported later.
   ============================================================ */

const API = "/api";
const TOP_N = 5;

const state = {
  userId: null,
  attractions: [],
  ratings: {},      // attraction_id -> rating (for the current user)
};

/* ----------------------------- fetch helpers ----------------------------- */
async function getJSON(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}
async function postJSON(url, body) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

/* ------------------------------- toast ----------------------------------- */
let toastTimer;
function toast(msg) {
  const el = document.getElementById("toast");
  el.textContent = msg;
  el.classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => el.classList.remove("show"), 2200);
}

/* --------------------------- model info panel ---------------------------- */
async function loadModelInfo() {
  const { model_name, metrics } = await getJSON(`${API}/model-info/`);
  document.getElementById("modelBadge").textContent = `Engine: ${model_name}`;
  renderMetrics(metrics);
}

function renderMetrics(metrics) {
  const wrap = document.getElementById("metricsTable");
  if (!metrics || !metrics.results) { wrap.innerHTML = ""; return; }
  const served = metrics.served;
  const rows = Object.entries(metrics.results)
    .sort((a, b) => a[1].rmse - b[1].rmse);

  let html = `<div class="metric-row head">
      <span>Model</span><span>RMSE</span><span>MAE</span></div>`;
  for (const [key, r] of rows) {
    const isServed = key === served;
    html += `<div class="metric-row ${isServed ? "served" : ""}">
        <span class="metric-name">${r.name}${isServed ? '<span class="served-tag">in app</span>' : ""}</span>
        <span class="metric-val">${r.rmse.toFixed(4)}</span>
        <span class="metric-val">${r.mae.toFixed(4)}</span>
      </div>`;
  }
  wrap.innerHTML = html;
}

/* ------------------------------- users ----------------------------------- */
async function loadUsers() {
  const [{ users }, newUser] = await Promise.all([
    getJSON(`${API}/users/`),
    getJSON(`${API}/new-user/`),
  ]);
  const sel = document.getElementById("userSelect");
  sel.innerHTML =
    `<option value="${newUser.user_id}">✦ New visitor</option>` +
    users.map(u => `<option value="${u.user_id}">${u.user_name} (#${u.user_id})</option>`).join("");
  sel.addEventListener("change", () => switchUser(parseInt(sel.value, 10)));
  state.userId = newUser.user_id;
}

async function switchUser(userId) {
  state.userId = userId;
  const { ratings } = await getJSON(`${API}/ratings/?user_id=${userId}`);
  state.ratings = {};
  for (const [aid, r] of Object.entries(ratings)) state.ratings[parseInt(aid, 10)] = r;
  renderGrid();
  await refreshRecommendations();
}

/* --------------------------- attraction grid ----------------------------- */
async function loadAttractions() {
  const { attractions } = await getJSON(`${API}/attractions/`);
  state.attractions = attractions;
  renderGrid();
}

function renderGrid() {
  const grid = document.getElementById("grid");
  grid.innerHTML = "";
  for (const att of state.attractions) {
    const rated = state.ratings[att.attraction_id];
    const card = document.createElement("div");
    card.className = "card";
    card.innerHTML = `
      <div class="thumb" style="background-image:url('${att.image_url}')"></div>
      <div class="card-body">
        <span class="loc">${att.city} · ${att.region}</span>
        <h3>${att.name}</h3>
        <p class="desc">${att.description}</p>
        <div class="stars" data-id="${att.attraction_id}">
          ${[1,2,3,4,5].map(n => `<button class="star ${rated && n <= rated ? "on" : ""}" data-n="${n}">★</button>`).join("")}
        </div>
        <span class="rated-note">${rated ? `You rated this ${rated}/5` : ""}</span>
      </div>`;

    // chip on the image
    const t = card.querySelector(".thumb");
    t.insertAdjacentHTML("beforeend", `<span class="chip">${att.category}</span>`);
    // graceful image fallback
    const img = new Image();
    img.src = att.image_url;
    img.onerror = () => {
      t.classList.add("fallback");
      t.style.backgroundImage = "none";
      t.insertAdjacentHTML("afterbegin", `<span>${att.name}</span>`);
    };

    // star interactions
    const starsEl = card.querySelector(".stars");
    starsEl.querySelectorAll(".star").forEach(btn => {
      const n = parseInt(btn.dataset.n, 10);
      btn.addEventListener("mouseenter", () => paintStars(starsEl, n));
      btn.addEventListener("mouseleave", () => paintStars(starsEl, state.ratings[att.attraction_id] || 0));
      btn.addEventListener("click", () => rate(att, n, card));
    });

    grid.appendChild(card);
  }
}

function paintStars(starsEl, value) {
  starsEl.querySelectorAll(".star").forEach(b => {
    b.classList.toggle("on", parseInt(b.dataset.n, 10) <= value);
  });
}

/* ------------------------------- rating ---------------------------------- */
async function rate(att, value, card) {
  try {
    const resp = await postJSON(`${API}/rate/`, {
      user_id: state.userId,
      attraction_id: att.attraction_id,
      rating: value,
      n: TOP_N,
    });
    state.ratings[att.attraction_id] = value;
    card.querySelector(".rated-note").textContent = `You rated this ${value}/5`;
    paintStars(card.querySelector(".stars"), value);
    renderRecommendations(resp.recommendations);
    document.getElementById("recsHint").textContent =
      "Updated from your latest rating.";
    toast(`Saved ${value}★ for ${att.name} — recommendations refreshed`);
  } catch (e) {
    toast("Could not save your rating.");
    console.error(e);
  }
}

/* --------------------------- recommendations ----------------------------- */
async function refreshRecommendations() {
  const { recommendations } = await getJSON(
    `${API}/recommendations/?user_id=${state.userId}&n=${TOP_N}`);
  renderRecommendations(recommendations);
}

function renderRecommendations(recs) {
  const track = document.getElementById("recsTrack");
  track.innerHTML = "";
  recs.forEach((r, idx) => {
    const card = document.createElement("article");
    card.className = "rec-card";
    card.style.animationDelay = `${idx * 60}ms`;
    card.innerHTML = `
      <div style="position:relative">
        <span class="rec-rank">${idx + 1}</span>
        <div class="thumb" style="height:130px;background-image:url('${r.image_url}')"></div>
      </div>
      <div class="rec-body">
        <h3>${r.name}</h3>
        <span class="rec-loc">${r.city} · ${r.region}</span>
        <span class="rec-score">predicted ${r.score}/5</span>
        <span class="rec-reason">${r.reason}</span>
      </div>`;
    const t = card.querySelector(".thumb");
    const img = new Image();
    img.src = r.image_url;
    img.onerror = () => {
      t.classList.add("fallback");
      t.style.backgroundImage = "none";
      t.insertAdjacentHTML("afterbegin", `<span>${r.name}</span>`);
    };
    track.appendChild(card);
  });
}

/* ------------------------------- bootstrap ------------------------------- */
async function init() {
  try {
    await loadModelInfo();
    await loadUsers();
    await loadAttractions();
    await refreshRecommendations();
  } catch (e) {
    toast("Failed to start. Is the server running?");
    console.error(e);
  }
}

document.addEventListener("DOMContentLoaded", init);
