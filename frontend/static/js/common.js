/* ============================================================
   common.js — shared helpers used by every page
   (structured as small reusable functions; maps cleanly to
   Angular services/components if ported later)
   ============================================================ */

/* ----------------------------- fetch ----------------------------- */
function getCookie(name) {
  const m = document.cookie.match("(^|;)\\s*" + name + "\\s*=\\s*([^;]+)");
  return m ? m.pop() : "";
}
async function getJSON(url) {
  const res = await fetch(url, { headers: { "X-Requested-With": "fetch" } });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}
async function postJSON(url, body) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json", "X-CSRFToken": getCookie("csrftoken") },
    body: JSON.stringify(body),
  });
  return res; // caller inspects status
}

/* ----------------------------- toast ----------------------------- */
let _toastTimer;
function toast(msg) {
  const el = document.getElementById("toast");
  if (!el) return;
  el.textContent = msg;
  el.classList.add("show");
  clearTimeout(_toastTimer);
  _toastTimer = setTimeout(() => el.classList.remove("show"), 2400);
}

/* ------------------------ image fallback ------------------------- */
function withImageFallback(thumbEl, name) {
  const url = thumbEl.style.backgroundImage.slice(5, -2);
  if (!url) return;
  const img = new Image();
  img.src = url;
  img.onerror = () => {
    thumbEl.classList.add("fallback");
    thumbEl.style.backgroundImage = "none";
    thumbEl.insertAdjacentHTML("afterbegin", `<span>${name}</span>`);
  };
}

/* --------------------- scroll reveal animation ------------------- */
function initReveal() {
  const els = document.querySelectorAll(".reveal");
  if (!("IntersectionObserver" in window)) {
    els.forEach(e => e.classList.add("in"));
    return;
  }
  const obs = new IntersectionObserver((entries) => {
    entries.forEach(e => { if (e.isIntersecting) { e.target.classList.add("in"); obs.unobserve(e.target); } });
  }, { threshold: 0.12 });
  els.forEach(e => obs.observe(e));
}

/* --------------------------- navbar ------------------------------ */
function initNavbar() {
  const toggle = document.getElementById("navToggle");
  const links = document.getElementById("navLinks");
  if (!toggle || !links) return;
  toggle.addEventListener("click", () => {
    toggle.classList.toggle("open");
    links.classList.toggle("open");
  });
}

/* --------------------- attraction card builder ------------------- */
// opts: { rateable: bool, rated: number|undefined }
function buildAttractionCard(att, opts = {}) {
  const card = document.createElement("div");
  card.className = "card reveal";
  const rated = opts.rated;
  const stars = opts.rateable
    ? `<div class="stars" data-id="${att.id}">
         ${[1,2,3,4,5].map(n => `<button class="star ${rated && n <= rated ? "on" : ""}" data-n="${n}">★</button>`).join("")}
       </div>
       <span class="rated-note">${rated ? `You rated this ${rated}/5` : ""}</span>`
    : "";
  card.innerHTML = `
    <div class="thumb" style="background-image:url('${att.image_url}')">
      <span class="chip">${att.place_type}</span>
    </div>
    <div class="card-body">
      <span class="loc">${att.city} · ${att.region}</span>
      <h3>${att.name}</h3>
      <p class="desc">${att.description}</p>
      ${stars}
    </div>`;
  withImageFallback(card.querySelector(".thumb"), att.name);

  if (opts.rateable) {
    const starsEl = card.querySelector(".stars");
    starsEl.querySelectorAll(".star").forEach(btn => {
      const n = parseInt(btn.dataset.n, 10);
      btn.addEventListener("mouseenter", () => paintStars(starsEl, n));
      btn.addEventListener("mouseleave", () => paintStars(starsEl, parseInt(starsEl.dataset.current || 0, 10)));
      btn.addEventListener("click", () => rateAttraction(att, n, card, starsEl, btn));
    });
    if (rated) starsEl.dataset.current = rated;
  }
  return card;
}

function paintStars(starsEl, value) {
  starsEl.querySelectorAll(".star").forEach(b =>
    b.classList.toggle("on", parseInt(b.dataset.n, 10) <= value));
}

async function rateAttraction(att, value, card, starsEl, btn) {
  if (!window.IS_AUTHENTICATED) {
    toast("Please log in to rate places.");
    setTimeout(() => (window.location = window.LOGIN_URL), 900);
    return;
  }
  const res = await postJSON("/api/rate/", { attraction_id: att.id, rating: value, n: 6 });
  if (res.status === 401) { window.location = window.LOGIN_URL; return; }
  if (!res.ok) { toast("Could not save your rating."); return; }
  const data = await res.json();
  starsEl.dataset.current = value;
  paintStars(starsEl, value);
  btn.classList.remove("burst"); void btn.offsetWidth; btn.classList.add("burst");
  card.querySelector(".rated-note").textContent = `You rated this ${value}/5`;
  toast(`Saved ${value}★ for ${att.name} — recommendations updated`);
  if (typeof renderRecommendations === "function") renderRecommendations(data.recommendations);
}

/* ---------------------- skeleton card builders -------------------- */
function renderSkeletons(n) {
  return Array.from({ length: n }, () => `
    <div class="card skeleton-card">
      <div class="thumb skeleton-thumb"></div>
      <div class="card-body">
        <div class="skeleton-line w50"></div>
        <div class="skeleton-line w90 tall"></div>
        <div class="skeleton-line w70"></div>
        <div class="skeleton-line w30"></div>
      </div>
    </div>`).join("");
}

function renderRecSkeletons(n) {
  return Array.from({ length: n }, () => `
    <article class="rec-card">
      <div class="rec-thumb skeleton-rec-thumb"></div>
      <div class="rec-body">
        <div class="skeleton-rec-line w90 tall"></div>
        <div class="skeleton-rec-line w60"></div>
        <div class="skeleton-rec-line w40"></div>
      </div>
    </article>`).join("");
}

/* ------------------- recommendation card builder ----------------- */
function buildRecCard(r, idx) {
  const el = document.createElement("article");
  el.className = "rec-card reveal";
  el.style.setProperty("--d", idx);
  el.innerHTML = `
    <div class="rec-thumb" style="background-image:url('${r.image_url}')">
      <span class="rec-rank">${idx + 1}</span>
      <span class="chip">${r.place_type}</span>
    </div>
    <div class="rec-body">
      <h3>${r.name}</h3>
      <span class="muted">${r.city} · ${r.region}</span>
      <span class="rec-score">predicted ${r.score}/5</span>
      <span class="rec-reason">${r.reason}</span>
    </div>`;
  withImageFallback(el.querySelector(".rec-thumb"), r.name);
  requestAnimationFrame(() => el.classList.add("in"));
  return el;
}

/* ---------------------------- bootstrap -------------------------- */
document.addEventListener("DOMContentLoaded", () => {
  initNavbar();
  initReveal();
  // give server-rendered rec thumbs their fallback behaviour
  document.querySelectorAll(".rec-thumb[data-name]").forEach(t =>
    withImageFallback(t, t.dataset.name));
});
