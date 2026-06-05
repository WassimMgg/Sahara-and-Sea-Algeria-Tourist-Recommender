/* home.js — home-page enhancements
   - count-up animation for the stat ribbon
   - subtle parallax on the hero image
   - graceful fallback for hero / collage / category images
   (reveal animations + rec-card fallbacks are handled by common.js)        */

(function () {
  const reduce = window.matchMedia &&
    window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  /* ---- count-up stats ---- */
  function countUp(el) {
    const target = parseInt(el.dataset.count || "0", 10);
    if (reduce || !target) { el.textContent = target; return; }
    const dur = 1100, t0 = performance.now();
    function tick(now) {
      const p = Math.min(1, (now - t0) / dur);
      const eased = 1 - Math.pow(1 - p, 3);          // easeOutCubic
      el.textContent = Math.round(target * eased).toLocaleString();
      if (p < 1) requestAnimationFrame(tick);
    }
    requestAnimationFrame(tick);
  }

  const ribbon = document.querySelector(".stat-ribbon");
  if (ribbon) {
    const nums = ribbon.querySelectorAll(".stat-n");
    if ("IntersectionObserver" in window) {
      const obs = new IntersectionObserver((entries) => {
        entries.forEach((e) => {
          if (e.isIntersecting) { nums.forEach(countUp); obs.disconnect(); }
        });
      }, { threshold: 0.4 });
      obs.observe(ribbon);
    } else {
      nums.forEach(countUp);
    }
  }

  /* ---- hero parallax (background shifts slower than scroll) ---- */
  const media = document.querySelector(".hero-media");
  if (media && !reduce) {
    let ticking = false;
    window.addEventListener("scroll", () => {
      if (ticking) return;
      ticking = true;
      requestAnimationFrame(() => {
        const offset = Math.min(60, window.scrollY * 0.18);
        media.style.backgroundPositionY = `calc(50% + ${offset}px)`;
        ticking = false;
      });
    }, { passive: true });
  }

  /* ---- image fallbacks for background-image elements ---- */
  function bgFallback(el, isTile) {
    const m = el.style.backgroundImage.match(/url\(["']?(.*?)["']?\)/);
    if (!m || !m[1]) return;
    const img = new Image();
    img.src = m[1];
    img.onerror = () => {
      el.style.backgroundImage = "none";
      el.style.background = "linear-gradient(135deg, var(--terracotta), var(--teal))";
      if (isTile && el.dataset.name) {
        el.classList.add("cat-fallback");
        el.textContent = el.dataset.name;
      }
    };
  }
  document.querySelectorAll(".hero-media, .polaroid").forEach((el) => bgFallback(el, false));
  document.querySelectorAll(".cat-media").forEach((el) => bgFallback(el, true));
})();
