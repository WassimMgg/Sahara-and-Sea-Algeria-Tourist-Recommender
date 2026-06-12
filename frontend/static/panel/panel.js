/* ==========================================================================
   Sahara & Sea — control panel behaviour
   Count-up KPIs, toast lifecycle, delete-confirm modal, inline star editing,
   live image preview, mobile sidebar.
   ========================================================================== */
(() => {
  "use strict";

  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

  /* ------------------------------ CSRF ------------------------------ */
  function csrfToken() {
    const m = document.cookie.match(/(?:^|;\s*)csrftoken=([^;]+)/);
    if (m) return m[1];
    const input = $("input[name=csrfmiddlewaretoken]");
    return input ? input.value : "";
  }

  /* ----------------------------- toasts ----------------------------- */
  function toast(text, kind = "success") {
    let wrap = $("#toasts");
    if (!wrap) {
      wrap = document.createElement("div");
      wrap.id = "toasts";
      wrap.className = "toasts";
      document.body.appendChild(wrap);
    }
    const el = document.createElement("div");
    el.className = `toast ${kind}`;
    el.textContent = text;
    wrap.appendChild(el);
    dismissLater(el);
  }

  function dismissLater(el) {
    setTimeout(() => {
      el.classList.add("hide");
      el.addEventListener("animationend", () => el.remove(), { once: true });
    }, 4200);
  }
  $$(".toast").forEach(dismissLater);

  /* --------------------------- count-up KPIs --------------------------- */
  $$("[data-countup]").forEach((el) => {
    const target = parseInt(el.textContent.replace(/\D/g, ""), 10);
    if (!Number.isFinite(target) || target === 0) return;
    const start = performance.now();
    const dur = Math.min(1100, 400 + target);
    const tick = (now) => {
      const p = Math.min(1, (now - start) / dur);
      const eased = 1 - Math.pow(1 - p, 3);
      el.textContent = Math.round(target * eased).toLocaleString();
      if (p < 1) requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  });

  /* ----------------------- delete-confirm modal ----------------------- */
  const modal = $("#confirm-modal");
  let pendingForm = null;

  document.addEventListener("submit", (e) => {
    const form = e.target.closest("form[data-confirm]");
    if (!form || form.dataset.confirmed) return;
    e.preventDefault();
    pendingForm = form;
    $("#confirm-text").textContent = form.dataset.confirm;
    modal.hidden = false;
    $("#confirm-ok").focus();
  });

  if (modal) {
    const close = () => { modal.hidden = true; pendingForm = null; };
    $("#confirm-cancel").addEventListener("click", close);
    modal.addEventListener("click", (e) => { if (e.target === modal) close(); });
    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape" && !modal.hidden) close();
    });
    $("#confirm-ok").addEventListener("click", () => {
      if (!pendingForm) return;
      pendingForm.dataset.confirmed = "1";
      pendingForm.requestSubmit();
      modal.hidden = true;
    });
  }

  /* ----------------------- inline star editing ----------------------- */
  function paintStars(holder, value) {
    holder.innerHTML = "";
    for (let s = 1; s <= 5; s++) {
      const star = document.createElement("i");
      star.textContent = "★";
      if (value >= s) star.className = "on";
      else if (value >= s - 0.5) star.className = "half";
      star.dataset.value = s;
      holder.appendChild(star);
    }
  }

  $$(".stars[data-rating-id]").forEach((holder) => {
    paintStars(holder, parseFloat(holder.dataset.value));

    holder.addEventListener("click", async (e) => {
      const star = e.target.closest("i");
      if (!star) return;
      const value = parseInt(star.dataset.value, 10);
      const old = parseFloat(holder.dataset.value);
      if (value === old) return;

      holder.classList.add("saving");
      try {
        const resp = await fetch(`/admin/ratings/${holder.dataset.ratingId}/inline/`, {
          method: "POST",
          headers: { "Content-Type": "application/json", "X-CSRFToken": csrfToken() },
          body: JSON.stringify({ rating: value }),
        });
        if (!resp.ok) throw new Error();
        holder.dataset.value = value;
        paintStars(holder, value);
        toast(`Rating updated to ${value} ★`);
      } catch {
        paintStars(holder, old);
        toast("Could not update the rating.", "error");
      } finally {
        holder.classList.remove("saving");
      }
    });
  });

  /* ------------------------ live image preview ------------------------ */
  const imgInput = $("#id_image_url");
  const preview = $("#img-preview");
  if (imgInput && preview) {
    const emptyNote = $("#img-preview-empty");
    let timer;
    imgInput.addEventListener("input", () => {
      clearTimeout(timer);
      timer = setTimeout(() => {
        const url = imgInput.value.trim();
        if (!url) {
          preview.style.display = "none";
          if (emptyNote) emptyNote.style.display = "";
          return;
        }
        preview.src = url;
        preview.style.display = "";
        if (emptyNote) emptyNote.style.display = "none";
      }, 350);
    });
    preview.addEventListener("error", () => { preview.style.display = "none"; });
  }

  /* -------------------------- mobile sidebar -------------------------- */
  const burger = $("#burger");
  const sidebar = $("#sidebar");
  const scrim = $("#scrim");
  if (burger && sidebar) {
    const toggle = (open) => {
      sidebar.classList.toggle("open", open);
      document.body.classList.toggle("nav-open", open);
    };
    burger.addEventListener("click", () => toggle(!sidebar.classList.contains("open")));
    if (scrim) scrim.addEventListener("click", () => toggle(false));
  }
})();
