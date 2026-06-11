# Sahara & Sea — Tourist Attraction Recommender for Algeria

A full-stack web application that recommends Algerian tourist attractions
(Roman ruins, Saharan landscapes, casbahs, coastlines…) using **collaborative
filtering implemented from scratch with NumPy** — no recommender libraries.
Users sign up, rate places, and watch their recommendations adapt instantly.
Administrators get a custom dashboard and can **switch the serving algorithm
live** from the admin panel.

Built as a university project for the *Recommender Systems / Data Exploration*
course (Białystok University of Technology).

---

## 1. Features

**Public site**
- Editorial-style multi-page UI (Home, Places, Search, Used model, About)
- Account signup / login / logout (Django auth, hashed passwords, CSRF)
- Rate any attraction 1–5 ★ — recommendations refresh immediately
- Cold-start handling: new users see non-personalized crowd favourites
- Explanations on every recommendation (“Matches your taste for Roman Ruins”)
- Live search + filter by place type; fully responsive; reduced-motion friendly

**Recommendation engine (all from scratch)**
- 7 algorithms across the 4 course families (see §4)
- 5-fold cross-validated evaluation: RMSE, MAE, Precision@5, Recall@5
- All models trained and bundled; the **active one is switchable at runtime**

**Admin panel** (`/admin/`)
- Custom-branded UI: fixed sidebar, light theme, KPI cards, CSS charts
- Animated dashboard (count-up stats, growing bars) with quick actions
- **Recommender engine page**: compare every algorithm's metrics and switch
  the one being served with one click — no restart, no redeploy
- Attractions with image previews, per-attraction rating breakdown chart,
  inline ratings, quick-edit fields, CSV export on every model

---

## 2. Technology stack

| Layer      | Technology | Used for |
|------------|------------|----------|
| Language   | Python 3.10+ | everything backend & ML |
| Backend    | Django 5/6 | routing, ORM, auth, admin, templates |
| Database   | SQLite (dev) | attractions, visitors, ratings, settings |
| ML / data  | NumPy, pandas | matrices, similarities, SGD, cleaning |
| Frontend   | HTML + CSS + vanilla JS | no framework, no build step |
| Fonts      | Fraunces + Outfit (Google Fonts) | typography |
| Images     | Wikimedia Commons (`Special:FilePath`) | attraction photos |

No external CSS/JS frameworks, no recommender libraries — by design, so every
moving part is inspectable for the course.

---

## 3. Project structure

```
tourist-recommender/
├── data/
│   ├── attractions.csv        # 15 attractions (id, name, city, region, …)
│   └── ratings_raw.csv        # ~635 messy historical ratings
├── ml/
│   ├── clean_data.py          # raw CSV -> ratings_clean.csv
│   ├── recommender.py         # ALL algorithms, from scratch (registry: ALL_MODELS)
│   ├── train.py               # evaluation + trains & bundles every model
│   ├── ratings_clean.csv      # generated
│   ├── models.pkl             # generated — every trained model + default key
│   └── metrics.json           # generated — CV metrics for each algorithm
├── api/                       # the Django app
│   ├── models.py              # Attraction, Visitor, Rating, RecommenderSetting
│   ├── services.py            # loads bundle, switching API, recommend_for()
│   ├── views.py               # page views + JSON API
│   ├── admin.py               # custom admin site, dashboard, recommender page
│   ├── management/commands/   # seed_db, create_admin
│   └── migrations/
├── recommender_project/       # Django project (settings, urls, wsgi/asgi)
├── frontend/
│   ├── templates/             # base + pages + admin/ overrides
│   └── static/                # css/, js/, admin_custom/
├── manage.py
├── requirements.txt
└── README.md                  # you are here
```

---

## 4. The recommendation algorithms

All implemented in `ml/recommender.py`, mapped to the course sections:

| # | Course family | Implementation (registry key) |
|---|---------------|-------------------------------|
| 1 | **Non-personalized — mean calculations** | `pop_mean` damped item means · `baseline` global mean + user/item bias |
| 2 | **Similarity calculations** | cosine (mean-centered/adjusted) and Pearson (co-rated) — pluggable into both CF families |
| 3 | **User-user CF** | `user_cf_cosine`, `user_cf_pearson` |
| 3 | **Item-item CF** | `item_cf_cosine`, `item_cf_pearson` |
| 4 | **Evaluation** | 5-fold CV in `ml/train.py` (RMSE, MAE, Precision@5, Recall@5) |
| + | Bonus | `mf` matrix factorization (SGD latent factors, ridge fold-in for new users) |

Every model shares one interface, so they are interchangeable at runtime:

```python
model.fit(df)                                  # df: user_id, attraction_id, rating
model.predict(user_id, item_id)                # known training user
model.predict_from_ratings(item_id, ratings)   # NEW user: {item_id: rating}
model.recommend(ratings, top_n=5)              # -> [(item_id, score), …]
```

### Current evaluation results (5-fold CV, 576 ratings, 50 users × 15 places)

| Algorithm | RMSE ↓ | MAE ↓ | P@5 ↑ | R@5 ↑ |
|---|---|---|---|---|
| Mean ratings (non-pers.) | 0.903 | 0.732 | 0.381 | 0.816 |
| Bias baseline (non-pers.) | 0.908 | 0.730 | 0.381 | 0.816 |
| **User-user CF · cosine** ← default | **0.922** | 0.744 | **0.389** | **0.840** |
| User-user CF · Pearson | 0.924 | 0.749 | 0.390 | 0.840 |
| Item-item CF · Pearson | 0.969 | 0.777 | 0.382 | 0.817 |
| Matrix factorization | 0.993 | 0.810 | 0.379 | 0.813 |
| Item-item CF · cosine | 1.051 | 0.845 | 0.386 | 0.831 |

**Why is the default a personalized model when the plain mean has the lowest
RMSE?** Because the non-personalized models rank attractions identically for
every user — good *prediction* numbers, zero *personalization*. The default is
therefore the most efficient **personalized** algorithm (lowest RMSE among
them: user-user cosine), which also happens to win on both ranking metrics.
This trade-off is exactly the point of the evaluation section of the course.

### Switching the algorithm

Three equivalent ways:

1. **Admin UI (recommended)** — `/admin/` → *Recommender* in the sidebar →
   pick a card → **Apply selected algorithm**. Takes effect instantly.
2. **Python API** —
   ```python
   from api import services
   services.set_active_algorithm("item_cf_pearson")   # any registry key
   services.get_active_key()                          # check what's serving
   services.available_algorithms()                    # metadata + metrics
   ```
3. **Database** — the choice is one row in the `RecommenderSetting` table
   (`key="active_algorithm"`); deleting it falls back to the trained default.

After retraining (`python ml/train.py`) click **Reload models from disk** on
the same admin page (or call `services.reload_models()`).

---

## 5. Data pipeline

1. `data/ratings_raw.csv` is intentionally messy (≈23 spellings of countries,
   mixed date formats, ratings like `"5 stars"`, `4,5`, blanks, out-of-range
   values, duplicate user/place pairs).
2. `ml/clean_data.py` normalizes countries and genders, parses 8 date formats
   to ISO, coerces ratings to floats in [1, 5], averages duplicates →
   **576 valid ratings** out of 635 rows → `ml/ratings_clean.csv`.
3. `ml/train.py` evaluates all 7 algorithms (5-fold CV), retrains each on the
   full data, and saves `models.pkl` + `metrics.json`.
4. `manage.py seed_db` loads attractions + historical ratings into SQLite.
5. The web app reads everything from the database; ratings made in the app are
   stored with the user's account and folded into the model live (no retrain
   needed for personalization — fold-in uses `predict_from_ratings`).

---

## 6. Installation & setup

Prerequisites: **Python 3.10+** and pip. (SQLite ships with Python.)

```bash
# 0. unzip / clone, then enter the folder
cd tourist-recommender

# 1. (recommended) virtual environment
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 2. dependencies
pip install -r requirements.txt

# 3. data pipeline: clean the ratings, evaluate & train ALL models
cd ml
python clean_data.py
python train.py                   # prints the comparison table
cd ..

# 4. database: create tables and load the data
python manage.py migrate
python manage.py seed_db

# 5. an admin account for the panel  (admin / admin12345 — change it!)
python manage.py create_admin

# 6. run
python manage.py runserver
```

Open **http://127.0.0.1:8000/** — sign up, rate a few places on *Places*, and
watch *Picked for you* change. The admin panel is at
**http://127.0.0.1:8000/admin/**.

> `create_admin` accepts `--username/--password` flags or `ADMIN_USER` /
> `ADMIN_PASSWORD` environment variables. The standard
> `python manage.py createsuperuser` works too.

---

## 7. Using the app

**Visitor flow** — browse *Places*, use the type chips or *Search*; sign up;
click the stars on any card; the *Picked for you* strip and the home page
update on the next load. *Used model* explains, in plain language, every
algorithm with its cross-validated metrics and shows which one is live.

**Admin flow** — log in at `/admin/`:
- *Dashboard*: animated KPI cards, rating-distribution and avg-by-type charts,
  top/most-rated, recent app ratings, quick actions.
- *Recommender*: the control room — compare all algorithms and switch the live
  one (see §4).
- *Attractions / Ratings / Visitors*: full CRUD with image previews, rating
  breakdown per attraction, inline editing and **Export selected to CSV**.

---

## 8. HTTP API (used by the frontend JS)

| Endpoint | Method | Auth | Purpose |
|---|---|---|---|
| `/api/attractions/?q=&type=` | GET | — | list/search attractions (+my rating if logged in) |
| `/api/recommendations/` | GET | optional | top-5 for the current user (popular if anonymous) |
| `/api/my-ratings/` | GET | required | the logged-in user's ratings |
| `/api/rate/` | POST JSON `{attraction_id, rating}` | required | create/update a rating, returns fresh recommendations |

All POSTs require Django's CSRF token (the frontend handles this).

---

## 9. Management commands

| Command | What it does |
|---|---|
| `python manage.py seed_db` | wipe + reload attractions and historical ratings from `data/` (keeps app-user ratings) |
| `python manage.py create_admin` | create/update a superuser non-interactively |
| `python ml/clean_data.py` | rebuild `ratings_clean.csv` from the raw file |
| `python ml/train.py` | re-evaluate, retrain and re-bundle all models |

**Retraining workflow:** edit/extend data → `clean_data.py` → `train.py` →
admin *Recommender* page → *Reload models from disk*.

---

## 10. Troubleshooting

| Symptom | Fix |
|---|---|
| `no such table` errors | run `python manage.py migrate` then `seed_db` |
| Images don't load | the URLs point at Wikimedia `Special:FilePath`; re-run `seed_db` if you changed `attractions.csv` (the app reads the DB, not the CSV) |
| Admin looks unstyled | hard-refresh (Ctrl/Cmd-Shift-R) to reload `admin_custom/admin.css` |
| `FileNotFoundError: models.pkl` | run `python ml/train.py` first |
| Switched algorithm “didn't change” the list | with 15 items and dense crowd favourites, several algorithms agree on the top picks — check `/model/` to confirm which is active |

---

## 11. Production notes (beyond the course)

This is a teaching project configured for local use. Before any real
deployment: set `DEBUG = False`, move `SECRET_KEY` to an environment variable,
restrict `ALLOWED_HOSTS`, switch SQLite → PostgreSQL, serve static files via
`collectstatic` + WhiteNoise/Nginx, and change the default admin password.

## 12. Credits

Data: hand-curated attraction list; photos via Wikimedia Commons
(`Special:FilePath`, CC-licensed works by their respective photographers).
Course: Recommender Systems, Białystok University of Technology.
