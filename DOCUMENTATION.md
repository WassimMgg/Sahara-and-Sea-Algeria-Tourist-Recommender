# Code Documentation — Sahara &amp; Sea

This document explains how the project is built: the architecture, every file,
the data flow, the database schema, and the API. It is meant as a reference you
can read top-to-bottom or jump around in.

---

## 1. Big picture

The project has three layers:

```
   CSV files                Python (ml/)                Django (backend + DB)             Browser
 ┌───────────┐   clean   ┌──────────────┐  train   ┌───────────────────────┐  HTTP   ┌──────────────┐
 │ raw data  │ ────────► │ ratings_clean│ ───────► │ model.pkl             │ ──────► │ HTML/CSS/JS  │
 │ (messy)   │           │ + metrics    │          │ + SQLite database     │ ◄────── │ pages        │
 └───────────┘           └──────────────┘          └───────────────────────┘  JSON   └──────────────┘
                                  │  seed_db
                                  └──────────────► database tables (Attraction, Visitor, Rating)
```

* **CSV → clean → train** happens offline, in the `ml/` folder. The CSVs are used
  ONLY here: to clean the data, train and compare models, and save the winner.
* The **database** is seeded once from the cleaned CSV (`seed_db`). After that the
  running app reads from the database (fast), not from CSV.
* The **Django backend** loads `model.pkl`, reads ratings from the database, and
  serves both HTML pages and a small JSON API.
* The **frontend** is server-rendered HTML with vanilla JavaScript that calls the
  JSON API for filtering, searching and rating.

---

## 2. Folder map

```
tourist-recommender/
├── data/
│   ├── attractions.csv          # catalogue source (15 places)
│   └── ratings_raw.csv          # messy ratings source
├── ml/                          # offline machine-learning code (pure Python)
│   ├── clean_data.py            # raw CSV -> ratings_clean.csv
│   ├── recommender.py           # 4 algorithms (Baseline, ItemCF, UserCF, MF)
│   ├── train.py                 # cross-validates, selects, saves model.pkl
│   ├── ratings_clean.csv        # (generated)
│   ├── model.pkl                # (generated) the served model
│   └── metrics.json             # (generated) comparison numbers
├── recommender_project/         # Django project config
│   ├── settings.py              # apps, middleware, auth, paths
│   ├── urls.py                  # all routes (pages + auth + api)
│   ├── wsgi.py / asgi.py        # servers entry points
├── api/                         # the Django app
│   ├── models.py                # database tables
│   ├── services.py              # model loading + recommendation logic
│   ├── views.py                 # page, auth and API views
│   ├── migrations/              # database schema migrations
│   └── management/commands/
│       └── seed_db.py           # loads CSV data into the database
├── frontend/
│   ├── templates/               # base.html + one template per page
│   └── static/
│       ├── css/styles.css       # the whole stylesheet
│       └── js/                  # common.js + home.js + places.js + search.js
├── manage.py
├── requirements.txt
├── README.md                    # how to run + project overview
└── DOCUMENTATION.md             # this file
```

---

## 3. The machine-learning layer (`ml/`)

### 3.1 `clean_data.py`
Reads `data/ratings_raw.csv` and writes `ml/ratings_clean.csv`. Each function fixes
one category of mess:

| Function | Fixes |
|---|---|
| `normalize_country` | `DZ`, `Algerie`, `ALGERIA`, `Algérie` … → `Algeria` (accents stripped, lookup table) |
| `normalize_gender` | `F`/`Female`/`female` → `Female`, etc. |
| `parse_rating` | `"5 stars"`, `"4,5"`, `"5.0"` → number; out-of-range (`0,6,7,10`) → invalid |
| `parse_date` | tries ~8 date formats → ISO `YYYY-MM-DD` |
| duplicates | same `(user, attraction)` rated twice → averaged |

Result: **635 raw rows → 576 clean ratings**.

### 3.2 `recommender.py`
Four model classes, each with the same interface:

```python
model.fit(df)                      # learn from a DataFrame of ratings
model.predict(user_id, item_id)    # predict one rating (used during cross-validation)
model.recommend(ratings_dict, n)   # rank unseen items for a user given their ratings
```

The `recommend(ratings_dict, ...)` method is what makes live updates possible: it
takes a plain `{attraction_id: rating}` dictionary, so a brand-new user's ratings can
be scored immediately **without retraining**.

| Class | Idea |
|---|---|
| `BaselineModel` | global mean + per-user bias + per-item bias |
| `ItemBasedCF` | cosine similarity between attractions (mean-centered) |
| `UserBasedCF` | cosine similarity between users (mean-centered) |
| `MatrixFactorization` | latent factors learned by gradient descent (SVD-style); folds in a new user with a small ridge-regression step |

### 3.3 `train.py`
Runs **5-fold cross-validation** on all four models, printing RMSE and MAE. It then:
1. reports the lowest-RMSE model overall, and
2. selects the lowest-RMSE **personalised** model (excludes the baseline, which ranks
   everyone identically) to actually power the app,
3. retrains that model on all data and saves it to `model.pkl`, plus the comparison
   numbers to `metrics.json`.

> Key idea: RMSE measures *rating prediction*, not *personalised ranking*. The trivial
> baseline can win on RMSE yet be useless as a recommender, so the served model is the
> best one that genuinely personalises.

---

## 4. The database (`api/models.py`)

Three tables:

```
Attraction                Visitor                    Rating
----------                -------                    ------
id (PK, = CSV id)         id (PK, = CSV user id)     id
name                      name                       visitor   -> Visitor   (nullable)
city                      age                        account   -> auth User (nullable)
region                    gender                     attraction-> Attraction
category                  home_country               rating (float)
place_type (filterable)                              visit_date
description
image_url
```

* **Attraction** — the catalogue. `place_type` is derived from `category`
  (`"Roman Ruins / UNESCO"` → `"Roman Ruins"`) and powers the filter chips.
* **Visitor** — the historical "friends" imported from the cleaned CSV. Their ratings
  give the "Popular with visitors" cold-start ranking.
* **Rating** — one row per rating. It links to **either** a `Visitor` (historical data)
  **or** an `account` (a registered user rating inside the app). A `UniqueConstraint`
  makes sure an account rates a given attraction at most once (re-rating updates it).

### Seeding (`api/management/commands/seed_db.py`)
`python manage.py seed_db` wipes the previously-seeded historical data and reloads it
from `attractions.csv` and `ratings_clean.csv` using fast `bulk_create`. Registered
account ratings are preserved.

---

## 5. The service layer (`api/services.py`)

This is where the model meets the database. Important functions:

| Function | What it does |
|---|---|
| `_load_model()` | loads `model.pkl` once and caches it (thread-safe) |
| `popularity_ranking()` | average rating per attraction from Visitor ratings (cold start) |
| `account_ratings(user)` | a logged-in user's `{attraction_id: rating}` from the DB |
| `recommend_for(user, n)` | the main entry point — see below |
| `place_types()` | the distinct `place_type` values for the filter |

`recommend_for(user, n)` logic:
1. If the user has no ratings → return the most popular attractions (**cold start**).
2. Otherwise → call `model.recommend(their_ratings)` and attach a friendly reason
   (e.g. *"Matches your taste for Roman Ruins"* when the recommended place's type equals
   the user's favourite type).

---

## 6. Views and routes

### Page views (return HTML)
| Route | View | Template |
|---|---|---|
| `/` | `home` | `home.html` (hero + recommendations) |
| `/places/` | `places` | `places.html` (rate + filter + live recs) |
| `/search/` | `search` | `search.html` (text search + filter) |
| `/model/` | `used_model` | `model.html` (algorithm comparison) |
| `/about/` | `about` | `about.html` |
| `/login/` `/signup/` `/logout/` | auth views | `login.html` / `signup.html` |

`home` and `places` use `@ensure_csrf_cookie` so the browser holds a CSRF token before
the JavaScript makes a `POST /api/rate/`.

### API views (return JSON)
| Method | Route | Purpose |
|---|---|---|
| GET | `/api/attractions/?q=&type=` | catalogue, optionally filtered by search text and type |
| GET | `/api/recommendations/?n=` | current recommendations for the logged-in user |
| GET | `/api/my-ratings/` | the logged-in user's existing ratings (to pre-fill stars) |
| POST | `/api/rate/` | save `{attraction_id, rating}` (login required), returns fresh recommendations |

`POST /api/rate/` returns **401** if not logged in, and updates the recommendations in
the same response — this is requirement *"when a user rates an attraction, the current
recommendations are presented."*

---

## 7. The frontend

### Templates
`base.html` holds the responsive **navbar** (Home, Search, Places, Used model, About,
plus Login/Sign up or the username + Logout), the flash-message area, the footer, and a
hidden toast element. Every page `{% extends "base.html" %}` and fills the `content`
block. Two small globals are injected for the JS: `window.IS_AUTHENTICATED` and
`window.LOGIN_URL`.

### JavaScript (`frontend/static/js/`)
| File | Role |
|---|---|
| `common.js` | shared helpers: `getJSON`/`postJSON` (with CSRF header), `toast`, scroll-reveal animation, navbar toggle, image fallback, and the card builders `buildAttractionCard` / `buildRecCard` plus the `rateAttraction` flow |
| `home.js` | placeholder — home recs are server-rendered, animations handled by `common.js` |
| `places.js` | loads the grid, wires the type filter, pre-fills stars from `/api/my-ratings/`, and refreshes the recommendation strip after each rating |
| `search.js` | debounced text search + type filter, rendering read-only discovery cards |

The card builders are deliberately framework-agnostic functions. If you port the
frontend to **Angular**, each `build*` function becomes a component template, the
`state` object becomes component state, and `getJSON`/`postJSON` become an `HttpClient`
service — the API contract does not change.

### Styling and animation
`styles.css` uses CSS variables for the palette (parchment / terracotta / sea-teal /
gold) and includes:
* a sticky blurred navbar with an animated underline and a hamburger menu under 820px,
* scroll-reveal (`.reveal` → `.in`, driven by an `IntersectionObserver`),
* floating hero "orbs", card hover lifts, a star "burst" pop on rating,
* a `prefers-reduced-motion` block that disables animation for accessibility,
* responsive breakpoints at 820px (mobile nav) and 520px (stacked CTAs).

---

## 8. Request lifecycle examples

**Filtering by type on the Places page**
```
user taps "Roman Ruins" chip
  -> places.js sets state.type and calls GET /api/attractions/?type=Roman Ruins
     -> api_attractions filters the Attraction queryset by place_type
        -> returns JSON; places.js rebuilds the grid
```

**Rating an attraction (the core loop)**
```
user clicks 4 stars on "Tassili n'Ajjer"
  -> common.js POST /api/rate/  {attraction_id, rating:4}  (+ X-CSRFToken header)
     -> api_rate checks login, upserts a Rating(account=user)
        -> services.recommend_for(user) folds the new rating into the model
           -> returns refreshed recommendations
              -> common.js re-renders the "Your recommendations" strip
```

---

## 9. How to extend

* **Add an attraction** — add a row to `attractions.csv`, re-run
  `python manage.py seed_db`. (If you want the model to recommend it from collaborative
  signal, you also need ratings for it and a retrain.)
* **Make app ratings affect the trained model** — append registered-account ratings to
  `ratings_clean.csv` and re-run `python ml/train.py`. (Currently they are folded in at
  request time, which is enough for personalised recommendations.)
* **Swap the served algorithm** — change which model `train.py` saves, or adjust the
  `personalized` list in `train.py`.
* **Persist a chosen model permanently** — `model.pkl` is a normal pickle of the model
  object; the backend just unpickles and calls `.recommend(...)`.
