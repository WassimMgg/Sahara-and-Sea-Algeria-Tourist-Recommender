# Sahara & Sea — Algeria Tourist Recommender

A multi-page web application that recommends Algerian tourist attractions,
personalised to each visitor's taste. Sign up, rate the places you know, and your
recommendations update instantly.

The pipeline: **messy CSV → cleaning → model comparison → trained model → seeded
database → live Django app**.

---

## Quick start

You need **Python 3.10+**.

```bash
# 1. install dependencies
pip install -r requirements.txt

# 2. prepare data + train the model (CSV is used ONLY here)
cd ml
python clean_data.py          # data/ratings_raw.csv -> ml/ratings_clean.csv
python train.py               # compares 4 models -> model.pkl + metrics.json
cd ..

# 3. build the database and load the data into it
python manage.py migrate      # create the tables
python manage.py seed_db      # load attractions + historical ratings from CSV

# 4. create an admin account for the admin panel
python manage.py create_admin   # makes admin / admin12345  (change the password!)

# 5. run the app
python manage.py runserver
```

Open **http://127.0.0.1:8000/**. Click **Sign up**, then go to **Places**, rate a few
attractions, and watch the recommendations update.

The **admin panel** is at **http://127.0.0.1:8000/admin/** — log in with the account
from step 4 (or your own, via `python manage.py createsuperuser`).

---

## Pages

| Page | What it does |
|---|---|
| **Home** | hero + your recommendations (popular ones until you log in and rate) |
| **Search** | search by name/city/region + filter by place type |
| **Places** | the full catalogue; rate places and see live recommendations |
| **Used model** | the algorithm comparison and why the chosen model is used |
| **About** | project overview |
| **Login / Sign up** | account creation and authentication |

## Admin panel

The project includes a full admin panel at **`/admin/`** (Django admin, customised
for this project). After logging in as a staff user you get:

- a **dashboard** on the home page — total attractions, registered users, historical
  visitors and ratings, plus top-rated / most-rated attractions and recent app ratings;
- **Attractions** — list with image thumbnails, rating counts and average score;
  searchable and filterable by region and type; editable with a live image preview;
- **Visitors** — the historical raters, filterable by country and gender;
- **Ratings** — every rating, showing whether it came from an app user or the
  historical data, filterable by score and source;
- **Users / Groups** — standard account management.

Staff users also see an **Admin** link in the site navbar. Create an admin with
`python manage.py create_admin` (or the standard `python manage.py createsuperuser`).

---

## How it meets the task requirements

1. **Recommend attractions for users from my country (Algeria).** 15 real Algerian
   attractions with images and descriptions; the app ranks them per user.
2. **Attraction data + friends' ratings.** 15 attractions and 576 valid ratings from
   50 people after cleaning (deliberately messy raw data).
3. **Select the most efficient recommender.** `ml/train.py` compares four algorithms
   with 5-fold cross-validation (RMSE + MAE) — see the **Used model** page.
4. **Use it in the app: rating shows current recommendations.** `POST /api/rate/`
   stores the rating and returns refreshed recommendations in the same response.
5. **Django backend, HTML/CSS/JS frontend, Python for training.** All present.

---

## Architecture in one paragraph

The CSV files are used **only** to clean the data and train the model (`ml/`). The
cleaned data is then loaded into a **SQLite database** once via `python manage.py
seed_db`, so the running app reads from the database (fast) rather than parsing CSVs.
Django serves the HTML pages and a small JSON API; the trained model (`model.pkl`) is
loaded once and folds in each logged-in user's ratings on the fly to produce
personalised recommendations.

The historical raters from the CSV are stored as `Visitor` rows (used for the
"popular" cold-start ranking) but are **not shown** in the interface. Recommendations
for a logged-in user come from their own ratings stored against their account.

---

## Notes

* **Detailed code documentation** is in **`DOCUMENTATION.md`** (architecture, every
  file, database schema, API, request lifecycles, how to extend).
* **Re-seeding** (`seed_db`) reloads the historical CSV data and keeps registered-user
  ratings.
* **Images** come from Wikimedia links; with internet they load automatically, and any
  unavailable image falls back to a coloured panel with the place name.
* **Angular**: the frontend is plain HTML/CSS/JS structured as component-like functions
  so it ports to Angular without backend changes (the JSON API stays identical). See
  `DOCUMENTATION.md` §7.
