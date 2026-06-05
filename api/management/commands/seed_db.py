"""
seed_db
=======
Loads the CSV files into the database tables ONE TIME.

    python manage.py seed_db

  * attractions.csv     -> Attraction rows
  * ml/ratings_clean.csv -> Visitor rows + their Rating rows

Run `python manage.py migrate` first. Re-running this command wipes and reloads
the seeded data (registered-account ratings are kept).
"""

import csv
from django.core.management.base import BaseCommand
from django.conf import settings
from django.db import transaction

from api.models import Attraction, Visitor, Rating


def derive_place_type(category: str) -> str:
    """'Roman Ruins / UNESCO' -> 'Roman Ruins'  (the part before the first '/')."""
    return category.split("/")[0].strip() if category else "Other"


class Command(BaseCommand):
    help = "Load attractions and historical ratings from the CSV files into the database."

    @transaction.atomic
    def handle(self, *args, **options):
        attractions_csv = settings.DATA_DIR / "attractions.csv"
        ratings_csv = settings.ML_DIR / "ratings_clean.csv"

        if not ratings_csv.exists():
            self.stderr.write("ml/ratings_clean.csv not found. Run "
                              "`python ml/clean_data.py` first.")
            return

        # ---- wipe previously seeded data (keep registered-account ratings) ----
        Rating.objects.filter(visitor__isnull=False).delete()
        Visitor.objects.all().delete()
        Attraction.objects.all().delete()

        # ---- attractions ----
        attractions = []
        with open(attractions_csv, encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                attractions.append(Attraction(
                    id=int(row["attraction_id"]),
                    name=row["name"],
                    city=row["city"],
                    region=row["region"],
                    category=row["category"],
                    place_type=derive_place_type(row["category"]),
                    description=row["description"],
                    image_url=row["image_url"],
                ))
        Attraction.objects.bulk_create(attractions)
        self.stdout.write(f"Inserted {len(attractions)} attractions")

        # ---- visitors + their ratings ----
        visitors = {}
        ratings = []
        valid_ids = set(Attraction.objects.values_list("id", flat=True))
        with open(ratings_csv, encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                vid = int(row["user_id"])
                aid = int(row["attraction_id"])
                if aid not in valid_ids:
                    continue
                if vid not in visitors:
                    visitors[vid] = Visitor(
                        id=vid,
                        name=row["user_name"],
                        age=int(row["age"]) if row["age"].strip() else None,
                        gender=row["gender"],
                        home_country=row["home_country"],
                    )
                ratings.append((vid, aid, float(row["rating"]), row["visit_date"]))

        Visitor.objects.bulk_create(list(visitors.values()))
        Rating.objects.bulk_create([
            Rating(visitor_id=vid, attraction_id=aid, rating=r, visit_date=d)
            for vid, aid, r, d in ratings
        ])
        self.stdout.write(f"Inserted {len(visitors)} visitors and {len(ratings)} ratings")
        self.stdout.write(self.style.SUCCESS("Database seeded successfully."))
