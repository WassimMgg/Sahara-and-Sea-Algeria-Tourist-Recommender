"""
api/models.py
=============
The database schema.

The CSV files are used ONLY to train the recommender model and to seed these
tables once (see `python manage.py seed_db`). After seeding, the running app
reads everything from the database, which is much faster than parsing CSVs on
every request.

Tables:
  * Attraction - the catalogue of places (from attractions.csv)
  * Visitor    - the "friends" who gave the historical ratings (from ratings.csv)
  * Rating     - a single rating, linked to EITHER a Visitor (historical data)
                 OR a registered account (ratings made inside the app)
"""

from django.conf import settings
from django.db import models


class Attraction(models.Model):
    # we reuse the CSV's attraction_id as the primary key so it matches the
    # item ids the trained model knows about.
    id = models.IntegerField(primary_key=True)
    name = models.CharField(max_length=200)
    city = models.CharField(max_length=100)
    region = models.CharField(max_length=50)
    category = models.CharField(max_length=100)        # e.g. "Roman Ruins / UNESCO"
    place_type = models.CharField(max_length=60, db_index=True)  # e.g. "Roman Ruins"
    description = models.TextField()
    image_url = models.URLField(max_length=500)

    class Meta:
        ordering = ["id"]

    def __str__(self):
        return self.name


class Visitor(models.Model):
    """A historical rater imported from ratings.csv (used for training/popularity)."""
    id = models.IntegerField(primary_key=True)
    name = models.CharField(max_length=100)
    age = models.IntegerField(null=True, blank=True)
    gender = models.CharField(max_length=20, blank=True)
    home_country = models.CharField(max_length=60, blank=True)

    def __str__(self):
        return f"{self.name} (#{self.id})"


class Rating(models.Model):
    visitor = models.ForeignKey(
        Visitor, null=True, blank=True, on_delete=models.CASCADE, related_name="ratings")
    account = models.ForeignKey(
        settings.AUTH_USER_MODEL, null=True, blank=True,
        on_delete=models.CASCADE, related_name="ratings")
    attraction = models.ForeignKey(
        Attraction, on_delete=models.CASCADE, related_name="ratings")
    rating = models.FloatField()
    visit_date = models.CharField(max_length=20, blank=True)

    class Meta:
        # a given account rates a given attraction at most once
        constraints = [
            models.UniqueConstraint(
                fields=["account", "attraction"],
                condition=models.Q(account__isnull=False),
                name="unique_account_attraction",
            ),
        ]

    def __str__(self):
        who = self.account or self.visitor
        return f"{who} -> {self.attraction} = {self.rating}"
