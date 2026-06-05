"""
create_admin
============
Convenience command to create (or update) an admin account for the panel,
so you don't have to run the interactive `createsuperuser`.

    python manage.py create_admin
    python manage.py create_admin --username me --password secret123

Defaults to admin / admin12345 if not given. This is a DEV helper — change the
password for anything real.
"""

import os
from django.core.management.base import BaseCommand
from django.contrib.auth.models import User


class Command(BaseCommand):
    help = "Create or update a superuser for the admin panel."

    def add_arguments(self, parser):
        parser.add_argument("--username", default=os.environ.get("ADMIN_USER", "admin"))
        parser.add_argument("--email", default=os.environ.get("ADMIN_EMAIL", "admin@example.com"))
        parser.add_argument("--password", default=os.environ.get("ADMIN_PASSWORD", "admin12345"))

    def handle(self, *args, **opts):
        user, created = User.objects.get_or_create(
            username=opts["username"], defaults={"email": opts["email"]})
        user.is_staff = True
        user.is_superuser = True
        user.set_password(opts["password"])
        user.save()
        action = "Created" if created else "Updated"
        self.stdout.write(self.style.SUCCESS(
            f"{action} admin '{opts['username']}'. Log in at /admin/ "
            f"(remember to change the password)."))
