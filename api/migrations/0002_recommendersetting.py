from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("api", "0001_initial"),
    ]

    operations = [
        migrations.CreateModel(
            name="RecommenderSetting",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True,
                                           serialize=False, verbose_name="ID")),
                ("key", models.CharField(max_length=50, unique=True)),
                ("value", models.CharField(max_length=100)),
            ],
        ),
    ]
