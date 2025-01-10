# Generated by Django 4.2.17 on 2025-01-09 07:01

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):
    dependencies = [
        ("core", "0003_alter_story_expanded_plot"),
        ("characters", "0001_initial"),
    ]

    operations = [
        migrations.AddField(
            model_name="character",
            name="story",
            field=models.ForeignKey(
                default=None,
                on_delete=django.db.models.deletion.CASCADE,
                related_name="characters",
                to="core.story",
            ),
            preserve_default=False,
        ),
    ]
