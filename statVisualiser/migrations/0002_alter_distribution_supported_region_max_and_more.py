# Generated by Django 4.0.6 on 2022-07-15 10:38

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('statVisualiser', '0001_initial'),
    ]

    operations = [
        migrations.AlterField(
            model_name='distribution',
            name='supported_region_max',
            field=models.BigIntegerField(blank=True, default=9223372036854775807),
        ),
        migrations.AlterField(
            model_name='distribution',
            name='supported_region_min',
            field=models.BigIntegerField(blank=True, default=-9223372036854775808),
        ),
    ]
