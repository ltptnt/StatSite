# Generated by Django 4.0.6 on 2022-07-15 07:38

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('statVisualiser', '0002_distribution_delete_distributions'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='distribution',
            name='supportedRegion',
        ),
        migrations.AddField(
            model_name='distribution',
            name='requiredVariableCount',
            field=models.IntegerField(default=0),
        ),
        migrations.AddField(
            model_name='distribution',
            name='requiredVariableNames',
            field=models.JSONField(blank=True, default={}),
        ),
        migrations.AddField(
            model_name='distribution',
            name='supportedRegionMax',
            field=models.BigIntegerField(blank=True, default=-9223372036854775808),
        ),
        migrations.AddField(
            model_name='distribution',
            name='supportedRegionMin',
            field=models.BigIntegerField(blank=True, default=9223372036854775807),
        ),
    ]
