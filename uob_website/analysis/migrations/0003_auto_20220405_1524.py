# Generated by Django 2.1.15 on 2022-04-05 07:24

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('analysis', '0002_auto_20220405_1520'),
    ]

    operations = [
        migrations.AlterField(
            model_name='audio',
            name='create_time',
            field=models.TimeField(default='15:24:22'),
        ),
        migrations.AlterField(
            model_name='audio',
            name='update_time',
            field=models.TimeField(default='15:24:22'),
        ),
        migrations.AlterField(
            model_name='sttresult',
            name='create_time',
            field=models.TimeField(default='15:24:22'),
        ),
    ]