# Generated by Django 3.2.12 on 2022-04-20 15:06

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('analysis', '0022_auto_20220420_2305'),
    ]

    operations = [
        migrations.CreateModel(
            name='Version',
            fields=[
                ('version_id', models.AutoField(primary_key=True, serialize=False)),
                ('version_name', models.CharField(max_length=30, unique=True)),
                ('version_value', models.CharField(max_length=30)),
            ],
        ),
        migrations.AlterField(
            model_name='audio',
            name='create_time',
            field=models.TimeField(default='23:06:15'),
        ),
        migrations.AlterField(
            model_name='audio',
            name='update_time',
            field=models.TimeField(default='23:06:15'),
        ),
    ]
