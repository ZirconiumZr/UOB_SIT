# Generated by Django 3.2.12 on 2022-04-26 10:37

import datetime
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('analysis', '0030_auto_20220422_0134'),
    ]

    operations = [
        migrations.AlterModelOptions(
            name='audio',
            options={'managed': False},
        ),
        migrations.AlterField(
            model_name='upload',
            name='document',
            field=models.FileField(blank=True, upload_to='E:/EBAC/Internship/UOB/Projects/uob_web_SIT-copy/audio/'),
        ),
        migrations.AlterField(
            model_name='upload',
            name='uploaded_at',
            field=models.DateTimeField(default=datetime.date(2022, 4, 26)),
        ),
        migrations.DeleteModel(
            name='STTresult',
        ),
    ]