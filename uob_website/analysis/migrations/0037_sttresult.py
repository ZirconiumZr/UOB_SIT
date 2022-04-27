# Generated by Django 3.2.12 on 2022-04-26 15:43

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('analysis', '0036_delete_sttresult'),
    ]

    operations = [
        migrations.CreateModel(
            name='STTresult',
            fields=[
                ('audio_slice_id', models.CharField(max_length=30, primary_key=True, serialize=False)),
                ('slice_id', models.IntegerField()),
                ('start_time', models.FloatField(default=0)),
                ('end_time', models.FloatField(default=0)),
                ('duration', models.FloatField(default=0)),
                ('speaker_label', models.CharField(default='', max_length=20)),
                ('text', models.CharField(blank=True, max_length=5000, null=True)),
                ('slice_name', models.CharField(default='', max_length=100)),
                ('slice_path', models.CharField(default='', max_length=150)),
                ('create_by', models.CharField(default='tongtong', max_length=50)),
                ('create_date', models.DateField(default='2022-04-26')),
                ('create_time', models.TimeField(default='23:43:13')),
            ],
            options={
                'managed': False,
            },
        ),
    ]
