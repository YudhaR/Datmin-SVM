# Generated by Django 4.2.7 on 2023-11-23 03:41

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('analysis', '0005_initial'),
    ]

    operations = [
        migrations.AlterField(
            model_name='tweet',
            name='created_at',
            field=models.CharField(max_length=255),
        ),
    ]
