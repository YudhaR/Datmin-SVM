# Generated by Django 4.2.7 on 2023-12-05 17:01

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('analysis', '0012_remove_clean_sentimen_remove_tweet_sentimen_and_more'),
    ]

    operations = [
        migrations.AddField(
            model_name='tweet',
            name='idt',
            field=models.IntegerField(default=0),
        ),
    ]