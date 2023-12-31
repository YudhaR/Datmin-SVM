# Generated by Django 4.2.7 on 2023-11-07 10:08

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('analysis', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='Tweet',
            fields=[
                ('id', models.AutoField(primary_key=True, serialize=False)),
                ('created_at', models.DateTimeField()),
                ('id_str', models.CharField(max_length=255)),
                ('full_text', models.TextField()),
                ('quote_count', models.IntegerField()),
                ('reply_count', models.IntegerField()),
                ('retweet_count', models.IntegerField()),
                ('favorite_count', models.IntegerField()),
                ('lang', models.CharField(max_length=10)),
                ('user_id_str', models.CharField(max_length=255)),
                ('conversation_id_str', models.CharField(max_length=255)),
                ('username', models.CharField(max_length=255)),
                ('tweet_url', models.URLField()),
                ('sentimen', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='analysis.sentimen')),
            ],
        ),
    ]
