# Generated by Django 5.1.1 on 2024-09-29 15:00

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('app', '0005_remove_userinfo_image_field'),
    ]

    operations = [
        migrations.AddField(
            model_name='userinfo',
            name='image_field',
            field=models.ImageField(null=True, upload_to='images/'),
        ),
    ]
