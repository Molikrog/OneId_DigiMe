# Generated by Django 5.1.1 on 2024-09-29 01:01

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('app', '0003_alter_userinfo_vectors'),
    ]

    operations = [
        migrations.AddField(
            model_name='userinfo',
            name='image_field',
            field=models.ImageField(blank=True, null=True, upload_to='face_images/'),
        ),
    ]
