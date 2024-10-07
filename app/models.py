from django.db import models
from django.contrib.auth.models import User

class UserInfo(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    vector = models.TextField(null=True, )
    image_field = models.ImageField(upload_to='images/', null=True)

class Documents(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    government_id = models.ImageField(upload_to='media/documents/', null=True, blank=True)
    drivers_license = models.ImageField(upload_to='media/documents/', null=True, blank=True)
    index = models.ImageField(upload_to='media/documents/', null=True, blank=True)
    medical_insurance = models.ImageField(upload_to='media/documents/', null=True, blank=True)



class IDCheck(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    picture = models.ImageField(upload_to='id_checks/')
    date = models.DateTimeField(auto_now_add=True)
    result = models.CharField(max_length=255, null=True, blank=True)

    def __str__(self):
        return f"ID Check for {self.user.username}"