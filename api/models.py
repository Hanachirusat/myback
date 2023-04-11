from django.db import models

# Create your models here.

class User(models.Model):
    username = models.CharField(max_length=255)
    openid = models.CharField(max_length=255)


class History(models.Model):
    name = models.CharField(max_length=255)
    sex = models.CharField(max_length=255)
    age = models.CharField(max_length=255)
    describe = models.CharField(max_length=255)
    rate = models.CharField(max_length=255)
    disease = models.CharField(max_length=255)
    drug = models.CharField(max_length=255)
    time = models.DateTimeField()
    openid = models.CharField(max_length=255)

class recommend(models.Model):
    diseasename=models.CharField(max_length=255)
    drugname=models.CharField(max_length=255)
