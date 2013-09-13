from django.db import models

# Create your models here.

class User(models.Model):
    pub_date = models.DateTimeField('date published')

class Picture(models.Model):
    user = models.ForeignKey(User)
    url = models.CharField(max_length=200)
    
class Featured_Picture(models.Model):
    pictures = models.ForeignKey(Picture)
    rank = models.IntegerField()