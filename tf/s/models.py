from django.db import models

# Create your models here.

class User(models.Model):
    pub_date = models.DateTimeField('date published', null=True)

class Picture(models.Model):
    user = models.ForeignKey(User)
    url = models.CharField(max_length=200, null=True)
    pub_date = models.DateTimeField('date published', null=True)
    
class Featured_Picture(models.Model):
    picture = models.ForeignKey(Picture)
    rank = models.IntegerField()