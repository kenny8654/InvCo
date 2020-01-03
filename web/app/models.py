from django.db import models

# Create your models here.

class ImgSave(models.Model):
    img_url = models.ImageField(upload_to='img')

class RecipeSave(models.Model):
    title = models.TextField(u'title',default="")
    content = models.TextField(u'Content')