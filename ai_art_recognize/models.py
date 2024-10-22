from django.db import models


# 定义用于保存用户上传图片的数据库模型：
# Create your models here.


class ArtWork(models.Model):
    image = models.ImageField(upload_to='artworks/')
    uploaded_at = models.DateTimeField(auto_now_add=True)
    prediction = models.CharField(max_length=20, blank=True, null=True)

    def __str__(self):
        return f'Artwork uploaded at {self.uploaded_at}'
