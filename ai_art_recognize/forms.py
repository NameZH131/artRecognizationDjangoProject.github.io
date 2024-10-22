# 创建表单类：
from django import forms
from .models import ArtWork


class ArtWorkForm(forms.ModelForm):
    class Meta:
        model = ArtWork
        fields = ['image']