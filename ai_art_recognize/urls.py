# 定义应用的 URL 路由：
from django.urls import path
from . import views

urlpatterns = [
    path('upload/', views.classify_artwork, name='upload_artwork'),
    path('', views.home, name='home'),  # 添加主页路由
]
