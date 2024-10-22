# 视图函数，用于处理用户上传的图片并进行分类预测：
from django.shortcuts import render
from django.http import JsonResponse
from .models import ArtWork
from .forms import ArtWorkForm
from django.core.files.storage import default_storage
from PIL import Image
import numpy as np
import tensorflow as tf
import os
from django.conf import settings


def home(request):
    return render(request, 'upload.html')


# 加载预训练模型
MODEL_PATH = os.path.join(settings.BASE_DIR, 'create_model', 'simple_model.h5')
model = tf.keras.models.load_model(MODEL_PATH)


# 处理上传的图片
def preprocess_image(image_file, target_size):
    img = Image.open(image_file)  # 打开图片文件
    img = img.resize(target_size)  # 调整大小
    img_array = np.array(img) / 255.0  # 归一化
    img_array = np.expand_dims(img_array, axis=0)  # 增加batch维度
    return img_array


# 分类预测视图
def classify_artwork(request):
    if request.method == 'POST':
        form = ArtWorkForm(request.POST, request.FILES)
        if form.is_valid():
            artwork = form.save()

            # 获取模型输入的高度和宽度
            height, width = model.input_shape[1:3]  # 忽略 batch_size 和 channels
            target_size = (height, width)

            # 直接从表单中获取上传的图片文件
            image_file = request.FILES['image']
            img_array = preprocess_image(image_file, target_size)  # 预处理图片

            # 进行分类预测
            prediction = model.predict(img_array)
            predicted_class = 'AI-generated' if prediction[0] > 0.5 else 'Human-created'

            # 保存预测结果
            artwork.prediction = predicted_class
            artwork.save()

            return JsonResponse({'result': predicted_class})
        else:
            return JsonResponse({'error': 'Invalid form submission'})
    else:
        form = ArtWorkForm()
    return render(request, 'upload.html', {'form': form})
