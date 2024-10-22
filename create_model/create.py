import os
import sys
import django
from datetime import datetime
from tensorflow.keras import layers, models
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 添加项目路径到系统路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

# 设置Django环境
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'art_recognition_djangoProject.settings')
django.setup()

from django.conf import settings  # 导入settings模块

def create_model():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu', kernel_regularizer=l2(0.001)))  # 添加L2正则化
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dropout(0.5))  # 添加Dropout
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model

if __name__ == '__main__':
    # 创建ImageDataGenerator实例，用于数据增强和加载
    train_datagen = ImageDataGenerator(rescale=1.0/255,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True)

    val_datagen = ImageDataGenerator(rescale=1.0/255)

    # 设置训练数据和验证数据的目录
    train_data_dir = os.path.join(settings.BASE_DIR, 'exercise_data/train_data')  # 替换为实际训练数据的路径
    val_data_dir = os.path.join(settings.BASE_DIR, 'exercise_data/val_data')  # 替换为实际验证数据的路径

    # 加载训练数据
    train_data = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='binary'
    )

    # 加载验证数据
    val_data = val_datagen.flow_from_directory(
        val_data_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='binary'
    )

    model = create_model()

    # 早期停止
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # 使用数据集进行训练
    history = model.fit(train_data, epochs=30, validation_data=val_data, callbacks=[early_stopping])

    # 保存模型
    model.save(f'improved_model_{datetime.now().strftime("%Y%m%d_%H%M%S")}.h5')
