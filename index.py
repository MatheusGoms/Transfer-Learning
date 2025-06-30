# Etapa 1: Importar as bibliotecas necessárias
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
import os
import matplotlib.pyplot as plt

# Etapa 2: Configurar o dataset (substitua pelas suas classes)
# Certifique-se de que o dataset está organizado em pastas: "train/Classe_A", "train/Classe_B"
from google.colab import drive
drive.mount('/content/drive')  # Conectar ao Google Drive (caso o dataset esteja lá)

dataset_path = "/content/drive/MyDrive/seu_dataset"  # Caminho do seu dataset no Drive

# Etapa 3: Preparação dos dados (ImageDataGenerator para pré-processamento)
train_datagen = ImageDataGenerator(
    rescale=1./255,  # Normalização
    rotation_range=20,  # Rotação
    width_shift_range=0.2,  # Deslocamento horizontal
    height_shift_range=0.2,  # Deslocamento vertical
    shear_range=0.2,  # Distorção
    zoom_range=0.2,  # Zoom
    horizontal_flip=True,  # Espelhamento
    fill_mode='nearest'  # Preenchimento
)

test_datagen = ImageDataGenerator(rescale=1./255)  # Apenas normalização para dados de teste

train_generator = train_datagen.flow_from_directory(
    os.path.join(dataset_path, 'train'),
    target_size=(224, 224),  # Tamanho padrão para MobileNetV2
    batch_size=32,
    class_mode='binary'  # Para duas classes
)

test_generator = test_datagen.flow_from_directory(
    os.path.join(dataset_path, 'test'),
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

# Etapa 4: Carregar o modelo pré-treinado (MobileNetV2)
base_model = MobileNetV2(weights='imagenet', include_top=False)  # Modelo sem a última camada

# Congelar as camadas do modelo pré-treinado
for layer in base_model.layers:
    layer.trainable = False

# Etapa 5: Construir o modelo
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),  # Camada para redução dimensional
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')  # Saída para duas classes
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Etapa 6: Treinamento do modelo
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=10,  # Ajuste conforme necessário
    validation_data=test_generator,
    validation_steps=len(test_generator)
)

# Etapa 7: Avaliação e visualização dos resultados
# Gráfico de precisão e perda
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Accuracy')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Loss')
plt.show()

# Avaliação final no conjunto de teste
test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Test Loss: {test_loss:.2f}")
print(f"Test Accuracy: {test_accuracy:.2f}")
