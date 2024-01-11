import os
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import VGG19
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import preprocess_input, decode_predictions
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import preprocess_input

x_data = []
y_data = []
# Especificar o diretório que você quer listar
diretorio_train_normal = './train/NORMAL'
diretorio_train_pneumonia = './train/PNEUMONIA'

# Listar arquivos e diretórios
arquivos_train_normal = os.listdir(diretorio_train_normal)
arquivos_train_pneumonia = os.listdir(diretorio_train_pneumonia)
# Exibir os nomes dos arquivos

for normal in arquivos_train_normal:
     if normal != '.DS_Store':
        #print(normal,pneumonia)
        imagem = image.load_img('./train/NORMAL/'+normal, target_size=(224, 224))
        img_array = image.img_to_array(imagem)
        img_array = preprocess_input(img_array)
        x_data.append(img_array)
        y_data.append(0)

for pneumonia in arquivos_train_pneumonia:
    if pneumonia != '.DS_Store':
        #print(normal,pneumonia)
        imagem = image.load_img('./train/PNEUMONIA/'+pneumonia, target_size=(224, 224))
        img_array = image.img_to_array(imagem)
        img_array = preprocess_input(img_array)
        x_data.append(img_array)
        y_data.append(1)
print('terminou a leitura das imagens')
x_data = np.array(x_data)
y_data = np.array(y_data)
x_treino, x_teste, y_treino, y_teste = train_test_split(x_data, y_data,
test_size=0.3,random_state=42) #Amostras de 70% para treino e 30% para teste#

num_classes = 2
#VGG19 pré-treinado
#VGG19 sem as camadas totalmente conectadas
base_model = VGG19(weights='imagenet', include_top=False)

#camadas totalmente conectadas para a nova tarefa de classificação
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)

#Novo modelo combinando a base pré-treinada com as camadas personalizadas
model = Model(inputs=base_model.input, outputs=predictions)

# Congelar as camadas do modelo base
for layer in base_model.layers:
    layer.trainable = False

# Compilar o modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Tranformando os rótulos em codificação one-hot
y_train_one_hot = to_categorical(y_treino, num_classes=num_classes)
y_test_one_hot = to_categorical(y_teste, num_classes=num_classes)

# Treinando o modelo
model.fit(x_treino, y_train_one_hot, epochs=10, batch_size=32, validation_data=(x_teste, y_test_one_hot))

# Medindo o desempenho do modelo nos dados de teste
accuracy = model.evaluate(x_teste, y_test_one_hot)[1]
print(f'Acurácia nos dados de teste: {accuracy * 100:.2f}%')



