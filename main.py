import cv2
import os
import threading
import tkinter as tk
from tkinter import messagebox
import mediapipe as mp
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import glob

# Variáveis globais
gravando = False
cap = None
out = None
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
modelo_classificacao = None  # Modelo para classificação de posturas
janela_testar = None          # Janela para teste em tempo real

# Função para iniciar a gravação


def iniciar_gravacao():
    global gravando, cap, out
    gravando = True

    # Abrir webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        messagebox.showerror("Erro", "Não foi possível acessar a câmera.")
        return

    # Resolução da câmera
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    # Caminho temporário para gravar o vídeo
    pasta_destino = os.path.join(os.getcwd(), "videos")
    if not os.path.exists(pasta_destino):
        os.makedirs(pasta_destino)

    # Nome temporário para o arquivo
    caminho_temp = os.path.join(pasta_destino, "gravacao_temp.mp4")

    # Definir o codec e criar o objeto VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec para o formato mp4
    out = cv2.VideoWriter(caminho_temp, fourcc, 20.0,
                          (frame_width, frame_height))

    # Thread para gravar o vídeo com pose estimation
    threading.Thread(target=gravar_video).start()


def gravar_video():
    global gravando, cap, out
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while gravando:
            ret, frame = cap.read()
            if not ret:
                break

            # Converter o frame de BGR para RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Fazer a pose estimation
            results = pose.process(rgb_frame)

            # Desenhar os landmarks da pose no frame original
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Escrever o frame com pose no arquivo de vídeo
            out.write(frame)

            # Mostrar o vídeo em tempo real com as linhas da pose
            cv2.imshow('Gravando com Pose Estimation...', frame)

            # Se pressionar ESC, para a gravação
            if cv2.waitKey(1) & 0xFF == 27:
                parar_gravacao()
                break

# Função para parar a gravação


def parar_gravacao():
    global gravando, cap, out
    gravando = False

    if cap:
        cap.release()
    if out:
        out.release()

    cv2.destroyAllWindows()

    # Chamar a janela de escolha para salvar
    abrir_janela_escolha()

# Função para adicionar um sufixo numérico se o arquivo já existir


def adicionar_sufixo_arquivo(caminho_base):
    sufixo = 1
    caminho_final = caminho_base + ".mp4"

    while os.path.exists(caminho_final):
        caminho_final = f"{caminho_base}_{sufixo}.mp4"
        sufixo += 1

    return caminho_final

# Função para salvar o vídeo com base na escolha


def salvar_video(escolha):
    pasta_destino = os.path.join(os.getcwd(), "videos")
    caminho_temp = os.path.join(pasta_destino, "gravacao_temp.mp4")

    if escolha == 'bad':
        nome_base = os.path.join(pasta_destino, "bad_posture")
    elif escolha == 'good':
        nome_base = os.path.join(pasta_destino, "good_posture")
    else:
        return

    # Gerar caminho final com sufixo se necessário
    caminho_final = adicionar_sufixo_arquivo(nome_base)

    # Renomear o arquivo temporário
    os.rename(caminho_temp, caminho_final)
    messagebox.showinfo("Info", f"Gravação salva como {
                        os.path.basename(caminho_final)}!")

# Função para abrir a janela de escolha (botões bad/good)


def abrir_janela_escolha():
    global janela_escolha
    janela_escolha = tk.Toplevel()
    janela_escolha.title("Escolher nome do vídeo")
    janela_escolha.geometry("300x150")

    label = tk.Label(janela_escolha, text="Escolha o nome do vídeo:")
    label.pack(pady=10)

    botao_bad = tk.Button(janela_escolha, text="Bad Posture",
                          width=20, command=lambda: salvar_video('bad'))
    botao_bad.pack(pady=5)

    botao_good = tk.Button(janela_escolha, text="Good Posture",
                           width=20, command=lambda: salvar_video('good'))
    botao_good.pack(pady=5)

# Função para treinar a CNN


def treinar_cnn():
    global modelo_classificacao
    pasta_videos = os.path.join(os.getcwd(), "videos")
    video_files = glob.glob(os.path.join(pasta_videos, "*.mp4"))

    frames = []
    labels = []

    # Extrair frames e labels dos vídeos
    for video_file in video_files:
        label = "good posture" if "good_posture" in video_file else "bad posture"
        capture = cv2.VideoCapture(video_file)

        while True:
            ret, frame = capture.read()
            if not ret:
                break
            # Redimensionar e normalizar os frames
            frame = cv2.resize(frame, (64, 64))
            frames.append(frame)
            labels.append(label)

        capture.release()

    # Converter listas em arrays numpy
    frames = np.array(frames, dtype="float32") / 255.0  # Normalizar
    labels = np.array(labels)

    # Converter labels para uma forma numérica
    labels = np.where(labels == "good posture", 1,
                      0)  # 1 para good, 0 para bad
    labels = to_categorical(labels)

    # Dividir os dados em conjuntos de treino e teste
    X_train, X_test, y_train, y_test = train_test_split(
        frames, labels, test_size=0.2, random_state=42)

    # Criar a arquitetura da CNN
    modelo_classificacao = Sequential()
    modelo_classificacao.add(
        Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
    modelo_classificacao.add(MaxPooling2D(pool_size=(2, 2)))
    modelo_classificacao.add(Conv2D(64, (3, 3), activation='relu'))
    modelo_classificacao.add(MaxPooling2D(pool_size=(2, 2)))
    modelo_classificacao.add(Flatten())
    modelo_classificacao.add(Dense(64, activation='relu'))
    modelo_classificacao.add(Dense(2, activation='softmax'))  # Duas classes

    # Compilar o modelo
    modelo_classificacao.compile(
        optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Treinar o modelo
    modelo_classificacao.fit(X_train, y_train, epochs=10,
                             validation_data=(X_test, y_test))

    # Avaliar o modelo
    loss, accuracy = modelo_classificacao.evaluate(X_test, y_test)
    messagebox.showinfo("Treinamento Concluído", f"Loss: {
                        loss:.4f}\nAcurácia: {accuracy:.4f}")

# Função para testar vídeo em tempo real


def testar_video():
    global janela_testar
    janela_testar = tk.Toplevel()
    janela_testar.title("Teste de Postura")
    janela_testar.geometry("640x480")

    # Iniciar a captura de vídeo
    threading.Thread(target=capturar_video).start()


def capturar_video():
    global cap, modelo_classificacao
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        messagebox.showerror("Erro", "Não foi possível acessar a câmera.")
        return

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Processar o frame para pose estimation
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_frame)

            # Desenhar os landmarks da pose no frame original
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Preparar o frame para classificação
            resized_frame = cv2.resize(frame, (64, 64))
            normalized_frame = np.array(resized_frame, dtype="float32") / 255.0
            input_frame = np.expand_dims(normalized_frame, axis=0)

            # Realizar a classificação
            if modelo_classificacao:
                predictions = modelo_classificacao.predict(input_frame)
                class_index = np.argmax(predictions[0])
                if class_index == 1:
                    cor = (0, 255, 0)  # Verde para good posture
                    texto = "Good Posture"
                else:
                    cor = (0, 0, 255)  # Vermelho para bad posture
                    texto = "Bad Posture"

                # Desenhar o retângulo e o texto no frame
                cv2.rectangle(frame, (10, 10), (300, 50), cor, -1)
                cv2.putText(frame, texto, (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # Mostrar o vídeo
            cv2.imshow('Teste de Postura em Tempo Real', frame)

            # Se pressionar ESC, para o teste
            if cv2.waitKey(1) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()

# Função para fechar a janela


def fechar():
    global cap, janela_testar
    if cap:
        cap.release()
    if janela_testar:
        janela_testar.destroy()
    janela.destroy()


# Criar a interface gráfica
janela = tk.Tk()
janela.title("Gravação de Vídeo")
janela.geometry("300x200")

botao_gravar = tk.Button(janela, text="Gravar",
                         width=20, command=iniciar_gravacao)
botao_gravar.pack(pady=10)

botao_parar = tk.Button(janela, text="Parar", width=20, command=parar_gravacao)
botao_parar.pack(pady=10)

botao_treinar = tk.Button(janela, text="Treinar CNN",
                          width=20, command=treinar_cnn)
botao_treinar.pack(pady=10)

botao_testar = tk.Button(janela, text="Testar Vídeo",
                         width=20, command=testar_video)
botao_testar.pack(pady=10)

janela.protocol("WM_DELETE_WINDOW", fechar)
janela.mainloop()
