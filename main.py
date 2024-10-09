import cv2
import os
import threading
import tkinter as tk
from tkinter import messagebox
import mediapipe as mp
import numpy as np
from keras.models import load_model

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

# Função para treinar a CNN (placeholder)


def treinar_cnn():
    messagebox.showinfo(
        "Info", "Função de treinamento da CNN não implementada ainda.")

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

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_frame)

            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Classificação da postura (placeholder)
            postura = classificar_postura(frame)

            # Desenhar o resultado na tela
            if postura == "good":
                cv2.rectangle(frame, (10, 10), (630, 100), (0, 255, 0), -1)
                cv2.putText(frame, "Good Posture", (20, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
            elif postura == "bad":
                cv2.rectangle(frame, (10, 10), (630, 100), (0, 0, 255), -1)
                cv2.putText(frame, "Bad Posture", (20, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)

            # Mostrar o vídeo em tempo real
            cv2.imshow('Teste de Postura em Tempo Real', frame)

            if cv2.waitKey(1) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()

# Função para classificar a postura (placeholder)


def classificar_postura(frame):
    # Aqui você pode adicionar lógica para classificar usando o modelo treinado
    # Exemplo: retornar "good" ou "bad" com base em uma previsão do modelo
    return "good" if np.random.rand() > 0.5 else "bad"  # Substitua pela lógica real

# Função para fechar a aplicação


def fechar():
    if gravando:
        messagebox.showwarning(
            "Aviso", "Por favor, pare a gravação antes de fechar.")
    else:
        janela.destroy()


# Configuração da interface gráfica
janela = tk.Tk()
janela.title("Gravar Vídeo")
janela.geometry("300x300")

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
