import cv2
import os
import threading
import tkinter as tk
from tkinter import messagebox
import mediapipe as mp

# Variáveis globais
gravando = False
cap = None
out = None
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

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

    # Abrir a janela de escolha (bad/good)
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

    janela_escolha.destroy()  # Fechar a janela de escolha
    janela.destroy()          # Fechar a janela principal

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
janela.geometry("300x150")

botao_gravar = tk.Button(janela, text="Gravar",
                         width=20, command=iniciar_gravacao)
botao_gravar.pack(pady=10)

botao_parar = tk.Button(janela, text="Parar", width=20, command=parar_gravacao)
botao_parar.pack(pady=10)

janela.protocol("WM_DELETE_WINDOW", fechar)
janela.mainloop()
