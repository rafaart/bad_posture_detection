import tkinter as tk
import cv2
import threading
import os
import numpy as np
import pygetwindow as gw
import time
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import mediapipe as mp
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tkinter import messagebox
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Variáveis globais
modelo_classificacao = None
cap = None
tempo_ma_postura = 0
tempo_boa_postura = 0
gravando = False
out = None

# Função para treinar a CNN


def treinar_cnn():
    # Aqui você pode carregar os vídeos gravados e realizar o treinamento
    frames = []  # Colete os frames dos vídeos aqui
    labels = []  # Colete os labels correspondentes aos vídeos

    # Convertendo os dados e labels em numpy arrays
    frames = np.array(frames)
    labels = np.array(labels)

    labels = to_categorical(labels, num_classes=2)  # One-hot encoding

    # Dividir os dados em treinamento e teste
    X_train, X_test, y_train, y_test = train_test_split(
        frames, labels, test_size=0.2, random_state=42)

    # Criar o modelo
    modelo_classificacao = Sequential()
    modelo_classificacao.add(
        Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
    modelo_classificacao.add(MaxPooling2D(pool_size=(2, 2)))
    modelo_classificacao.add(Flatten())
    modelo_classificacao.add(Dense(64, activation='relu'))
    modelo_classificacao.add(Dense(2, activation='softmax'))  # 2 classes

    # Compilar o modelo
    modelo_classificacao.compile(
        optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Treinar o modelo
    modelo_classificacao.fit(X_train, y_train, epochs=10,
                             batch_size=32, validation_data=(X_test, y_test))

    # Salvar o modelo
    modelo_classificacao.save("modelo_postura.h5")

    # Avaliar o desempenho do modelo
    loss, accuracy = modelo_classificacao.evaluate(X_test, y_test)
    messagebox.showinfo("Treinamento Concluído", f"Acurácia: {
                        accuracy:.2f}, Perda: {loss:.2f}")

# Função para carregar o modelo


def carregar_modelo():
    global modelo_classificacao
    if os.path.exists("modelo_postura.h5"):
        modelo_classificacao = load_model("modelo_postura.h5")
    else:
        messagebox.showerror(
            "Erro", "Não há um modelo salvo, use a função treinar modelo para gerar um novo.")

# Função para iniciar a gravação de vídeo


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

    while gravando:
        ret, frame = cap.read()
        if ret:
            # Escrever o frame no arquivo de vídeo
            out.write(frame)

            # Mostrar o vídeo em tempo real
            cv2.imshow('Gravando', frame)

            # Se pressionar a tecla 'q', parar a gravação
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break


def parar_gravacao():
    global gravando, cap, out

    gravando = False

    # Liberar a câmera e fechar todas as janelas
    if gravando:
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print("Gravação parada")

        # Abre a janela para escolher o nome do arquivo
        salvar_video()


def salvar_video():
    global janela_salvar

    # Cria uma nova janela para salvar o arquivo
    janela_salvar = Toplevel(root)
    janela_salvar.title("Salvar Vídeo")

    # Label para instruir o usuário
    Label(janela_salvar, text="Salvar o arquivo como:").pack(pady=10)


def salvar_como(nome_arquivo):
    # Adiciona um sufixo numérico ao nome do arquivo se já existir
    i = 1
    novo_nome = f"{nome_arquivo}.avi"
    while os.path.exists(novo_nome):
        novo_nome = f"{nome_arquivo}_{i}.avi"
        i += 1

    # Renomeia o arquivo gravado
    os.rename('video.avi', novo_nome)
    janela_salvar.destroy()  # Fecha a janela de salvar
    print(f"Arquivo salvo como {novo_nome}")

# Função para testar vídeo em tempo real


def testar_video():
    global cap
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        messagebox.showerror("Erro", "Não foi possível acessar a câmera.")
        return

    carregar_modelo()

    if modelo_classificacao is None:
        return

    # Thread para o teste em tempo real
    threading.Thread(target=real_time_testing).start()

# Função para testar vídeo em tempo real


def real_time_testing():
    global cap, tempo_ma_postura, tempo_boa_postura

    # Variáveis para rastrear o tempo
    tempo_inicial = time.time()  # Tempo inicial da execução
    estado_anterior = None  # Variável para armazenar o estado anterior
    # Flag para verificar se a janela está ativa em primeiro plano
    fullscreen_ativado = False

    # Obtém a janela de vídeo do OpenCV
    janela_video = None

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while True:
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

            # Redimensionar o frame para a classificação
            resized_frame = cv2.resize(frame, (64, 64))
            normalized_frame = np.array(resized_frame, dtype="float32") / 255.0
            input_frame = np.expand_dims(normalized_frame, axis=0)

            # Realizar a classificação
            if modelo_classificacao:
                predictions = modelo_classificacao.predict(input_frame)
                class_index = np.argmax(predictions[0])

                # Debug: Imprimir estado e contadores
                print(f"Classificação atual: {class_index} | Tempo má postura: {
                      tempo_ma_postura} | Tempo boa postura: {tempo_boa_postura}")

                # Se o estado atual for diferente do anterior, resetar contadores
                if class_index != estado_anterior:
                    print("Mudança de estado detectada, resetando contador.")
                    tempo_boa_postura = 0
                    tempo_ma_postura = 0
                    tempo_inicial = time.time()  # Resetar o tempo inicial

                # Atualizar o estado anterior
                estado_anterior = class_index

                # Verifica a classe e atualiza o contador
                if class_index == 1:
                    cor = (0, 255, 0)  # Verde para good posture
                    texto = "Good Posture"
                    # Contagem em segundos para boa postura
                    tempo_boa_postura = int(time.time() - tempo_inicial)
                else:
                    cor = (0, 0, 255)  # Vermelho para bad posture
                    texto = "Bad Posture"
                    # Contagem em segundos para má postura
                    tempo_ma_postura = int(time.time() - tempo_inicial)

                # Desenhar o retângulo e o texto no frame
                cv2.rectangle(frame, (10, 10), (300, 50), cor, -1)
                cv2.putText(frame, texto, (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                # Exibir o contador na tela
                contador = tempo_boa_postura if class_index == 1 else tempo_ma_postura
                cv2.putText(frame, f"Contador: {
                            contador} seg", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                # Verificar a postura do usuário
                if class_index == 0:  # Bad Posture
                    if tempo_ma_postura >= 10 and not fullscreen_ativado:
                        print("Restaurando janela por 10 segundos de má postura.")

                        # Obter a janela do OpenCV
                        if janela_video is None:
                            janela_video = gw.getWindowsWithTitle(
                                'Teste de Postura em Tempo Real')[0]

                        try:
                            # Restaurar a janela se estiver minimizada
                            if janela_video.isMinimized:
                                janela_video.restore()
                            # Ativar a janela se não estiver ativa
                            if not janela_video.isActive:
                                janela_video.activate()
                            fullscreen_ativado = True
                        except Exception as e:
                            print(f"Erro ao ativar/restaurar a janela: {e}")

                else:  # Good Posture
                    if tempo_boa_postura >= 10 and fullscreen_ativado:
                        print("Minimizando janela após 10 segundos de boa postura.")

                        # Minimizar a janela
                        try:
                            janela_video.minimize()
                            fullscreen_ativado = False
                        except Exception as e:
                            print(f"Erro ao minimizar a janela: {e}")

            # Mostrar o vídeo
            cv2.imshow('Teste de Postura em Tempo Real', frame)

            # Se pressionar ESC, para o teste
            if cv2.waitKey(1) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()


# Função para parar o teste


def parar_teste():
    global cap
    if cap:
        cap.release()
        cv2.destroyAllWindows()

# Função para fechar a janela


def fechar():
    global cap
    if cap:
        cap.release()
    janela.destroy()


# Criar a interface gráfica
janela = tk.Tk()
janela.title("Gravação e Teste de Postura")
janela.geometry("300x250")

botao_gravar = tk.Button(janela, text="Gravar",
                         width=20, command=iniciar_gravacao)
botao_gravar.pack(pady=5)

botao_parar = tk.Button(janela, text="Parar", width=20, command=parar_gravacao)
botao_parar.pack(pady=5)

botao_treinar = tk.Button(janela, text="Treinar CNN",
                          width=20, command=treinar_cnn)
botao_treinar.pack(pady=5)

botao_testar = tk.Button(janela, text="Testar Vídeo",
                         width=20, command=testar_video)
botao_testar.pack(pady=5)

botao_parar_teste = tk.Button(
    janela, text="Parar Teste", width=20, command=parar_teste)
botao_parar_teste.pack(pady=5)

janela.protocol("WM_DELETE_WINDOW", fechar)

janela.mainloop()
