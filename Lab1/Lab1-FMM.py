import sounddevice as sd
from scipy.signal import resample
import numpy as np
from scipy.io.wavfile import write
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import simpledialog, messagebox
import threading


def filtroMediaMovil(señal, M):
    y = np.zeros_like(señal)
    for n in range(M, len(señal)):
        y[n] = np.sum(señal[n-M+1:n+1]) / M
    return y



def aplicarNyquist(data, oldSampleRate, newSampleRate):
    nMuestras = round(len(data) * float(newSampleRate) / oldSampleRate)
    dataResampled = resample(data, nMuestras)
    outputFilename = f'grabacion_voz_{newSampleRate}Hz.wav'
    write(outputFilename, int(newSampleRate), dataResampled)
    sd.play(dataResampled, newSampleRate)
    sd.wait()


def graficarSeñales(t, señalSin, filtered_signal, fs):
    plt.figure(figsize=(10, 6))
    plt.plot(t, señalSin, label='Señal original')
    plt.plot(t, filtered_signal, label='Señal filtrada', linestyle='--')
    plt.title(f'Señal Original vs Señal Filtrada a {fs} Hz')
    plt.xlabel('Tiempo [s]')
    plt.ylabel('Amplitud')
    plt.legend()
    plt.grid(True)
    plt.show()


def mostrar_menu(señalSin, fs, duration):
    menu_window = tk.Tk()
    menu_window.geometry("300x200")
    menu_window.title("Opciones")

    def aplicar_filtro():
        M = simpledialog.askinteger("Filtro de Media Móvil", "Ingresa el orden del filtro (M):", initialvalue=10)
        if M is None: return
        
        filtered_signal = filtroMediaMovil(señalSin[:, 0], M)
        t = np.linspace(0, duration, len(señalSin))

        filtro_win = tk.Toplevel(menu_window)
        filtro_win.geometry("300x200")
        filtro_win.title("Filtro Media Móvil")

        def play_orig():
            sd.play(señalSin, fs)

        def play_filt():
            sd.play(filtered_signal, fs)

        def plot_sigs():
            graficarSeñales(t, señalSin, filtered_signal, fs)

        tk.Button(filtro_win, text="Reproducir grabación original", command=play_orig).pack(pady=10)
        tk.Button(filtro_win, text="Reproducir señal filtrada", command=play_filt).pack(pady=10)
        tk.Button(filtro_win, text="Ver gráfica", command=plot_sigs).pack(pady=10)

    def aplicar_nyquist():
        nyquist_win = tk.Toplevel(menu_window)
        nyquist_win.geometry("300x300")
        nyquist_win.title("Criterio de Nyquist")

        def play_orig():
            sd.play(señalSin, fs)

        tk.Button(nyquist_win, text="Reproducir original", command=play_orig).pack(pady=10)
        tk.Button(nyquist_win, text="Reproducir a 1/4 de criterio de nyquist", command=lambda: aplicarNyquist(señalSin[:, 0], fs, int(fs * 1/8))).pack(pady=10)
        tk.Button(nyquist_win, text="Reproducir a 3/5 de criterio de nyquist", command=lambda: aplicarNyquist(señalSin[:, 0], fs, int(fs * 3/10))).pack(pady=10)
        tk.Button(nyquist_win, text="Reproducir a 6/5 de criterio de nyquist", command=lambda: aplicarNyquist(señalSin[:, 0], fs, int(fs * 6/10))).pack(pady=10)
        tk.Button(nyquist_win, text="Reproducir a 8/5 de criterio de nyquist", command=lambda: aplicarNyquist(señalSin[:, 0], fs, int(fs * 8/10))).pack(pady=10)

    def volver_grabar():
        menu_window.destroy()
        grabar_y_mostrar_menu()

    tk.Button(menu_window, text="Aplicar Filtro de Media Móvil", command=aplicar_filtro).pack(pady=10)
    tk.Button(menu_window, text="Aplicar Criterio de Nyquist", command=aplicar_nyquist).pack(pady=10)
    tk.Button(menu_window, text="Volver a Grabar", command=volver_grabar).pack(pady=10)


def grabar_audio(fs, duration):
    label_status.config(text="El audio está siendo grabado...", fg="blue")
    root.update()
    señalSin = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()
    label_status.config(text="Grabación completada", fg="green")
    root.update()
    return señalSin


def grabar_y_mostrar_menu():
    try:
        fs = int(entry_fs.get())
        duration = int(entry_duration.get())
        
        señalSin = grabar_audio(fs, duration)
        mostrar_menu(señalSin, fs, duration)
    
    except ValueError:
        messagebox.showerror("Error", "Por favor ingrese valores válidos.")

def run_gui():
    global root, entry_fs, entry_duration, label_status

    root = tk.Tk()
    root.title("Grabación y Procesamiento de Señal de Audio")

    root.geometry("300x200")
    root.configure(padx=20, pady=20)


    tk.Label(root, text="Frecuencia de Muestreo (Hz):").pack()
    entry_fs = tk.Entry(root, width=20)
    entry_fs.pack()
    entry_fs.insert(0, "40000")

    tk.Label(root, text="Duración de la grabación (s):").pack()
    entry_duration = tk.Entry(root, width=20)
    entry_duration.pack()
    entry_duration.insert(0, "5")

    label_status = tk.Label(root, text="", fg="red")
    label_status.pack(pady=10)

    tk.Button(root, text="Iniciar Grabación", command=grabar_y_mostrar_menu).pack(pady=5)

    root.mainloop()

run_gui()
