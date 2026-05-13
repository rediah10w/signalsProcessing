import tkinter as tk
from tkinter import ttk
import sounddevice as sd
import wavio
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.widgets import CheckButtons
fs = 44100  
fc1=1500
fc2=4000

def ecuacion_diferencia(x):
    y = np.zeros_like(x)
    for n in range(len(x)):
         y[n] = (
            (0.026453909361) * x[n]
            -(0.052941461253) * x[n - 2]
            + (0.026453909361) * x[n - 4]
            - (0.686901908922) * y[n - 4]
            + (2.768390086440) * y[n - 3]
            - (4.462863134502) * y[n - 2]
            + (3.362908710365) * y[n - 1]
        )
    return y


class AudioApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Grabadora de Audio")
        self.duration = tk.DoubleVar(value=2.0)  
        self.audio_data = None
        self.señal_Filtrada = None
        self.create_widgets()

    def create_widgets(self):
        ttk.Label(self.root, text="Duración de la grabación (segundos):").grid(
            column=0, row=0, padx=10, pady=10
        )
        self.duration_entry = ttk.Entry(self.root, textvariable=self.duration)
        self.duration_entry.grid(column=1, row=0, padx=10, pady=10)

        self.record_button = ttk.Button(
            self.root, text="Grabar", command=self.record_audio
        )
        self.record_button.grid(column=0, row=1, padx=10, pady=10)

        self.status_label = ttk.Label(self.root, text="")
        self.status_label.grid(column=1, row=1, columnspan=2, padx=10, pady=10)

        self.play_button = ttk.Button(
            self.root, text="Reproducir Audio Original", command=self.play_audio
        )
        self.play_button.grid(column=0, row=2, padx=10, pady=10)
        self.play_button = ttk.Button(
            self.root, text="Reproducir Audio filtrado", command=self.play_audioFiltrado
        )
        self.play_button.grid(column=0, row=3, padx=10, pady=10)

        self.spectrum_button = ttk.Button(
            self.root, text="Mostrar Graficas", command=self.show_spectrum
        )
        self.spectrum_button.grid(column=0, row=4, padx=10, pady=10)

    def record_audio(self):
        duration = self.duration.get()
        self.status_label.config(text="Grabando señal...")
        self.root.update_idletasks()  # Actualizar la interfaz para mostrar el mensaje

        print(f"Grabando por {duration} segundos...")
        self.audio_data = sd.rec(
            int(duration * fs),
            samplerate=fs,
            channels=1,
            dtype="float32",
        )
        sd.wait()
        
        self.status_label.config(text="Grabación finalizada.")
        self.root.update_idletasks()
        print("Grabación finalizada.")

        self.directorio_actual = os.path.dirname(os.path.abspath(__file__))
        ruta_archivo = os.path.join(self.directorio_actual, "Audio Original.wav")

        wavio.write(ruta_archivo, self.audio_data, fs, sampwidth=2)

        self.audio_Principal = self.audio_data.flatten()
        self.fft_audio = np.fft.fft(self.audio_Principal)
        self.señal_Filtrada = ecuacion_diferencia(self.audio_Principal)
        self.fftSeñalfiltrada = np.fft.fft(self.señal_Filtrada)

        ruta_archivo = os.path.join(self.directorio_actual, "Audio Filtrado.wav")
        wavio.write(ruta_archivo, self.señal_Filtrada, fs, sampwidth=2)

    def play_audio(self):
        if self.audio_data is not None:
            print("Reproduciendo...")
            self.status_label.config(text="Reproduciendo....")
            self.root.update_idletasks()
            sd.play(self.audio_data, fs)
            sd.wait()
            print("Reproducción finalizada.")
            self.status_label.config(text="Grabación finalizada.")
            self.root.update_idletasks()
        else:
            print("No hay audio grabado para reproducir.")
            self.status_label.config(text="No hay audio grabado para reproducir.")

    def play_audioFiltrado(self):
        if self.señal_Filtrada is not None:
            print("Reproduciendo...")
            self.status_label.config(text="Reproduciendo...")
            self.root.update_idletasks()
            sd.play(self.señal_Filtrada, fs)
            sd.wait()
            print("Reproducción finalizada.")
            self.status_label.config(text="Grabación finalizada.")
            self.root.update_idletasks()
        else:
            print("No hay audio grabado para reproducir.")
            self.status_label.config(text="No hay audio grabado para reproducir.")

    def show_spectrum(self):
        if self.audio_data is not None:

            L = len(self.audio_Principal)
            Ts = 1 / fs  # Ts debe ser 1/fs, no 2/fs
            # Crear vector tiempo
            t = Ts * np.arange(0, L)
            Hzs = np.fft.fftfreq(L, Ts)
            # Crear figura y ejes
            fig, axs = plt.subplots(2, figsize=(10, 8))

         
            
            # Gráfica del espectro de la señal de audio en el dominio de la frecuencia
            axs[0].plot(Hzs, np.abs(self.fft_audio))
            axs[0].set_xlabel("Frecuencia (Hz)")
            axs[0].set_ylabel("Amplitud")
            axs[0].set_title("Espectro de la señal de audio")
 
            axs[0].grid()
          
            
            axs[1].plot(Hzs, np.abs(self.fftSeñalfiltrada))
            axs[1].set_title("Espectro de la señal audio filtrada")
            axs[1].set_xlabel("Frecuencia (Hz)")
            axs[1].set_ylabel("Amplitud")
          
            axs[1].grid()
            axs[1].set_ylim(0, np.max(np.abs(self.fft_audio)))

            zone1 =axs[1].axvline(x=-fc2, color="red", label="Banda de rechazo")
            zone2 =axs[1].axvline(x=-fc1,color="red", label="Banda de rechazo")
            zone3 =axs[1].axvline(x=fc1,color="red", label="Banda de rechazo")
            zone4 =axs[1].axvline(x=fc2,color="red", label="Banda de rechazo")
            zone5 =axs[0].axvline(x=-fc2, color="red", label="Banda de rechazo")
            zone6 =axs[0].axvline(x=-fc1,color="red", label="Banda de rechazo")
            zone7 =axs[0].axvline(x=fc1,color="red", label="Banda de rechazo")
            zone8 =axs[0].axvline(x=fc2,color="red", label="Banda de rechazo")

            visibility_status = [False]
            check_ax = plt.axes([0.82, 0, 0.17, 0.03])
            check_buttons = CheckButtons(check_ax, ['Banda de rechazo'], visibility_status)

            def update_visibility(label):
                if label == 'Banda de rechazo':
                    visible = not zone1.get_visible()
                    zone1.set_visible(visible)
                    zone2.set_visible(visible)
                    zone3.set_visible(visible)
                    zone4.set_visible(visible)
                    zone5.set_visible(visible)
                    zone6.set_visible(visible)
                    zone7.set_visible(visible)
                    zone8.set_visible(visible)
                fig.canvas.draw()
            update_visibility('Banda de rechazo')
            check_buttons.on_clicked(update_visibility)
            fig.tight_layout()

            # Mostrar la figura con las dos gráficas
            plt.show()
        else:
            print("No hay audio grabado,es imposible mostrar graficas.")


if __name__ == "__main__":
    root = tk.Tk()
    app = AudioApp(root)
    root.mainloop()
