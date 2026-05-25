import tkinter as tk
from tkinter import ttk
import sounddevice as sd
import wavio
import numpy as np
import matplotlib.pyplot as plt
import os
import threading
import time
from matplotlib.widgets import CheckButtons
fs = 44100  
fc1=1500
fc2=4000

def ecuacion_diferencia(x):
    y = np.zeros_like(x)
    for n in range(len(x)):
         y[n] = (
            (0.02517611) * x[n]
            -(0.05035223) * x[n - 2]
            + (0.02517611) * x[n - 4]
            - (0.60439980) * y[n - 4]
            + (2.54723148) * y[n - 3]
            - (4.24459577) * y[n - 2]
            + (3.29022663) * y[n - 1]
        )
    return y


class AudioApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Grabadora de Audio")
        self.root.geometry("300x380")
        self.root.grid_columnconfigure(0, weight=1)
        self.duration = tk.DoubleVar(value=6.0)
        self.audio_data = None
        self.audio_Principal = None
        self.señal_Filtrada = None
        self._recording_done = False
        self._progress_after_id = None
        self._recording_start_time = None
        self._current_duration = 2.0
        self.create_widgets()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def create_widgets(self):
        # Título y frecuencias
        self.info_frame = ttk.Frame(self.root)
        self.info_frame.grid(column=0, row=0, columnspan=2, padx=10, pady=15)
        
        title_label = ttk.Label(self.info_frame, text="Filtro Pasabandas", 
                               font=("Arial", 16, "bold"))
        title_label.pack()
        
        freq_label = ttk.Label(self.info_frame, text="Frecuencia 1: 1500 Hz\nFrecuencia 2: 4000 Hz",
                              font=("Arial", 10), justify="center")
        freq_label.pack(pady=10)

        self.grabar_button = tk.Button(
            self.root, text="Grabar", command=self.show_duration_input,
            bg="red", fg="white", font=("Arial", 12, "bold"), padx=20, pady=10
        )
        self.grabar_button.grid(column=0, row=1, columnspan=2, padx=10, pady=10)

        self.duration_frame = ttk.Frame(self.root)
        ttk.Label(self.duration_frame, text="Duración (s):").grid(column=0, row=0, padx=5, pady=5)
        self.duration_entry = ttk.Entry(self.duration_frame, textvariable=self.duration, width=8)
        self.duration_entry.grid(column=1, row=1, padx=5, pady=5)
        self.confirm_button = ttk.Button(
            self.duration_frame, text="Iniciar grabación", command=self.record_audio
        )
        self.confirm_button.grid(column=2, row=0, padx=5, pady=5)

        self.progress_canvas = tk.Canvas(
            self.root, width=100, height=100, highlightthickness=0
        )
        self.progress_canvas.configure(bg=self.root.cget("bg"))


        self.status_label = ttk.Label(self.root, text="")
        self.status_label.grid(column=0, row=3, columnspan=2, padx=10, pady=2)

        self.play_original_button = ttk.Button(
            self.root, text="Reproducir Audio Original", command=self.play_audio
        )
        self.play_original_button.grid(column=0, row=4, columnspan=2, padx=10, pady=10)
        self.play_original_button.grid_remove()

        self.play_filtered_button = ttk.Button(
            self.root, text="Reproducir Audio filtrado", command=self.play_audioFiltrado
        )
        self.play_filtered_button.grid(column=0, row=5, columnspan=2, padx=10, pady=10)
        self.play_filtered_button.grid_remove()

        self.spectrum_button = ttk.Button(
            self.root, text="Visualizar espectro de frecuencias", command=self.show_spectrum
        )
        self.spectrum_button.grid(column=0, row=6, columnspan=2, padx=10, pady=10)
        self.spectrum_button.grid_remove()

        self.amplitude_button = ttk.Button(
            self.root, text="Visualizar espectro de amplitud", command=self.show_amplitude_spectrum
        )
        self.amplitude_button.grid(column=0, row=7, columnspan=2, padx=10, pady=10)
        self.amplitude_button.grid_remove()

    def show_duration_input(self):
        self.grabar_button.config(state=tk.DISABLED)
        self.duration_frame.grid(column=0, row=1, columnspan=2, padx=10, pady=5)
        self.duration_entry.focus()

    def _draw_progress(self, fraction):
        self.progress_canvas.delete("all")
        cx, cy, r = 50, 50, 36

        self.progress_canvas.create_oval(
            cx - r, cy - r, cx + r, cy + r,
            outline="#d9d9d9", width=10
        )

        extent = -fraction * 360
        if abs(extent) > 0.5:
            self.progress_canvas.create_arc(
                cx - r, cy - r, cx + r, cy + r,
                start=90,
                extent=extent,
                outline="#4a9eff",
                width=10,
                style=tk.ARC,
            )
   
        pct = int(fraction * 100)
        self.progress_canvas.create_text(
            cx, cy, text=f"{pct}%", font=("Arial", 11, "bold"), fill="#333333"
        )

    def _animate_progress_loop(self):
        if self._recording_start_time is None:
            return
        elapsed = time.time() - self._recording_start_time
        fraction = min(elapsed / self._current_duration, 1.0)
        self._draw_progress(fraction)
        if not self._recording_done:
            self._progress_after_id = self.root.after(50, self._animate_progress_loop)
        else:
            self._draw_progress(1.0)
            self.root.after(400, self._finish_recording_ui)

    def record_audio(self):
        self._current_duration = self.duration.get()

        self.duration_frame.grid_remove()
        self.progress_canvas.grid(column=0, row=1, columnspan=2, padx=10, pady=10)
        self._draw_progress(0.0)
        self.status_label.config(text="Grabando señal...")

        self._recording_done = False
        self._recording_start_time = time.time()

        thread = threading.Thread(target=self._do_recording, args=(self._current_duration,), daemon=True)
        thread.start()
        self._animate_progress_loop()

    def _do_recording(self, duration):
        print(f"Grabando por {duration} segundos...")
        self.audio_data = sd.rec(
            int(duration * fs),
            samplerate=fs,
            channels=1,
            dtype="float32",
        )
        sd.wait()

        self.directorio_actual = os.path.dirname(os.path.abspath(__file__))
        ruta_archivo = os.path.join(self.directorio_actual, "Audio Original.wav")
        wavio.write(ruta_archivo, self.audio_data, fs, sampwidth=2)

        self.audio_Principal = self.audio_data.flatten()
        self.fft_audio = np.fft.fft(self.audio_Principal)
        self.señal_Filtrada = ecuacion_diferencia(self.audio_Principal)
        self.fftSeñalfiltrada = np.fft.fft(self.señal_Filtrada)

        ruta_archivo = os.path.join(self.directorio_actual, "Audio Filtrado.wav")
        wavio.write(ruta_archivo, self.señal_Filtrada, fs, sampwidth=2)

        self._recording_done = True
        print("Grabación finalizada.")

    def _finish_recording_ui(self):
        self.progress_canvas.grid_remove()
        self.grabar_button.config(state=tk.NORMAL)
        self.status_label.config(text="Grabación finalizada.")
        self._recording_start_time = None
        self.info_frame.grid_remove()
        if self.audio_Principal is not None and self.señal_Filtrada is not None:
            self.play_original_button.grid()
            self.play_filtered_button.grid()
            self.spectrum_button.grid()
            self.amplitude_button.grid()

    def _on_close(self):
        if self._progress_after_id is not None:
            self.root.after_cancel(self._progress_after_id)
        self.root.destroy()

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

    def show_amplitude_spectrum(self):
        if self.audio_Principal is not None and self.señal_Filtrada is not None:
            L = len(self.audio_Principal)
            Ts = 1 / fs
            t = Ts * np.arange(0, L)

            fig, axs = plt.subplots(2, figsize=(10, 8))

            axs[0].plot(t, self.audio_Principal)
            axs[0].set_xlabel("Tiempo (s)")
            axs[0].set_ylabel("Amplitud")
            axs[0].set_title("Amplitud de la señal de audio original")
            axs[0].grid()

            axs[1].plot(t, self.señal_Filtrada)
            axs[1].set_xlabel("Tiempo (s)")
            axs[1].set_ylabel("Amplitud")
            axs[1].set_title("Amplitud de la señal de audio filtrada")
            axs[1].grid()

            fig.tight_layout()
            plt.show()
        else:
            print("No hay audio grabado, es imposible mostrar la gráfica de amplitud.")

    def show_spectrum(self):
        if self.audio_data is not None:

            L = len(self.audio_Principal)
            Ts = 1 / fs  

            t = Ts * np.arange(0, L)
            Hzs = np.fft.fftfreq(L, Ts)
    
            fig, axs = plt.subplots(2, figsize=(10, 8))


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

            plt.show()
        else:
            print("No hay audio grabado,es imposible mostrar graficas.")


if __name__ == "__main__":
    root = tk.Tk()
    app = AudioApp(root)
    root.mainloop()
