import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
import tkinter as tk
from tkinter import ttk, messagebox
import threading
import time


FS         = 44100  
CHANNELS   = 1      
DTYPE      = 'float32'


class AppFFT(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Grabación + FFT")
        self.resizable(False, False)
        self._audio_data: np.ndarray | None = None   # guardado tras grabar
        self._build_ui()


    def _build_ui(self):
        pad = {"padx": 14, "pady": 6}


        ttk.Label(self, text="Duración de grabación (s):").grid(
            row=1, column=0, **pad, sticky="e")
        self.dur_var = tk.StringVar(value="3")
        self.dur_entry = ttk.Entry(self, textvariable=self.dur_var, width=8)
        self.dur_entry.grid(row=1, column=1, **pad, sticky="w")


        self.canvas_size = 140
        self.canvas = tk.Canvas(self, width=self.canvas_size,
                                height=self.canvas_size, bg="#f0f0f0",
                                highlightthickness=0)
        self.canvas.grid(row=2, column=0, columnspan=2, pady=(6, 0))
        self._draw_ring(0)

        self.status_lbl = ttk.Label(self, text="Listo", foreground="#555")
        self.status_lbl.grid(row=3, column=0, columnspan=2, pady=(0, 6))


        self.btn = ttk.Button(self, text="Iniciar grabación",
                              command=self._start_recording)
        self.btn.grid(row=4, column=0, pady=(0, 14), sticky="e", padx=(0, 6))

        self.play_btn = ttk.Button(self, text="▶", width=3,
                                   command=self._play_audio,
                                   state="disabled")
        self.play_btn.grid(row=4, column=1, pady=(0, 14), sticky="w", padx=(6, 0))


    def _draw_ring(self, progress: float):
        """Redibuja el anillo; progress ∈ [0, 1]."""
        self.canvas.delete("all")
        m = 12                         
        s = self.canvas_size
   
        self.canvas.create_arc(m, m, s - m, s - m,
                               start=90, extent=-360,
                               outline="#d0d0d0", width=10, style="arc")
  
        if progress > 0:
            self.canvas.create_arc(m, m, s - m, s - m,
                                   start=90, extent=-360 * progress,
                                   outline="#1a7abf", width=10, style="arc")

        pct = int(progress * 100)
        self.canvas.create_text(s // 2, s // 2,
                                text=f"{pct}%",
                                font=("Helvetica", 18, "bold"),
                                fill="#1a7abf")

  
    def _start_recording(self):
        try:
            duration = float(self.dur_var.get())
            if duration <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Error", "Ingresa una duración válida (> 0 s).")
            return

        self.btn.config(state="disabled")
        self.play_btn.config(state="disabled")
        self.dur_entry.config(state="disabled")
        self._audio_data = None
        self._draw_ring(0)
        self.status_lbl.config(text="Preparando…")

        threading.Thread(target=self._record_and_fft,
                         args=(duration,), daemon=True).start()

    def _record_and_fft(self, duration: float):
        n_samples = int(duration * FS)

        self.after(0, lambda: self.status_lbl.config(text="Grabando…"))
        recording = sd.rec(n_samples, samplerate=FS,
                           channels=CHANNELS, dtype=DTYPE)

     
        start = time.time()
        while True:
            elapsed = time.time() - start
            progress = min(elapsed / duration, 1.0)
            remaining = max(duration - elapsed, 0)
            self.after(0, self._draw_ring, progress)
            self.after(0, lambda r=remaining:
                       self.status_lbl.config(
                           text=f"Grabando… {r:.1f} s restantes"))
            if progress >= 1.0:
                break
            time.sleep(0.05)

        sd.wait()

        audio_flat = recording.flatten()
        self.after(0, self._draw_ring, 1.0)
        self.after(0, lambda: self.status_lbl.config(text="Calculando FFT…"))
        self.after(100, lambda: self._apply_fft(audio_flat, duration))

   
    def _apply_fft(self, signal: np.ndarray, duration: float):

        self._audio_data = signal

        N  = len(signal)
        Ts = 1.0 / FS

        fft_result = np.fft.fft(signal)
        magnitudes = np.abs(fft_result)

        freqs = np.fft.fftfreq(N, Ts)

        pos_mask  = freqs >= 0
        freqs_pos = freqs[pos_mask]
        mags_pos  = magnitudes[pos_mask]

        fig, axes = plt.subplots(2, 1, figsize=(10, 7))
        fig.suptitle(f"FFT", fontsize=13)

        fig.canvas.mpl_connect("close_event", self._on_plot_close)

        t = np.arange(N) * Ts

        axes[0].plot(t, signal, color="#2c7bb6", linewidth=0.7)
        axes[0].set_title("Señal grabada")
        axes[0].set_xlabel("Tiempo [s]")
        axes[0].set_ylabel("Amplitud")
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(freqs_pos, mags_pos, color="#97087d", linewidth=0.8)
        axes[1].set_title("Espectro de magnitud FFT")
        axes[1].set_xlabel("Frecuencia [Hz]")
        axes[1].set_ylabel("|FFT|")
        axes[1].set_xlim(0, FS / 2)
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        self.status_lbl.config(text="Mostrando gráfica…")
        self.btn.config(state="normal")
        self.dur_entry.config(state="normal")
        self.play_btn.config(state="normal")
        self._draw_ring(0)

        plt.show(block=False)

    def _on_plot_close(self, event):
        """Descarta el audio grabado al cerrar la ventana de matplotlib."""
        self._audio_data = None
        self.play_btn.config(state="disabled")
        self.status_lbl.config(text="Listo")

    def _play_audio(self):
        if self._audio_data is None:
            return
        threading.Thread(target=self._playback_thread, daemon=True).start()

    def _playback_thread(self):
        self.after(0, lambda: self.play_btn.config(state="disabled"))
        self.after(0, lambda: self.status_lbl.config(text="Reproduciendo…"))
        sd.play(self._audio_data, samplerate=FS)
        sd.wait()
        # Solo restaurar si el audio no fue descartado mientras se reproducía
        if self._audio_data is not None:
            self.after(0, lambda: self.play_btn.config(state="normal"))
            self.after(0, lambda: self.status_lbl.config(text="Mostrando gráfica…"))


if __name__ == "__main__":
    app = AppFFT()
    app.mainloop()
