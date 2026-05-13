#!/usr/bin/env python3
"""
Reconocimiento de Comandos de Voz por Banco de Filtros (Energías de Subbandas)
===============================================================================
Sistema de dos etapas accesible desde una sola UI con pestañas:
  · Entrenamiento – graba muestras, extrae energías de 3 subbandas vía FFT,
                    calcula media y desviación estándar por comando.
  · Reconocimiento – captura el comando en tiempo real, compara sus energías
                     con los umbrales almacenados y muestra el resultado.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write as wav_write, read as wav_read
import os
import json
import threading

# ─── Constantes globales ───────────────────────────────────────────────────────
FS = 16000           # frecuencia de muestreo (Hz) – cubre voz 300-8 000 Hz
N_SUBBANDS = 3       # número de subbandas del banco de filtros
VOCAB = [            # diccionario de comandos de voz
    "arriba", "abajo", "izquierda", "derecha", "stop", "inicio"
]

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
GRAB_DIR    = os.path.join(BASE_DIR, "grabaciones")
CONFIG_FILE = os.path.join(BASE_DIR, "config.json")

os.makedirs(GRAB_DIR, exist_ok=True)


# ─── Procesamiento de señal ────────────────────────────────────────────────────

def normalizar(signal: np.ndarray) -> np.ndarray:
    """Normaliza la amplitud a [-1, 1]; evita división por cero."""
    m = np.max(np.abs(signal))
    return signal / m if m > 0 else signal


def energias_subbandas(signal: np.ndarray, n: int = N_SUBBANDS) -> np.ndarray:
    """
    Banco de filtros indirecto vía FFT.

    Divide el espectro de magnitud cuadrada en n subbandas de igual ancho y
    calcula la energía promedio de cada una:

        E_i = (1 / N_i) * Σ |X(k)|²    k ∈ subband_i

    Trabajar directamente sobre los coeficientes de la FFT equivale a pasar
    la señal por n filtros pasabanda digitales en paralelo; el coste es un
    único cómputo de FFT más n sumas parciales, lo que resulta eficiente para
    señales cortas de voz.
    """
    N = len(signal)
    X = np.fft.rfft(signal)                  # espectro unilateral (N//2 + 1 bins)
    power = (np.abs(X) ** 2) / N            # densidad de potencia espectral discreta

    half_N = len(power)
    bin_size = half_N // n
    energies = np.zeros(n)
    for i in range(n):
        start = i * bin_size
        end   = (i + 1) * bin_size if i < n - 1 else half_N
        energies[i] = np.mean(power[start:end])
    return energies


def procesar_señal(signal: np.ndarray) -> np.ndarray:
    """Pipeline completo: normaliza → calcula energías de subbandas."""
    sig = normalizar(signal.flatten().astype(np.float64))
    return energias_subbandas(sig)


# ─── Entrenamiento ─────────────────────────────────────────────────────────────

def entrenar_desde_directorio() -> dict:
    """
    Para cada palabra en GRAB_DIR carga todos los .wav, extrae el vector de
    energías de subbandas y calcula media y desviación estándar por subbanda:

        E_ci  = (1 / M) * Σ E_i^(m)     m = 1 … M grabaciones
        σ_ci  = desviación estándar de { E_i^(m) }

    Estos dos vectores [E_c1, E_c2, E_c3] y [σ_c1, σ_c2, σ_c3] constituyen
    los umbrales del banco para el comando c, y se persisten en config.json.
    """
    config = {}
    for palabra in VOCAB:
        carpeta = os.path.join(GRAB_DIR, palabra)
        if not os.path.isdir(carpeta):
            continue
        wavs = sorted(f for f in os.listdir(carpeta) if f.endswith(".wav"))
        if not wavs:
            continue

        mat = []
        for wav in wavs:
            try:
                fs_wav, data = wav_read(os.path.join(carpeta, wav))
                sig = data.astype(np.float64)
                if sig.ndim > 1:
                    sig = sig[:, 0]
                # Remuestreo lineal si la fs difiere de la objetivo
                if fs_wav != FS:
                    n_new = int(len(sig) * FS / fs_wav)
                    sig = np.interp(
                        np.linspace(0, len(sig), n_new),
                        np.arange(len(sig)), sig
                    )
                mat.append(procesar_señal(sig))
            except Exception:
                continue

        if mat:
            mat = np.array(mat)                  # shape (M, N_SUBBANDS)
            config[palabra] = {
                "media":      mat.mean(axis=0).tolist(),
                "std":        mat.std(axis=0).tolist(),
                "n_muestras": len(mat),
            }

    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=2)
    return config


def cargar_config() -> dict:
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE) as f:
            return json.load(f)
    return {}


# ─── Reconocimiento ────────────────────────────────────────────────────────────

def reconocer(signal: np.ndarray, config: dict) -> tuple:
    """
    Compara el vector de energías del comando entrante con los umbrales.

    Distancia normalizada (z-score acumulado):
        d_c = Σ_i  |E_i - E_ci| / (σ_ci + ε)

    La normalización por σ hace que las subbandas con mayor variabilidad
    pesen menos en la decisión, lo que mejora la robustez frente a la
    variabilidad inter-locutor. El comando reconocido es el de menor d_c.
    """
    if not config:
        return "Sin configuración", {}

    e_test = procesar_señal(signal)
    distancias = {}
    for palabra, params in config.items():
        media = np.array(params["media"])
        std   = np.array(params["std"])
        eps   = 1e-12
        d     = float(np.sum(np.abs(e_test - media) / (std + eps)))
        distancias[palabra] = d

    ganador = min(distancias, key=distancias.get)
    return ganador, distancias


# ─── Interfaz Gráfica ──────────────────────────────────────────────────────────

class AppRecoVoz(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Reconocimiento de Comandos de Voz")
        self.resizable(False, False)
        self._config = cargar_config()
        self._build_ui()

    # ── construcción de widgets ────────────────────────────────────────────────

    def _build_ui(self):
        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True, padx=10, pady=10)

        tab_train = ttk.Frame(nb)
        tab_reco  = ttk.Frame(nb)
        nb.add(tab_train, text="  Entrenamiento  ")
        nb.add(tab_reco,  text="  Reconocimiento  ")

        self._build_training_tab(tab_train)
        self._build_recognition_tab(tab_reco)

    def _build_training_tab(self, parent):
        pad = {"padx": 12, "pady": 6}

        ttk.Label(parent, text="Palabra (comando):").grid(
            row=0, column=0, **pad, sticky="e")
        self.palabra_var = tk.StringVar(value=VOCAB[0])
        cb = ttk.Combobox(parent, textvariable=self.palabra_var,
                          values=VOCAB, state="readonly", width=14)
        cb.grid(row=0, column=1, **pad, sticky="w")

        ttk.Label(parent, text="Duración (s):").grid(
            row=1, column=0, **pad, sticky="e")
        self.dur_train_var = tk.StringVar(value="2")
        ttk.Entry(parent, textvariable=self.dur_train_var, width=8).grid(
            row=1, column=1, **pad, sticky="w")

        self.btn_grabar = ttk.Button(
            parent, text="Grabar muestra",
            command=self._grabar_muestra)
        self.btn_grabar.grid(row=2, column=0, columnspan=2, pady=(8, 2))

        self.btn_entrenar = ttk.Button(
            parent, text="Procesar y entrenar",
            command=self._entrenar)
        self.btn_entrenar.grid(row=3, column=0, columnspan=2, pady=2)

        self.lbl_train_status = ttk.Label(
            parent, text="", foreground="#555", wraplength=300)
        self.lbl_train_status.grid(
            row=4, column=0, columnspan=2, padx=12, pady=6)

        ttk.Label(parent, text="Grabaciones por comando:").grid(
            row=5, column=0, columnspan=2, padx=12, pady=(6, 2))
        cols = ("Comando", "N° grabaciones")
        self.tree = ttk.Treeview(
            parent, columns=cols, show="headings", height=len(VOCAB))
        for c in cols:
            self.tree.heading(c, text=c)
            self.tree.column(c, width=140, anchor="center")
        self.tree.grid(row=6, column=0, columnspan=2, padx=12, pady=4)
        self._actualizar_tabla()

    def _build_recognition_tab(self, parent):
        pad = {"padx": 12, "pady": 6}

        ttk.Label(parent, text="Duración (s):").grid(
            row=0, column=0, **pad, sticky="e")
        self.dur_reco_var = tk.StringVar(value="2")
        ttk.Entry(parent, textvariable=self.dur_reco_var, width=8).grid(
            row=0, column=1, **pad, sticky="w")

        self.btn_reco = ttk.Button(
            parent, text="Grabar y reconocer",
            command=self._grabar_y_reconocer)
        self.btn_reco.grid(row=1, column=0, columnspan=2, pady=(8, 4))

        self.lbl_resultado = ttk.Label(
            parent, text="—",
            font=("Helvetica", 18, "bold"), foreground="#1a7abf")
        self.lbl_resultado.grid(row=2, column=0, columnspan=2, pady=10)

        self.lbl_reco_status = ttk.Label(parent, text="", foreground="#555")
        self.lbl_reco_status.grid(
            row=3, column=0, columnspan=2, padx=12, pady=(0, 4))

        ttk.Label(parent, text="Distancias a cada comando:").grid(
            row=4, column=0, columnspan=2, padx=12, pady=(6, 2))
        cols2 = ("Comando", "Distancia")
        self.tree_dist = ttk.Treeview(
            parent, columns=cols2, show="headings", height=len(VOCAB))
        for c in cols2:
            self.tree_dist.heading(c, text=c)
            self.tree_dist.column(c, width=140, anchor="center")
        self.tree_dist.grid(row=5, column=0, columnspan=2, padx=12, pady=4)

    # ── acciones ───────────────────────────────────────────────────────────────

    def _grabar_muestra(self):
        try:
            dur = float(self.dur_train_var.get())
            if dur <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Error", "Duración inválida.")
            return
        self.btn_grabar.config(state="disabled")
        threading.Thread(
            target=self._grabar_thread, args=(dur,), daemon=True).start()

    def _grabar_thread(self, dur: float):
        palabra = self.palabra_var.get()
        carpeta = os.path.join(GRAB_DIR, palabra)
        os.makedirs(carpeta, exist_ok=True)
        idx = len([f for f in os.listdir(carpeta) if f.endswith(".wav")]) + 1
        ruta = os.path.join(carpeta, f"{palabra}_{idx:03d}.wav")

        self.after(0, lambda: self.lbl_train_status.config(
            text=f"Grabando '{palabra}' ({dur} s)…"))
        audio = sd.rec(int(dur * FS), samplerate=FS,
                       channels=1, dtype="float32")
        sd.wait()
        wav_write(ruta, FS, audio)
        self.after(0, lambda: self.lbl_train_status.config(
            text=f"Guardado: {os.path.basename(ruta)}"))
        self.after(0, self._actualizar_tabla)
        self.after(0, lambda: self.btn_grabar.config(state="normal"))

    def _entrenar(self):
        self.btn_entrenar.config(state="disabled")
        self.lbl_train_status.config(text="Procesando grabaciones…")
        self.update_idletasks()
        threading.Thread(target=self._entrenar_thread, daemon=True).start()

    def _entrenar_thread(self):
        try:
            config = entrenar_desde_directorio()
            self._config = config
            resumen = "  |  ".join(
                f"{p}: {v['n_muestras']} muestras"
                for p, v in config.items()
            )
            self.after(0, lambda: self.lbl_train_status.config(
                text=f"Entrenamiento completado.\n{resumen}"))
        except Exception as exc:
            self.after(0, lambda: self.lbl_train_status.config(
                text=f"Error: {exc}"))
        finally:
            self.after(0, lambda: self.btn_entrenar.config(state="normal"))

    def _grabar_y_reconocer(self):
        if not self._config:
            messagebox.showwarning(
                "Sin configuración",
                "Primero graba muestras y entrena el sistema en la pestaña "
                "Entrenamiento.")
            return
        try:
            dur = float(self.dur_reco_var.get())
            if dur <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Error", "Duración inválida.")
            return
        self.btn_reco.config(state="disabled")
        self.lbl_resultado.config(text="…")
        self.lbl_reco_status.config(text="Grabando…")
        threading.Thread(
            target=self._reco_thread, args=(dur,), daemon=True).start()

    def _reco_thread(self, dur: float):
        audio = sd.rec(int(dur * FS), samplerate=FS,
                       channels=1, dtype="float32")
        sd.wait()
        signal = audio.flatten()
        ganador, distancias = reconocer(signal, self._config)

        def _update():
            self.lbl_resultado.config(
                text=f"Comando '{ganador}' Reconocido")
            self.lbl_reco_status.config(text="")
            for row in self.tree_dist.get_children():
                self.tree_dist.delete(row)
            for pal, dist in sorted(distancias.items(), key=lambda x: x[1]):
                tag = "ganador" if pal == ganador else ""
                self.tree_dist.insert(
                    "", "end",
                    values=(pal, f"{dist:.4f}"),
                    tags=(tag,))
            self.tree_dist.tag_configure(
                "ganador",
                foreground="#1a7abf",
                font=("Helvetica", 10, "bold"))
            self.btn_reco.config(state="normal")

        self.after(0, _update)

    # ── helpers ────────────────────────────────────────────────────────────────

    def _actualizar_tabla(self):
        for row in self.tree.get_children():
            self.tree.delete(row)
        for palabra in VOCAB:
            carpeta = os.path.join(GRAB_DIR, palabra)
            n = (len([f for f in os.listdir(carpeta) if f.endswith(".wav")])
                 if os.path.isdir(carpeta) else 0)
            self.tree.insert("", "end", values=(palabra, n))


if __name__ == "__main__":
    app = AppRecoVoz()
    app.mainloop()
