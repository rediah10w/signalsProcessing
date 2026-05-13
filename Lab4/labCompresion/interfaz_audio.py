import numpy as np
import tkinter as tk
from tkinter import messagebox
import sounddevice as sd
from scipy.fftpack import dct, idct

class InterfazCompresionAudio:
    """
    Clase que define la interfaz gráfica para la compresión de audio.
    """

    def __init__(self, root):
        """
        Constructor de la clase InterfazCompresionAudio.

        Parámetros:
            - root: La ventana principal de la interfaz.
        """
        self.root = root
        # self.root.title("Interfaz de Compresión de Audio")
        
        # Centrar todo el contenido en la ventana
        # self.root.grid_rowconfigure(0, weight=1)
        for i in range(8):
            self.root.grid_columnconfigure(i, weight=1)

        
        # Agregar etiqueta de título
        titulo_etiqueta = tk.Label(root, text="Compresión de Audio", font=("Helvetica", 16, "bold"))
        titulo_etiqueta.grid(row=0, pady=20, padx=20, columnspan=8, sticky="nswe")

        # Botón para iniciar la grabación
        self.boton_grabar = tk.Button(root, text="Iniciar Grabación", command=self.grabar_audio)
        self.boton_grabar.grid(row=1, column=0, columnspan=2, pady=10, padx=10, sticky="nswe")
        
        # Entrada para el porcentaje de compresión
        tk.Label(root, text="Porcentaje de Compresión:").grid(row=1, column=2, pady=10, padx=10, sticky="nswe")
        self.entrada_porcentaje_compresion = tk.Entry(root)
        self.entrada_porcentaje_compresion.grid(row=1, column=3, pady=10, padx=10, sticky="nswe")
        self.entrada_porcentaje_compresion.insert(0, "50")
        self.entrada_porcentaje_compresion.config(state=tk.DISABLED)
        
        # Botón para comprimir el audio
        self.boton_comprimir = tk.Button(root, text="Comprimir", command=self.comprimir_audio)
        self.boton_comprimir.grid(row=1, column=5, columnspan=3, pady=10, padx=10, sticky="nswe")
        self.boton_comprimir.config(state=tk.DISABLED)

        # Botones para reproducir el audio original, comprimido y descomprimido
        self.boton_reproducir_original = tk.Button(root, text="Reproducir Original", command=self.reproducir_original)
        self.boton_reproducir_original.grid(row=3, column=0, columnspan=2, pady=10, padx=10, sticky="nswe")
        self.boton_reproducir_original.config(state=tk.DISABLED)
        
        self.boton_reproducir_comprimido = tk.Button(root, text="Reproducir Comprimido", command=self.reproducir_comprimido)
        self.boton_reproducir_comprimido.grid(row=3, column=2, columnspan=3, pady=10, padx=10, sticky="nswe")
        self.boton_reproducir_comprimido.config(state=tk.DISABLED)
        
        self.boton_reproducir_descomprimido = tk.Button(root, text="Reproducir Descomprimido", command=self.reproducir_descomprimido)
        self.boton_reproducir_descomprimido.grid(row=3, column=5, columnspan=3, pady=10, padx=10, sticky="nswe")
        self.boton_reproducir_descomprimido.config(state=tk.DISABLED)
        
        self.audio_grabado = None
        self.audio_comprimido = None
        self.audio_descomprimido = None

    def grabar_audio(self):
        """
        Método para grabar audio utilizando la biblioteca sounddevice.
        """
        global duracion, fs
        duracion = 2  # Duración de la grabación en segundos
        fs = 44100  # Frecuencia de muestreo

        try:
            print("Grabando voz...")
            self.audio_grabado = sd.rec(int(duracion * fs), samplerate=fs, channels=1, dtype='float64')
            sd.wait()
            print("Grabación completada.")
            messagebox.showinfo("Grabación", "Grabación completada.")
            self.entrada_porcentaje_compresion.config(state=tk.NORMAL)
            self.boton_comprimir.config(state=tk.NORMAL)
        except Exception as e:
            messagebox.showerror("Error", f"Error al grabar el audio: {str(e)}")
            
    def normalizar_audio(self, datos):
        """
        Método para normalizar los datos de audio.

        Parámetros:
            - datos: Los datos de audio a normalizar.

        Devoluciones:
            - datos_normalizados: Los datos de audio normalizados.
        """
        max_valor = np.max(np.abs(datos))
        datos_normalizados = datos / max_valor
        return datos_normalizados

    def reproducir_original(self):
        """
        Método para reproducir el audio original.
        """
        if self.audio_grabado is not None:
            self.disable_buttons()
            audio_normalizado = self.normalizar_audio(self.audio_grabado)
            sd.play(audio_normalizado, fs)
            # sd.play(self.audio_grabado, fs)
            self.root.after(int(duracion * 1000), self.enable_buttons)
        else:
            messagebox.showwarning("Advertencia", "Primero grabe el audio.")

    def comprimir_audio(self):
        """
        Método para comprimir el audio grabado.
        """
        if self.audio_grabado is not None:
            try:
                porcentaje = int(self.entrada_porcentaje_compresion.get())
                if porcentaje < 0 or porcentaje > 100:
                    raise ValueError("El porcentaje debe estar entre 0 y 100.")
                
                posiciones_necesarias, resultado = self.calcular_porcentaje(self.audio_grabado, porcentaje)
                self.audio_comprimido = resultado
                messagebox.showinfo("Compresión", f"Se necesitaron {posiciones_necesarias} posiciones para alcanzar el {porcentaje}% de compresión.")
                
                self.enable_buttons()
            except ValueError as ve:
                messagebox.showerror("Error", str(ve))
            except Exception as e:
                messagebox.showerror("Error", f"Error al comprimir el audio: {str(e)}")
        else:
            messagebox.showwarning("Advertencia", "Primero grabe el audio.")

    def reproducir_comprimido(self):
        """
        Método para reproducir el audio comprimido.
        """
        if self.audio_comprimido is not None:
            self.disable_buttons()
            audio_normalizado = self.normalizar_audio(self.audio_comprimido)
            sd.play(audio_normalizado, fs)
            # sd.play(self.audio_comprimido, fs)
            self.root.after(int(duracion * 1000), self.enable_buttons)
        else:
            messagebox.showwarning("Advertencia", "Primero comprima el audio.")

    def reproducir_descomprimido(self):
        """
        Método para reproducir el audio descomprimido.
        """
        if self.audio_comprimido is not None:
            try:
                self.audio_descomprimido = idct(self.audio_comprimido, type=2)
                self.disable_buttons()
                audio_normalizado = self.normalizar_audio(self.audio_descomprimido)
                sd.play(audio_normalizado, fs)
                # sd.play(self.audio_descomprimido, fs)
                self.root.after(int(duracion * 1000), self.enable_buttons)
            except Exception as e:
                messagebox.showerror("Error", f"Error al descomprimir el audio: {str(e)}")
        else:
            messagebox.showwarning("Advertencia", "Primero comprima el audio.")

    @staticmethod
    def calcular_porcentaje(datos, porcentaje):
        """
        Método estático para calcular el porcentaje de compresión de los datos de audio.

        Parámetros:
            - datos: Los datos de audio a comprimir.
            - porcentaje: El porcentaje de compresión deseado.

        Devoluciones:
            - posiciones_necesarias: El número de posiciones necesarias para alcanzar el porcentaje de compresión.
            - datos_modificados: Los datos de audio comprimidos.
        """
        total_elementos = np.prod(datos.shape)
        posiciones_necesarias = int(total_elementos * (porcentaje / 100))

        indices_menores = np.argsort(np.abs(datos), axis=None)
        datos_modificados = datos.flatten()
        datos_modificados[indices_menores[:posiciones_necesarias]] = 0
        datos_modificados = datos_modificados.reshape(datos.shape)

        return posiciones_necesarias, datos_modificados

    def disable_buttons(self):
        """
        Método para deshabilitar los botones mientras se reproduce el audio.
        """
        self.boton_grabar.config(state=tk.DISABLED)
        self.boton_comprimir.config(state=tk.DISABLED)
        self.boton_reproducir_original.config(state=tk.DISABLED)
        self.boton_reproducir_comprimido.config(state=tk.DISABLED)
        self.boton_reproducir_descomprimido.config(state=tk.DISABLED)

    def enable_buttons(self):
        """
        Método para habilitar los botones después de que finalice la reproducción del audio.
        """
        self.boton_grabar.config(state=tk.NORMAL)
        self.boton_comprimir.config(state=tk.NORMAL)
        self.boton_reproducir_original.config(state=tk.NORMAL)
        self.boton_reproducir_comprimido.config(state=tk.NORMAL)
        self.boton_reproducir_descomprimido.config(state=tk.NORMAL)
