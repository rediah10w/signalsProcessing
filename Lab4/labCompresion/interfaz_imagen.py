import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
from scipy.fftpack import dct, idct

class AplicacionCompresionImagen:
    def __init__(self, root):
        self.root = root
        
        # Centrar todo el contenido en la ventana
        # self.root.grid_rowconfigure(0, weight=1)
        # self.root.grid_columnconfigure(0, weight=1)
        for i in range(8):
            self.root.grid_columnconfigure(i, weight=1)

        
        # self.root.title("Compresión y Descompresión de Imágenes")
        
        # Etiqueta de título
        self.titulo_etiqueta = tk.Label(root, text="Compresión de Imágenes", font=("Helvetica", 16, "bold"))
        self.titulo_etiqueta.grid(row=0, pady=20, padx=20, columnspan=8, sticky="nswe")
        
        # Botón para cargar la imagen
        self.boton_cargar_imagen = tk.Button(root, text="Cargar Imagen", command=self.cargar_imagen)
        self.boton_cargar_imagen.grid(row=1, column=0, columnspan=3, pady=10, padx=10, sticky="nswe")
        
        # Entrada para ingresar el porcentaje de compresión
        self.etiqueta_porcentaje_compresion = tk.Label(root, text="Porcentaje de Compresión:")
        self.etiqueta_porcentaje_compresion.grid(row=1, column=3, pady=10, padx=10, sticky="nswe")
        self.entrada_porcentaje_compresion = tk.Entry(root)
        self.entrada_porcentaje_compresion.grid(row=1, column=4, pady=10, padx=10, sticky="nswe")
        self.entrada_porcentaje_compresion.insert(0, "15")  # Valor predeterminado
        self.entrada_porcentaje_compresion.config(state=tk.DISABLED)
        
        
        # Botón para comprimir la imagen
        self.boton_comprimir = tk.Button(root, text="Comprimir", command=self.comprimir_imagen)
        self.boton_comprimir.grid(row=1, column=5, columnspan=3, pady=10, padx=10, sticky="nswe")
        self.boton_comprimir.config(state=tk.DISABLED)
        
        
        # Frame para mostrar las imágenes y títulos
        self.marco_imagen = tk.Frame(root)
        self.marco_imagen.grid(row=2, column=0, padx=10, pady=10, columnspan=8, sticky="nswe")
        
        # Imagen Original
        self.marco_original = tk.Frame(self.marco_imagen)
        self.marco_original.grid(row=0, column=0, padx=10, pady=10, sticky="nswe")
        self.etiqueta_imagen_original = tk.Label(self.marco_original)
        self.etiqueta_imagen_original.grid(row=0, column=0)
        self.etiqueta_titulo_original = tk.Label(self.marco_original)
        self.etiqueta_titulo_original.grid(row=1, column=0)
        
    
        
        # Imagen Descomprimida
        self.marco_descomprimida = tk.Frame(self.marco_imagen)
        self.marco_descomprimida.grid(row=0, column=2, padx=10, pady=10, sticky="nswe")
        self.etiqueta_imagen_descomprimida = tk.Label(self.marco_descomprimida)
        self.etiqueta_imagen_descomprimida.grid(row=0, column=0)
        self.etiqueta_titulo_descomprimida = tk.Label(self.marco_descomprimida)
        self.etiqueta_titulo_descomprimida.grid(row=1, column=0)
        
   
    
    def calcular_porcentaje(self, datos, porcentaje):
        """
        Calcula el número de elementos a eliminar para alcanzar un cierto porcentaje de compresión.
        
        Parámetros:
            datos (numpy.ndarray): Matriz de datos de la imagen.
            porcentaje (int): Porcentaje de compresión deseado.
        
        Devoluciones:
            tuple: Un tuple que contiene el número de elementos eliminados y los datos modificados.
        """
        if isinstance(datos, list):
            datos = np.array(datos)
        if isinstance(datos, np.ndarray):
            elementos_totales = np.prod(datos.shape)
            posiciones_necesarias = int(elementos_totales * (porcentaje / 100))
            indices_menores = np.argsort(np.abs(datos), axis=None)
            datos_modificados = datos.flatten()
            datos_modificados[indices_menores[:posiciones_necesarias]] = 0
            datos_modificados = datos_modificados.reshape(datos.shape)
            return posiciones_necesarias, datos_modificados
        else:
            raise ValueError("El tipo de datos proporcionado no es compatible.")
    
    def cargar_imagen(self):
        """
        Abre un cuadro de diálogo para cargar una imagen y la muestra en la interfaz.
        """
        try:
            self.ruta_archivo = filedialog.askopenfilename(filetypes=[("Archivos de imagen", "*.png;*.jpg;*.jpeg;*.bmp;*.gif")])
            if self.ruta_archivo:
                self.imagen = cv2.imread(self.ruta_archivo, cv2.IMREAD_GRAYSCALE)
                self.entrada_porcentaje_compresion.config(state=tk.NORMAL)
                self.boton_comprimir.config(state=tk.NORMAL)
                
            else:
                messagebox.showwarning("Advertencia", "Por favor, carga una imagen antes de comprimirla.")
        except Exception as e:
            messagebox.showerror("Error", f"Error al cargar la imagen: {str(e)}")
    
    def comprimir_imagen(self):
        """
        Comprime la imagen cargada con el porcentaje especificado.
        """
        if hasattr(self, 'imagen'):
            try:
                porcentaje_compresion = int(self.entrada_porcentaje_compresion.get())
                
                if porcentaje_compresion < 0 or porcentaje_compresion > 100:
                    raise ValueError("El porcentaje debe estar entre 0 y 100.")
                  
                dct_tipo2 = dct(self.imagen, type=2, norm='ortho')
                
                posiciones_necesarias, dct_comprimida = self.calcular_porcentaje(dct_tipo2, porcentaje_compresion)
                
                img_comprimida = (dct_comprimida * 255).astype(np.uint8)
                
                idct_tipo2 = idct(dct_comprimida, type=2, norm='ortho')     
                self.mostrar_imagen_descomprimida(idct_tipo2)
                
                imagen_float32 = np.float32(self.imagen)
                img_descomprimida_float32 = np.float32(idct_tipo2)
                img_destacada_descomprimida = np.uint8(cv2.addWeighted(imagen_float32, 0.5, img_descomprimida_float32, 0.5, 0.5))
                
                
                messagebox.showinfo("Información", "Imagen comprimida y descomprimida correctamente.")
            except ValueError as ve:
                messagebox.showerror("Error", str(ve))
            except Exception as e:
                messagebox.showerror("Error", f"Error al comprimir la imagen: {str(e)}")
        else:
            messagebox.showwarning("Advertencia", "Por favor, carga una imagen antes de comprimirla.")
    

    def mostrar_imagen_original(self):
        """
        Muestra la imagen original en la interfaz.
        """
        imagen_original_pil = Image.fromarray(self.imagen)
        imagen_original_pil = self.escalar_imagen(imagen_original_pil, 300, 460)
        img_original_tk = ImageTk.PhotoImage(image=imagen_original_pil)
        self.etiqueta_imagen_original.config(image=img_original_tk)
        self.etiqueta_imagen_original.image = img_original_tk
        self.etiqueta_titulo_original.config(text="Original")

    def mostrar_imagen_comprimida(self, porcentaje_compresion, posiciones_necesarias, img_comprimida):
        """
        Muestra la imagen comprimida en la interfaz.
        
        Parámetros:
            porcentaje_compresion (int): Porcentaje de compresión aplicado.
            posiciones_necesarias (int): Número de posiciones eliminadas para la compresión.
            img_comprimida (numpy.ndarray): Imagen comprimida.
        """
        img_comprimida_pil = Image.fromarray(img_comprimida)
        img_comprimida_pil = self.escalar_imagen(img_comprimida_pil, 300, 460)
        img_comprimida_tk = ImageTk.PhotoImage(image=img_comprimida_pil)
        self.etiqueta_imagen_comprimida.config(image=img_comprimida_tk)
        self.etiqueta_imagen_comprimida.image = img_comprimida_tk
        self.etiqueta_titulo_comprimida.config(text=f"Comprimida al {porcentaje_compresion}%\n con {posiciones_necesarias} elementos eliminados")

    def mostrar_imagen_descomprimida(self, idct_tipo2):
        """
        Muestra la imagen descomprimida en la interfaz.
        
        Parámetros:
            idct_tipo2 (numpy.ndarray): Imagen descomprimida.
        """
        img_descomprimida = np.clip(idct_tipo2, 0, 255).astype(np.uint8)
        img_descomprimida_pil = Image.fromarray(img_descomprimida)
        img_descomprimida_pil = self.escalar_imagen(img_descomprimida_pil, 300, 460)
        img_descomprimida_tk = ImageTk.PhotoImage(image=img_descomprimida_pil)
        self.etiqueta_imagen_descomprimida.config(image=img_descomprimida_tk)
        self.etiqueta_imagen_descomprimida.image = img_descomprimida_tk
        self.etiqueta_titulo_descomprimida.config(text="Descomprimida")

    def mostrar_imagen_descomprimida_resaltada(self, img_destacada_descomprimida):
        """
        Muestra la imagen descomprimida resaltando los píxeles comprimidos en la interfaz.
        
        Parámetros:
            img_destacada_descomprimida (numpy.ndarray): Imagen descomprimida con píxeles comprimidos resaltados.
        """
        img_descomprimida_resaltada_pil = Image.fromarray(img_destacada_descomprimida)
        img_descomprimida_resaltada_pil = self.escalar_imagen(img_descomprimida_resaltada_pil, 300, 460)
        img_descomprimida_resaltada_tk = ImageTk.PhotoImage(image=img_descomprimida_resaltada_pil)
        self.etiqueta_imagen_descomprimida_resaltada.config(image=img_descomprimida_resaltada_tk)
        self.etiqueta_imagen_descomprimida_resaltada.image = img_descomprimida_resaltada_tk
        self.etiqueta_titulo_descomprimida_resaltada.config(text="Descomprimida resaltando los píxeles comprimidos")
        
    def escalar_imagen(self, image, max_width, max_height):
        width, height = image.size
        # Calcula la proporción para redimensionar la imagen
        ratio_width = max_width / width
        ratio_height = max_height / height
        # Elige la proporción más pequeña para no exceder el tamaño del marco
        ratio = min(ratio_width, ratio_height)
        new_width = int(width * ratio)
        new_height = int(height * ratio)
        return image.resize((new_width, new_height))