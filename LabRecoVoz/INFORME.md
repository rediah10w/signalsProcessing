# Reconocimiento de Comandos de Voz por Banco de Filtros

## Resumen

Se diseña e implementa un sistema de reconocimiento de comandos de voz que opera en dos etapas —entrenamiento y reconocimiento en tiempo real— dentro de una única interfaz gráfica con pestañas. El núcleo matemático es un banco de filtros indirecto construido sobre la FFT, que divide el espectro de la señal en tres subbandas y calcula la energía promedio de cada una para caracterizar cada comando.

## Marco Teórico

### Banco de filtros indirecto via FFT

El fundamento del sistema parte de que dos comandos de voz distintos exhiben distribuciones de energía espectral diferentes. Si se calcula la FFT de una señal y se divide el espectro en $n$ subbandas de igual ancho $BW/n$, la energía de cada subbanda:

$$E_i = \frac{1}{N_i} \sum_{k \in \text{subbanda}_i} \left| X(k) \right|^2$$

forma un vector de características $\mathbf{e} = [E_1,\, E_2,\, E_3]$ que resulta distintivo para cada comando. Trabajar directamente sobre los coeficientes de la FFT equivale a pasar la señal por $n$ filtros digitales pasabanda en paralelo, pero con un único cómputo de transformada.

### Etapa de entrenamiento

Para cada comando $c$ se graban $M$ muestras. De cada muestra se extrae el vector $\mathbf{e}^{(m)}$ y se promedian los resultados:

$$\bar{E}_{ci} = \frac{1}{M} \sum_{m=1}^{M} E_i^{(m)}, \qquad \sigma_{ci} = \text{std}\!\left\{E_i^{(m)}\right\}$$

Estos parámetros $[\bar{E}_{c1},\, \bar{E}_{c2},\, \bar{E}_{c3}]$ y $[\sigma_{c1},\, \sigma_{c2},\, \sigma_{c3}]$ se guardan en `config.json` y constituyen los umbrales del banco.

### Etapa de reconocimiento

Ante una señal de prueba con vector $\mathbf{e}_{\text{test}}$, se calcula la distancia normalizada (ponderada por la desviación estándar) respecto a cada comando:

$$d_c = \sum_{i=1}^{3} \frac{\left| e_{\text{test},i} - \bar{E}_{ci} \right|}{\sigma_{ci} + \varepsilon}$$

La normalización por $\sigma_{ci}$ hace que las subbandas con mayor variabilidad inter-grabación pesen menos en la decisión, lo que mejora la robustez frente a cambios de locutor y condiciones de ruido. El comando reconocido es el de menor $d_c$.

## Implementación

### Estructura del sistema

El proyecto vive en `LabRecoVoz/` con un único script `reconocimiento_voz.py` y un directorio `grabaciones/` generado automáticamente al grabar. Los parámetros del entrenamiento se persisten en `config.json`.

Se optó por una sola interfaz con dos pestañas en lugar de dos scripts independientes porque permite compartir el estado de configuración en memoria y ofrece una experiencia de uso más cohesionada para una demostración pública.

### Procesamiento de la señal

La función central del sistema es `energias_subbandas`, que construye el banco de filtros indirecto:

```python
def energias_subbandas(signal: np.ndarray, n: int = N_SUBBANDS) -> np.ndarray:
    N = len(signal)
    X = np.fft.rfft(signal)
    power = (np.abs(X) ** 2) / N

    half_N = len(power)
    bin_size = half_N // n
    energies = np.zeros(n)
    for i in range(n):
        start = i * bin_size
        end   = (i + 1) * bin_size if i < n - 1 else half_N
        energies[i] = np.mean(power[start:end])
    return energies
```

`np.fft.rfft` devuelve solo el espectro unilateral (bins de 0 a $f_s/2$), eliminando la redundancia de la simetría conjugada para señales reales. La división por $N$ normaliza la potencia espectral a unidades independientes de la longitud de la señal.

### Entrenamiento

```python
def entrenar_desde_directorio() -> dict:
    config = {}
    for palabra in VOCAB:
        carpeta = os.path.join(GRAB_DIR, palabra)
        wavs = sorted(f for f in os.listdir(carpeta) if f.endswith(".wav"))
        mat = []
        for wav in wavs:
            fs_wav, data = wav_read(os.path.join(carpeta, wav))
            sig = data.astype(np.float64)
            # Remuestreo lineal si la fs difiere de la objetivo
            if fs_wav != FS:
                n_new = int(len(sig) * FS / fs_wav)
                sig = np.interp(np.linspace(0, len(sig), n_new),
                                np.arange(len(sig)), sig)
            mat.append(procesar_señal(sig))
        mat = np.array(mat)
        config[palabra] = {
            "media":      mat.mean(axis=0).tolist(),
            "std":        mat.std(axis=0).tolist(),
            "n_muestras": len(mat),
        }
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=2)
    return config
```

El remuestreo lineal garantiza que grabaciones tomadas con distintas tarjetas de audio (que podrían diferir en frecuencia de muestreo) sean procesadas de manera uniforme antes de la extracción de características.

### Reconocimiento

```python
def reconocer(signal: np.ndarray, config: dict) -> tuple:
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
```

El parámetro `eps = 1e-12` evita la división por cero en subbandas donde todas las grabaciones tienen energía nula (por ejemplo, porque el comando carece de componentes en frecuencias altas).

### Interfaz gráfica

La UI usa `tkinter.ttk.Notebook` para separar visualmente las dos etapas sin duplicar código. Toda interacción con el micrófono se delega a hilos secundarios (`threading.Thread`) para mantener la interfaz responsiva durante las grabaciones. Las actualizaciones de los widgets desde esos hilos se redirigen al hilo principal con `self.after(0, callback)`, patrón recomendado por tkinter para evitar condiciones de carrera.

La pestaña de reconocimiento muestra una tabla de distancias ordenada de menor a mayor, con el comando ganador resaltado en azul, lo que facilita la interpretación de la confianza del sistema.

## Decisiones Técnicas Relevantes

La frecuencia de muestreo se fijó en 16 000 Hz (en lugar de los 44 100 Hz habituales) porque la voz humana relevante para reconocimiento de comandos se concentra por debajo de 8 000 Hz; reducir $f_s$ a la mitad disminuye a su vez el coste computacional de la FFT y el tamaño de los archivos WAV sin sacrificar información útil.

Se eligió la distancia z-score acumulada —y no la distancia euclidiana simple— porque la energía de cada subbanda tiene una escala y variabilidad distintas según el tipo de señal. Ponderar por la desviación estándar hace la métrica invariante a la escala y le otorga mayor robustez ante la variabilidad entre locutores.

La normalización de amplitud a $[-1, 1]$ antes de calcular la FFT garantiza que variaciones en el volumen de grabación no alteren el vector de características: lo que importa es la distribución relativa de energía entre subbandas, no su magnitud absoluta.

## Instrucciones de uso

Para ejecutar el sistema:

```bash
pip install numpy scipy sounddevice
python reconocimiento_voz.py
```

**Entrenamiento**: en la primera pestaña, seleccionar el comando del menú desplegable, fijar la duración (2 s es suficiente) y grabar al menos 5 muestras por palabra; luego pulsar *Procesar y entrenar* para calcular los umbrales y guardarlos en `config.json`.

**Reconocimiento**: en la segunda pestaña, pulsar *Grabar y reconocer*; el sistema grabará el audio, calculará sus energías de subbandas, las comparará con los umbrales y mostrará `Comando '<palabra>' Reconocido` junto con la tabla de distancias completa.

## Referencias

[1] A. V. Oppenheim y R. W. Schafer, *Discrete-Time Signal Processing*, 3ra ed. Pearson Prentice Hall, 2009.

[2] L. R. Rabiner y B.-H. Juang, *Fundamentals of Speech Recognition*. Prentice Hall, 1993.

[3] Documentación oficial de NumPy. "Discrete Fourier Transform (numpy.fft)". Disponible en: https://numpy.org/doc/stable/reference/routines.fft.html
