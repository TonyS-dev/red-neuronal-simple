### Instrucciones

1. ✅ Instalar Docker en tu sistema (Windows, macOS o Linux).  
2. 🛠️ Construir y ejecutar una imagen basada en tu `Dockerfile` para correr el script `neurona-simple.py`.

---

## 🚀 1. Instalar Docker

### 🔹 En **Windows 10/11**
1. Ve a [https://www.docker.com/products/docker-desktop/](https://www.docker.com/products/docker-desktop/)
2. Descarga **Docker Desktop for Windows**.
3. Ejecuta el instalador.
4. Asegúrate de habilitar:
   - **WSL 2** (te lo pedirá automáticamente si no está).
   - La integración con terminales como PowerShell o CMD.
5. Reinicia tu PC.
6. Abre Docker Desktop y asegúrate de que diga “Docker Engine is running”.

> 🧪 Verifica en terminal:
```bash
docker --version
```

---

### 🔹 En **macOS**
1. Descarga Docker Desktop desde [docker.com/products/docker-desktop](https://www.docker.com/products/docker-desktop/)
2. Instala el `.dmg` y sigue los pasos.
3. Verifica:
```bash
docker --version
```

---

### 🔹 En **Linux (Ubuntu/Debian)**
```bash
sudo apt update
sudo apt install -y docker.io
sudo systemctl enable docker
sudo systemctl start docker
sudo usermod -aG docker $USER
```

> 🔁 Sal y vuelve a iniciar sesión para que se apliquen los permisos del grupo `docker`.

Verifica:
```bash
docker --version
```

---

## 📦 2. Estructura esperada del proyecto

Asegúrate de que la estructura de tu carpeta esté así:

```
red-neuronal-simple/
├── Dockerfile
└── neurona-simple/
    └── salidas
        └── grafica_perdida.png
        └── hash_datos.txt
        └── modelo_entrenado.keras
    └── neurona-simple.py
```
La carpeta `salidas`, al momento de clonar el proyecto no tendrá ningún archivo, esta será utilizada para almacenar archivos que se compartirán entre el contenedor Docker y la máquina anfitrión para poder observar desde la gráfica de perdida, el hash de datos y recuperar los datos dle modelo entrenado 
---

## 🏗️ 3. Construir la imagen con Docker

Abre una terminal y navega a la carpeta donde tienes el `Dockerfile`. Luego ejecuta:

```bash
docker build -t neurona-simple .
```

- `-t neurona-simple`: le asigna un nombre a la imagen.
- `.`: indica que el `Dockerfile` está en el directorio actual.

---

## ▶️ 4. Ejecutar el contenedor

Una vez construida la imagen:

```bash
docker run --rm neuronasimple
```

- `--rm`: borra automáticamente el contenedor después de ejecutarse.
- Esto correrá el script `neurona-simple.py` que está dentro de la carpeta `neurona-simple`.

---

## 🧠 Recordatorio sobre el contenido del Dockerfile

Tu `Dockerfile` ya incluye:

```Dockerfile
FROM tensorflow/tensorflow:2.15.0

WORKDIR /app

COPY neurona-simple /app

RUN pip install matplotlib numpy

CMD ["python", "neurona-simple.py"]
```

Este archivo:
- Usa la imagen oficial de TensorFlow.
- Copia tu carpeta `neurona-simple` al contenedor.
- Instala `matplotlib` y `numpy`.
- Ejecuta tu script al iniciar el contenedor.

---

Comando para ejecutar el contenedor:  
```bash
docker run --rm -v "${PWD}\neurona-simple:/app" redneuronalsimple python neurona-simple.py 25.0
```

### Anotaciones de la funcionalidad

¿Se puede decir que este código es machine learning o inteligencia artificial?

Sí, **se puede decir que este código es un ejemplo de *machine learning***, y por extensión, también puede considerarse una forma de **inteligencia artificial (IA)**, aunque con algunos matices importantes y limitados. Podemos segmentar por partes la realidad de lo que representa el código:

---

### ✅ ¿Es *Machine Learning*?

**Sí.**  
Este código **entrena un modelo de regresión lineal simple** utilizando una red neuronal con una sola capa densa para aprender la relación entre grados Celsius y Fahrenheit. Estas son las características que lo confirman como un sistema de *machine learning*:

- Usa datos (pares de temperaturas) para que el modelo *aprenda* una función de conversión.
- Utiliza TensorFlow, una librería de *deep learning*.
- Aplica un algoritmo de optimización (*Adam*) para ajustar los pesos del modelo minimizando el error cuadrático medio (*mean squared error*).
- Guarda el modelo entrenado y reutiliza su conocimiento si los datos no cambian.

✅ Entonces: **es machine learning supervisado**, específicamente **regresión lineal implementada con una red neuronal simple.**

---

### 🤖 ¿Es *Inteligencia Artificial*?

**También sí, pero en un sentido más general.**  
Dentro del campo de la inteligencia artificial, el *machine learning* es una subcategoría. Esta definición aplica:

> *La inteligencia artificial es el campo que estudia cómo lograr que las máquinas imiten formas de inteligencia humana, y el aprendizaje automático (machine learning) es una técnica para lograr esto.*

Nuestro código **no tiene razonamiento, planificación ni realiza toma de decisiones**, por lo que no representa una IA avanzada o general, pero **sí cae dentro de lo conocido como IA débil o estrecha**, la cual está diseñada para tareas específicas (como la predicción de temperatura).

---

### 🔍 ¿Es *Deep Learning*?

**No**  
Aunque usamos TensorFlow (una herramienta de deep learning), el modelo es una sola neurona con una entrada y una salida y dos capas ocultas sin gran complejidad, así que **es un caso simple de ML, pero no deep learning como tal.**

---

### 🧠 Resumen

| Término                 | ¿Aplica?  | Justificación |
|-------------------------|-----------|----------------|
| Machine Learning        | ✅ Sí     | Entrena un modelo a partir de datos usando optimización |
| Inteligencia Artificial | ✅ Sí     | Automatiza una predicción mediante aprendizaje |
| Deep Learning           | ⚠️ Parcial| Usa una red neuronal, pero no profunda (solo una capa) |


----------------------

La propiedad `epochs`:

```python
historial = modelo.fit(celsius, fahrenheit, epochs=1000, verbose=False)
```

Y significa lo siguiente:

---

Un **epoch** (época) es **una pasada completa por todo el conjunto de datos de entrenamiento**.  
En otras palabras:

> **`epochs=1000` significa que el modelo va a recorrer los mismos datos 1000 veces durante el entrenamiento** para ajustar sus pesos e intentar minimizar el error (la función de pérdida).

---

### 🧠 ¿Por qué es importante?

Cada vez que el modelo recorre todos los datos:
- Calcula el error (con la función de pérdida `mean_squared_error`)
- Ajusta los pesos del modelo (usando el optimizador `Adam`)
- Intenta mejorar su precisión en la siguiente vuelta

---

### 🎛️ ¿Qué pasa si aumento o disminuyo `epochs`?

| Cantidad de `epochs` | Resultado típico |
|----------------------|------------------|
| Muy pocas (ej: 10)   | El modelo no aprende lo suficiente (subentrenamiento) |
| Muchas (ej: 10000)   | Puede sobreajustarse (memoriza los datos en vez de generalizar) |
| Valor adecuado (ej: 500–1000) | Aprende bien y generaliza si los datos y arquitectura lo permiten |

---

### 🧪 Tip: Ver si está aprendiendo

```python
plt.plot(historial.history["loss"])
```

El gráfico generado en grafica_perdida.png nos muestra **cómo disminuye la pérdida** con cada epoch. Si la curva se aplana demasiado rápido, podrías reducir los `epochs`. Si no ha convergido, podrías aumentarlos.

---
