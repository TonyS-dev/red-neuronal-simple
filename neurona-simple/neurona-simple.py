# =============================
# IMPORTACIÓN DE MÓDULOS
# =============================

import os                 # Para crear directorios y manejar rutas de archivos
import sys                # Para acceder a los argumentos de línea de comandos
import tensorflow as tf   # Librería para construir y entrenar redes neuronales
import numpy as np        # Para trabajar con arreglos numéricos
import matplotlib.pyplot as plt  # Para generar gráficas de pérdida
import hashlib            # Para calcular un hash de los datos y detectar cambios

# =============================
# VALIDACIÓN DE ARGUMENTOS
# =============================

# Verifica si se pasó un argumento adicional (valor de temperatura en Celsius)
if len(sys.argv) < 2:
    print("Por favor, pasa un valor de temperatura en Celsius como argumento.")
    sys.exit(1)  # Finaliza el script con código de error

# Intenta convertir el argumento a un número flotante
try:
    valor_celsius = float(sys.argv[1])
except ValueError:
    print("El valor ingresado no es un número válido.")
    sys.exit(1)

# =============================
# PREPARACIÓN DE DIRECTORIO DE SALIDA
# =============================

# Crea el directorio 'salidas' si no existe (para guardar modelos, gráficas, etc.)
os.makedirs("salidas", exist_ok=True)

# =============================
# DATOS DE ENTRENAMIENTO
# =============================

# Datos de ejemplo: temperatura en Celsius y su equivalente en Fahrenheit
celsius = np.array([-40, -10, 0, 8, 15, 22, 38], dtype=float)
fahrenheit = np.array([-40, 14, 32, 46, 59, 72, 100], dtype=float)

# =============================
# FUNCIÓN: CÁLCULO DE HASH DE DATOS
# =============================

def calcular_hash_datos(celsius, fahrenheit):
    """
    Calcula un hash MD5 a partir de los datos de entrenamiento,
    para detectar si han cambiado entre ejecuciones.
    """
    datos_concatenados = np.concatenate((celsius, fahrenheit)).tobytes()
    return hashlib.md5(datos_concatenados).hexdigest()

# =============================
# CONFIGURACIÓN DE MODELO Y HASH
# =============================

modelo_path = "salidas/modelo_entrenado.keras"   # Ruta para guardar o cargar el modelo
hash_path = "salidas/hash_datos.txt"             # Ruta donde se almacena el hash de los datos
hash_actual = calcular_hash_datos(celsius, fahrenheit)  # Hash generado con los datos actuales

entrenar = False  # Bandera para determinar si se debe entrenar el modelo
hash_antiguo = None  # Hash previamente guardado (si existe)

# =============================
# VERIFICACIÓN DE EXISTENCIA DE MODELO Y DATOS
# =============================

# Si existen el modelo y el archivo de hash, compara los hashes
if os.path.exists(modelo_path) and os.path.exists(hash_path):
    with open(hash_path, "r") as f:
        hash_antiguo = f.read().strip()
    if hash_antiguo != hash_actual:
        print("⚠️ Los datos de entrenamiento han cambiado. Se entrenará de nuevo.")
        entrenar = True
    else:
        print("✅ Modelo y datos intactos. Cargando modelo existente...")
        modelo = tf.keras.models.load_model(modelo_path)
else:
    # Si no hay modelo o no hay hash previo, se debe entrenar desde cero
    print("📦 No se encontró modelo entrenado o hash de datos. Entrenando desde cero.")
    entrenar = True

# =============================
# ENTRENAMIENTO DEL MODELO (SI ES NECESARIO)
# =============================

historial = None  # Variable para almacenar el historial del entrenamiento (loss por epoch)

if entrenar:
    # Crea una red neuronal simple con una capa densa y dos capas ocultas para simular el deep learning
    capa_oculta_1 = tf.keras.layers.Dense(units=3, input_shape=[1])
    capa_oculta_2 = tf.keras.layers.Dense(units=3)
    salida = tf.keras.layers.Dense(units=1)
    modelo = tf.keras.Sequential([capa_oculta_1, capa_oculta_2, salida])
    
    # Compila el modelo con el optimizador Adam y el error cuadrático medio
    modelo.compile(optimizer=tf.keras.optimizers.Adam(0.1), loss='mean_squared_error')
    
    # Entrena el modelo con los datos de ejemplo durante 1000 épocas
    historial = modelo.fit(celsius, fahrenheit, epochs=1000, verbose=False)
    
    # Guarda el modelo entrenado
    modelo.save(modelo_path)
    
    # Guarda el nuevo hash de los datos usados
    with open(hash_path, "w") as f:
        f.write(hash_actual)
    
    print("✅ Modelo entrenado y guardado con nuevo hash.")

# =============================
# GENERACIÓN DE LA GRÁFICA DE PÉRDIDA
# =============================

# Si hubo entrenamiento, genera y guarda la gráfica de pérdida
if historial is not None:
    plt.plot(historial.history["loss"])
    plt.title("Historial de pérdida")
    plt.xlabel("Épocas")
    plt.ylabel("Pérdida")
    plt.savefig("salidas/grafica_perdida.png")
    print("📈 Gráfica de pérdida guardada.")
else:
    print("📉 No se generó gráfica porque no hubo entrenamiento.")

# =============================
# PREDICCIÓN CON EL MODELO
# =============================

print("🔮 Haciendo predicción...")

# Usa el modelo para predecir la temperatura en Fahrenheit del valor ingresado
resultado = modelo.predict([valor_celsius])
print(f"🌡️ El resultado para {valor_celsius}°C es {resultado[0][0]:.2f} °F")

# =============================
# VISUALIZACIÓN DE PESOS DEL MODELO
# =============================

print("⚙️ Pesos internos del modelo:")

# Imprime los pesos aprendidos por la red (pendiente y sesgo del modelo lineal)
for layer in modelo.layers:
    print(layer.get_weights())
