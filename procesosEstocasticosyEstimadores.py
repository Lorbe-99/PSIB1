import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import scipy.signal as sig
from scipy import io
from scipy.fftpack import fft, ifft, fftfreq


def proceso_estocastico(media, varianza, tamaño_muestra, num_realizaciones):
    # Genera números aleatorios con distribución normal para múltiples realizaciones
    muestras = np.random.normal(loc=media, scale=np.sqrt(varianza), size=(tamaño_muestra, num_realizaciones))
    return muestras

# Parámetros del proceso estocástico
media = 5  # Media del proceso
varianza = 10  # Varianza del proceso
tamaño_muestra = 100  # Tamaño de la muestra
num_realizaciones = 5  # Número de realizaciones del proceso

# Genera las muestras del proceso estocástico
muestras = proceso_estocastico(media, varianza, tamaño_muestra, num_realizaciones)

# Genera secuencia de tiempo.
tiempo = np.arange(tamaño_muestra)

# Grafica las realizaciones del proceso estocástico en función del tiempo
for i in range(num_realizaciones):
    plt.scatter(tiempo, muestras[:, i], label=f'Realización {i+1}')

plt.ylim(-15,20)
plt.title('Realizaciones del Proceso Estocástico en función del Tiempo')
plt.xlabel('Tiempo')
plt.ylabel('Valor del Proceso Estocástico')
plt.legend()
plt.grid(True)
plt.plot([0,tiempo.shape[0]],[media + np.sqrt(varianza),media + np.sqrt(varianza)],'k--')
plt.plot([0,tiempo.shape[0]],[media,media],'k--')
plt.plot([0,tiempo.shape[0]],[media - np.sqrt(varianza),media - np.sqrt(varianza)],'k--')
plt.show()