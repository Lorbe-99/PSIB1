import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import scipy.signal as sig
from scipy import io
from scipy.fftpack import fft, ifft, fftfreq

def proceso_estocastico(media, varianza, tamaño_muestra, num_realizaciones):
    # Genera números aleatorios con distribución normal para múltiples realizaciones
    muestras = np.random.randn(tamaño_muestra, num_realizaciones) * np.sqrt(varianza) + media
    return muestras

# Parámetros del proceso estocástico
media = 5  # Media del proceso
varianza = 10  # Varianza del proceso
tamaño_muestra = 200  # Tamaño de la muestra
num_realizaciones = 5  # Número de realizaciones del proceso

# Genera las muestras del proceso estocástico
muestras = proceso_estocastico(media, varianza, tamaño_muestra, num_realizaciones)

# Genera secuencia de tiempo
tiempo = np.arange(tamaño_muestra)

# Grafica las realizaciones del proceso estocástico en función del tiempo
for i in range(num_realizaciones):
    plt.scatter(tiempo, muestras[:, i], label=f'Realización {i+1}')

plt.ylim(-15, 20)
plt.title('Realizaciones del Proceso Estocástico en función del Tiempo')
plt.xlabel('Tiempo')
plt.ylabel('Valor del Proceso Estocástico')
plt.legend()
plt.grid(True)
plt.plot([0, tiempo.shape[0]], [media + np.sqrt(varianza), media + np.sqrt(varianza)], 'k--')
plt.plot([0, tiempo.shape[0]], [media, media], 'k--')
plt.plot([0, tiempo.shape[0]], [media - np.sqrt(varianza), media - np.sqrt(varianza)], 'k--')
plt.show()

#-----------------------------------PUNTO 2 --------------------------------------------
num_realizaciones=1
muestras1 = 200
# Genera las muestras del proceso estocástico
ruido = proceso_estocastico(media, varianza, muestras1, num_realizaciones)

# Calcular la correlación en función del lag
lags = np.arange(-muestras1 + 1, muestras1)
correlacion_promedio = np.zeros_like(lags, dtype=float)

for i in range(num_realizaciones):
    correlacion = np.correlate(ruido[:, i], ruido[:, i], mode='full')
    correlacion_promedio += correlacion / num_realizaciones

# Normalizar correlación
correlacion_promedio /= np.max(correlacion_promedio)

# Gráfica de la correlación promedio
plt.plot(lags, correlacion_promedio)
plt.title('Autocorrelación Promedio del Proceso Estocástico')
plt.xlabel('Desfase Temporal')
plt.ylabel('Autocorrelación Normalizada')
plt.grid(True)
plt.show()


#SES INSES

# Genera eje de tiempo
tiempo = np.arange(muestras1)

plt.plot(tiempo, ruido)
plt.title('Ruido blanco')
plt.xlabel('Tiempo')
plt.ylabel('Amplitud')
plt.grid(True)
plt.show()

# Calcular la correlación en función del lag
lags = np.arange(-muestras1+1,muestras1)
correlacion = np.correlate(ruido[:,0], ruido[:,0], mode='full')

N=len(ruido[:,0])
Rrr1=np.ndarray(shape=(len(lags)), dtype=float)
for i in lags:
  Rrr1[i,]=correlacion[i,]/(N-np.abs(lags[i])) # Autocorrelción insesgada

Rrr2=correlacion/N # Autocorrelación sesgada

Srr1=fft(Rrr1)         # TF autocorrelcación insesgada
w1=fftfreq(len(Srr1))*N

Srr2=fft(Rrr2)         # TF autocorrelcación sesgada
w2=fftfreq(len(Srr2))*N

# Gráficos inses
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 5))
ax1.plot(lags,Rrr1)
ax1.grid(linestyle='--')
ax1.set_title("Autocorrelación insesgada")
ax1.set_xlabel("k lags")
ax2.plot(w1[0:N],np.abs(Srr1[0:N]))
ax2.grid(linestyle='--')
ax2.set_title("TF correlación")
ax2.set_xlabel("Frecuencia")
plt.show()
print('Media y varianza inses', np.mean(Rrr1), np.var(Rrr1))

#Graficos ses
fig, (ax3, ax4) = plt.subplots(1, 2, figsize=(18, 5))
ax3.plot(lags,Rrr2)
ax3.grid(linestyle='--')
ax3.set_title("Autocorrelación sesgada")
ax3.set_xlabel("lag")
ax4.plot(w2[0:100],np.abs(Srr2[0:100]))
ax4.grid(linestyle='--')
ax4.set_title("TF correlación")
ax4.set_xlabel("Frecuencia")
plt.show()
print('Media y varianza ses', np.mean(Rrr2), np.var(Rrr2))

#CON LOS GRAFICOS ME DOY CUENTA QUE NO CUMPLE CON QUE  
# la autocovarianza (o autocorrelación) del proceso depende únicamente del desfase temporal y no del tiempo en sí mismo

#FALTA CALCULAR LA SEGUNDA CONDICION QUE ES La media del proceso es constante a lo largo del tiempo.

# Calcular la media a lo largo del tiempo
media_tiempo = np.mean(muestras, axis=0)

# Graficar la media a lo largo del tiempo
plt.plot(media_tiempo)
plt.title('Media a lo largo del tiempo')
plt.xlabel('Realización')
plt.ylabel('Media')
plt.grid(True)
plt.show()

# Verificar si la media es constante
media_promedio = np.mean(media_tiempo)
if np.allclose(media_tiempo, media_promedio):
    print("La media del proceso es constante a lo largo del tiempo.")
else:
    print("La media del proceso no es constante a lo largo del tiempo.")
     
   
    
#-------PUNTO 3 Y 4-----------------------
#CALCULAR LA CORRELACION Y AUTOCORRELACION


def correlacion_autocorrelacion(signal):
    # Calcula la correlación cruzada (correlación con la misma señal desplazada)
    correlacion = np.convolve(signal, signal[::-1], mode='full') / len(signal)

    # Calcula la autocorrelación (correlación con la misma señal invertida)
    autocorrelacion = np.convolve(signal, signal[::-1], mode='full') / len(signal)

    # Eje de tiempo
    lags = np.arange(-len(signal) + 1, len(signal))

    return lags, correlacion, autocorrelacion

# Parámetros del proceso estocástico
media = 0  # Media del proceso
varianza = 1  # Varianza del proceso
muestras1 = 500  # Tamaño de la muestra
num_realizaciones = 1  # Número de realizaciones del proceso

# Genera las muestras del proceso estocástico
ruido1 = np.random.normal(loc=media, scale=np.sqrt(varianza), size=muestras1)

# Calcula la correlación y autocorrelación
lags, correlacion, autocorrelacion = correlacion_autocorrelacion(ruido1)

# Graficar la señal de ruido blanco
plt.plot(ruido1, color='red')
plt.title('Ruido blanco')
plt.xlabel('Tiempo')
plt.ylabel('Amplitud')
plt.grid(True)
plt.show()

# Graficar la correlación cruzada
plt.plot(lags, correlacion, color='blue')
plt.title('Correlación Cruzada')
plt.xlabel('Desfase')
plt.ylabel('Valor')
plt.grid(True)
plt.show()

# Graficar la autocorrelación
plt.plot(lags, autocorrelacion, color='green')
plt.title('Autocorrelación')
plt.xlabel('Desfase')
plt.ylabel('Valor')
plt.grid(True)
plt.show()
