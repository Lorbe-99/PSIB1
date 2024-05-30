import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
import scipy.io

# Lectura de archivos EEG
arch = scipy.io.loadmat("eeg_epilepsia.mat")
data = arch['data']
x = data['x']
fs = data['fs']
tiempo = data['tiempo']

x = x[0][0]
fs = fs[0][0][0][0]
t = tiempo[0][0]

cant_canales = x.shape[0]

# Frecuencias de corte baja y alta
lowcut = 0.05  # Frecuencia de corte baja en Hz
highcut = 100  # Frecuencia de corte alta en Hz
nyquist = 0.5 * fs  # Frecuencia de Nyquist
low = lowcut / nyquist
high = highcut / nyquist

# Diseño del filtro Butterworth pasa bajos
order_low = 5  # Orden del filtro pasa bajos
b_low, a_low = scipy.signal.butter(order_low, high, btype='low')

# Diseño del filtro Butterworth pasa altos
order_high = 5  # Orden del filtro pasa altos
b_high, a_high = scipy.signal.butter(order_high, low, btype='high')

# Aplicación del filtro pasa bajos y luego pasa altos a la señal
x_filt = scipy.signal.lfilter(b_low, a_low, x[18])
x_filt = scipy.signal.lfilter(b_high, a_high, x_filt)

# Tiempo en segundos, basado en la longitud de la señal y la frecuencia de muestreo
t = np.arange(len(x[18])) / fs

# Graficar la señal original y la filtrada
plt.figure(figsize=(15, 6))

plt.plot(t, x_filt, label='Señal filtrada', color='r')
plt.xlabel('Tiempo (s)')
plt.ylabel('Amplitud')
plt.title('Señal EEG filtrada entre 0.05 Hz y 100 Hz')
plt.legend()

plt.tight_layout()
plt.show()
