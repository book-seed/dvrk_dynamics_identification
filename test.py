import scipy
N = 5
Wn = 0.5
b = scipy.signal.butter(N, Wn, btype='low', analog=False, output='ba')

print(b[0])
print(b[1])
