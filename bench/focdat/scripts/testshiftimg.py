import numpy as np

nro = 33; oro = 0.98; dro = 0.00125

ros = np.linspace(oro,oro + (nro-1)*dro,nro)


print(ros)
print(ros[12:])

nnro = 21; noro = 0.9875; dro = 0.00125

nros = np.linspace(noro,noro + (nnro-1)*dro,nnro)

print(nros[int((nnro-1)/2)])

print(nros)

