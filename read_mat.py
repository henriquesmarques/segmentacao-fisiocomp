import scipy.io as sio # type: ignore
import numpy as np

data = sio.loadmat('Patient_3.mat')

""" if 'setstruct' in data:
    dados = data['setstruct']
    print(f'setstruct:\n{dados}')
else:
    print("Não encontrado.") """

for var_name, var_value in data.items():
    if (var_name == 'setstruct'):
        print(f"\nVariável: {var_name}")
        print(f"Tipo: {type(var_value)}")
        print(f"Shape: {var_value.shape}")
        # print(f"Dtype: {var_value.dtype}")
        # print(var_value['EndoY'])
    
endoX = data['setstruct']['EndoX'][0][0] #ve
xlv = endoX[:,0,8]
endoY = data['setstruct']['EndoY'][0][0]
ylv = endoY[:,0,8]

RVEndoX = data['setstruct']['RVEndoX'][0][0] #vd
xrv = RVEndoX[:,0,8]
RVEndoY = data['setstruct']['RVEndoY'][0][0]
yrv = RVEndoY[:,0,8]

RVEpiX = data['setstruct']['RVEpiX'][0][0] # epi
xepi = RVEpiX[:,0,8]
RVEpiY = data['setstruct']['RVEpiY'][0][0]
yepi = RVEpiY[:,0,8]


print(xlv)
print(endoY)

# print(data.keys())

""" # Exibir as variáveis de nível superior
print("## Variáveis de Nível Superior e Seus Valores ##")
for var_name, var_value in data.items():
    print(f"\nVariável: {var_name}")
    print(f"Tipo: {type(var_value)}")
    # Se for um array numpy, mostrar mais detalhes
    if isinstance(var_value, np.ndarray):
        print(f"Shape: {var_value.shape}")
        print(f"Dtype: {var_value.dtype}") """
