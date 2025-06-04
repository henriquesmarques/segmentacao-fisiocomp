import scipy.io as sio # type: ignore
import numpy as np # type: ignore

mat_contents = sio.loadmat('Patient_3.mat')

""" if 'contours_color_VE_13' in mat_contents:
    dados = mat_contents['contours_color_VE_13']
    print(f'Countours_color_1:\n{dados}')
else:
    print("Não encontrado.")

if 'contours_color_VD_13' in mat_contents:
    dados = mat_contents['contours_color_VD_13']
    print(f'Countours_color_2:\n{dados}')
else:
    print("Não encontrado.") """

print(mat_contents.keys())

# Exibir as variáveis de nível superior
print("## Variáveis de Nível Superior e Seus Valores ##")
for var_name, var_value in mat_contents.items():
    print(f"\nVariável: {var_name}")
    print(f"Tipo: {type(var_value)}")
    # Se for um array numpy, mostrar mais detalhes
    if isinstance(var_value, np.ndarray):
        print(f"Shape: {var_value.shape}")
        print(f"Dtype: {var_value.dtype}")
