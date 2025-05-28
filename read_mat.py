import scipy.io as sio # type: ignore

mat_contents = sio.loadmat('mat/contours_ES.mat')

if 'contours_color_VE_13' in mat_contents:
    dados = mat_contents['contours_color_VE_13']
    print(f'Countours_color_1:\n{dados}')
else:
    print("Não encontrado.")

if 'contours_color_VD_13' in mat_contents:
    dados = mat_contents['contours_color_VD_13']
    print(f'Countours_color_2:\n{dados}')
else:
    print("Não encontrado.")

print(mat_contents.keys())