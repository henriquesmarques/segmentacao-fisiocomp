import os
import numpy as np
import matplotlib.pyplot as plt # type: ignore
import scipy.io as sio # type: ignore
from nibabel import load # type: ignore
from cv2 import dilate # type: ignore
from skimage.measure import find_contours # type: ignore

data_dir = os.getcwd()
data_list = sorted(os.listdir(f'{data_dir}/input'))
section = ['ED', 'ES']

for data in data_list:
    os.makedirs(os.path.join(f'{data_dir}/output/{data}', 'contours-txt'), exist_ok=True)
    os.makedirs(os.path.join(f'{data_dir}/output/{data}', 'contours-png'), exist_ok=True)
    try:
        mat = sio.loadmat(f'{data_dir}/input/paciente_cine_2/Patient_2.mat')
        print('    Arquivo .MAT encontrado.')
    except FileNotFoundError:
        print('    Erro: Arquivo .MAT não encontrado.')


    for fr in section:
        # Lendo a imagem
        image_name = '{0}/{1}.nii.gz'.format(f'{data_dir}/input/{data}', f'seg_sa_{fr}')
        if not os.path.exists(image_name):
            print('   Directory {0} does not contain an image with file '
                    'name {1}. Skip.'.format(data_dir, os.path.basename(image_name)))
            continue
        print('  Reading {} ...'.format(image_name))

        nim = load(image_name)
        image = nim.get_fdata()
        X, Y, Z = image.shape

        # Dicionário para armazenar os contornos
        contours_dict = {}
        
        for frame in range(Z):
            # Convertendo a imagem para 2D
            slice_2d = image[:, :, frame]
            # Definindo tamanho da plotagem
            fig, ax = plt.subplots(figsize=(Y / 25, X / 25))

            # Ventrículo esquerdo (EndoX)
            mask = (slice_2d == 1)
            # Encontrando os contornos na máscara
            contours = find_contours(mask, level=0.5)
            # print(contours)
            # Salvando os contornos de cada máscara no dicionário
            contours_dict[f'{fr}_{frame}_VE'] = [contour.tolist() for contour in contours]
            # Plotando os contornos
            for contour in contours:
                ax.plot(contour[:, 1], contour[:, 0], linewidth=0.5, color='red')
                
            # Ventrículo direito
            mask = (slice_2d == 3)
            # Encontrando os contornos na máscara
            contours = find_contours(mask, level=0.5)
            # Salvando os contornos de cada máscara no dicionário
            contours_dict[f'{fr}_{frame}_VD'] = [contour.tolist() for contour in contours]
            # Plotando os contornos
            for contour in contours:
                ax.plot(contour[:, 1], contour[:, 0], linewidth=0.5, color='red')

            # Adicionando padding de 3mm
            # Obtendo o espaçamento dos pixels (x, y)
            pixel_spacing = nim.header['pixdim'][1:3]  
            # Convertendo 3mm para pixels
            kernel_size = max(3, int(np.ceil(3 / np.mean(pixel_spacing))))
            # Cria uma matriz quadrada de tamanho kernel_size x kernel_size
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            # Técnica de dilatação
            mask = dilate(mask.astype(np.uint8), kernel, iterations=1)
            # Subtraindo interseção entre a máscara 2
            mask = mask & ~(slice_2d == 2)
            # Adicionando padding na imagem original
            slice_2d += mask

            # Extraindo contornos da segmentação completa com adição do padding
            contours = find_contours(slice_2d, level=0.1)
            # Salvando os contornos no dicionário
            contours_dict[f'{fr}_{frame}_EP'] = [contour.tolist() for contour in contours]
            # Plotando os contornos
            for contour in contours:
                ax.plot(contour[:, 1], contour[:, 0], linewidth=0.5, color='blue')

            # Salvando os contornos em uma imagem .jpg
            contour_image_path = os.path.join(f'{data_dir}/output/{data}/contours-png', f'{fr}_{frame}.png')
            # ax.imshow(slice_2d)
            plt.axis('off')
            plt.savefig(contour_image_path, pad_inches=0)
            plt.close(fig)
            
        # Salvando os contornos em arquivos .txt separados para cada key
        for key, contours in contours_dict.items():
            key_file_path = os.path.join(f'{data_dir}/output/{data}/contours-txt', f'{key}.txt')
            with open(key_file_path, 'w') as txt_file:
                for contour in contours:
                    for i, point in enumerate(contour):
                        if i % 6 == 0:
                            txt_file.write(f'{point[0]:.6f} {point[1]:.6f} 0\n')

        # Salvando os contornos no arquivo .mat de entrada
        endox = np.zeros(shape = (80,1,Z))
        endoy = np.zeros(shape = (80,1,Z))
        rvendox = np.zeros(shape = (80,1,Z))
        rvendoy = np.zeros(shape = (80,1,Z))
        rvepix = np.zeros(shape = (80,1,Z))
        rvepiy = np.zeros(shape = (80,1,Z))
    
        for key, contours in contours_dict.items():
            for contour in contours:
                # print(f'        {key}')
                if (f'{fr}_{frame}_VE' == key):
                    for i, point in enumerate(contour):
                        if i < 80:  # Limite de 80 pontos
                            endoy[i, 0, frame] = point[0]
                            endox[i, 0, frame] = point[1]
                elif (f'{fr}_{frame}_VD' == key):
                    for i, point in enumerate(contour):
                        if i < 80:  # Limite de 80 pontos
                            rvendoy[i, 0, frame] = point[0]
                            rvendox[i, 0, frame] = point[1]
                elif (f'{fr}_{frame}_EP' == key):
                    for i, point in enumerate(contour):
                        if i < 80:  # Limite de 80 pontos
                            rvepiy[i, 0, frame] = point[0]
                            rvepix[i, 0, frame] = point[1]
            
            
        # print(contours_dict)
        """ for key, contours in contours_dict.items():
            for contour in contours:
                # print(f'        {key}')
                print(f'{fr}_{frame}_VE')
                k = l = m = 0
                for point in contour:
                    if (f'{fr}_{frame}_VE' == key):
                        # print(f'        {key}')
                        endoy[k,0,frame] = point[0]
                        endox[k,0,frame] = point[1]
                        k = k + 1
                    if (f'{fr}_{frame}_VD' == key):
                        # print(f'        {key}')
                        rvendoy[l,0,frame] = point[0]
                        rvendox[l,0,frame] = point[1]
                        l = l + 1
                    if (f'{fr}_{frame}_EP' == key):
                        # print(f'        {key}')
                        rvepiy[m,0,frame] = point[0]
                        rvepix[m,0,frame] = point[1]
                        m = m + 1 """

        # Reescrevendo arquivo .MAT
        if 'setstruct' in mat: 
            print('    Dimensions EndoX: ', mat['setstruct']['EndoX'][0][0].ndim)
            print('    Shape EndoX: ', mat['setstruct']['EndoX'][0][0].shape)
            print('    Size EndoX: ', mat['setstruct']['EndoX'][0][0].size)
            
            mat['setstruct']['EndoX'][0][0] = endox
            mat['setstruct']['EndoY'][0][0] = endoy
            mat['setstruct']['RVEndoX'][0][0] = rvendox
            mat['setstruct']['RVEndoY'][0][0] = rvendoy
            mat['setstruct']['RVEpiX'][0][0] = rvepix
            mat['setstruct']['RVEpiY'][0][0] = rvepiy

            # print('    Dimensions EndoX: ', mat['setstruct']['EndoX'][0][0].ndim)
            # print('    Shape EndoX: ', mat['setstruct']['EndoX'][0][0].shape)
            # print('    Size EndoX: ', mat['setstruct']['EndoX'][0][0].size)

            sio.savemat(f'{data_dir}/output/paciente_cine_2/Patient_2_Editado.mat', mat)
            print('    Arquivo .MAT reescrito.')
        else:
            print('    Erro: Variável "setstruct" não encontrada no arquivo .MAT.')
            
print ('Done.')
