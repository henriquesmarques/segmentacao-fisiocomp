import os
import numpy as np # type: ignore
import nibabel as nib # type: ignore
from skimage.measure import find_contours # type: ignore
import matplotlib.pyplot as plt # type: ignore
import cv2 as cv # type: ignore
from scipy.io import savemat # type: ignore

""" Entrada: 'seg_sa_ED.nii.gz' e/ou 'seg_sa_ES.nii.gz'
    Saída: Contornos de cada fatia em formato .TXT e .JPG """

data_dir = os.getcwd()
section = ['ED', 'ES']
os.makedirs(os.path.join(f'{data_dir}/output', 'contours-txt'), exist_ok=True)
os.makedirs(os.path.join(f'{data_dir}/output', 'contours-png'), exist_ok=True)

for fr in section:
    # Lendo a imagem
    image_name = '{0}/{1}.nii.gz'.format(f'{data_dir}/input', f'seg_sa_{fr}')
    if not os.path.exists(image_name):
        print('  Directory {0} does not contain an image with file '
                'name {1}. Skip.'.format(data_dir, os.path.basename(image_name)))
        continue
    print('  Reading {} ...'.format(image_name))
    # Convertendo imagem em um array numpy
    nim = nib.load(image_name)
    image = nim.get_fdata()
    X, Y, Z = image.shape

    for frame in range(Z):
        # Convertendo a imagem para 2D
        slice_2d = image[:, :, frame]
        # Extraindo quantidade de cores encontradas na segmentação
        unique_colors = np.unique(slice_2d)
        # Dicionário para armazenar os contornos
        contours_dict = {}
        # Iterando sobre cada cor única para extrair e salvar seus contornos
        fig, ax = plt.subplots(figsize=(Y / 25, X / 25))

        # Ventrículo esquerdo
        mask = (slice_2d == 1)
        # Encontrando os contornos na máscara
        contours = find_contours(mask, level=0.5)
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
        mask = cv.dilate(mask.astype(np.uint8), kernel, iterations=1)
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
        contour_image_path = os.path.join(f'{data_dir}/output/contours-png', f'{fr}_{frame}.png')
        # ax.imshow(slice_2d)
        plt.axis('off')
        plt.savefig(contour_image_path, pad_inches=0)
        plt.close(fig)

        # Salvando os contornos em arquivos .txt separados para cada key
        for key, contours in contours_dict.items():
            key_file_path = os.path.join(f'{data_dir}/output/contours-txt', f'{key}.txt')
            with open(key_file_path, 'w') as txt_file:
                # txt_file.write(f'{key}:\n')
                for contour in contours:
                    for point in contour:
                        txt_file.write(f'{point[0]:.6f} {point[1]:.6f}\n')
                    # print(f'    Contornos salvos em {key_file_path}')

        print(f'    Fatia {fr} {frame} salva.')

    """ # Salvando os contornos em um arquivo .mat
    mat_file_path = os.path.join(f'{data_dir}/contours/mat', f'contours_{fr}.mat')
    savemat(mat_file_path, contours_dict)
    print(f'  Contornos salvos em {mat_file_path}') """

print ('Done.')
