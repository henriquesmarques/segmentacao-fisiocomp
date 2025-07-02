import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat
from scipy.interpolate import interp1d
from nibabel import load # type: ignore
from cv2 import dilate, erode # type: ignore
from skimage.measure import find_contours # type: ignore
from shapely.geometry import Polygon, Point # type: ignore
    
def smooth_image(image, size_kernel):
    """ 
    Aplicando técnica de dilatação para cobrir pequenas imperfeições e logo após a técnica 
    de erosão para voltar ao tamanho original.
    """
    kernel = np.ones((size_kernel, size_kernel))
    image = dilate(image.astype(np.uint8), kernel, iterations=1)
    image = erode(image.astype(np.uint8), kernel, iterations=1)
    return image

def generate_closed_curve(contours, x, y, frame, saida):
    """
    Atualiza os arrays x e y com exatamente 80 pontos da maior curva fechada informada.
    """
    if len(contours) == 1:
        contour = contours[0]
        # retorna apenas o contorno válido
        saida[frame] = contour
        if len(contour) > 2:
            new_contours = resample_closed_curve(contour, 79)
            for ind, point in enumerate(new_contours):
                x[ind, 0, frame] = point[1]
                y[ind, 0, frame] = point[0]
            # Fechando a curva
            x[79, 0, frame] = x[0, 0, frame]
            y[79, 0, frame] = y[0, 0, frame]

    elif len(contours) > 1:
        maior_area = 0.0
        id_maior = 0
        for i, contour in enumerate(contours):
            area = area_closed_curve(contour)
            if area > maior_area:
                maior_area = area
                id_maior = i
        generate_closed_curve([contours[id_maior]], x, y, frame, saida)

def area_closed_curve(pontos):
    """
    Calcula a área de uma curva fechada (polígono) usando a Fórmula de Shoelace.

    Args:
        pontos: Uma lista de tuplas, onde cada tupla (x, y) representa um ponto
                no plano cartesiano que compõe a curva fechada.
                Os pontos devem estar em ordem sequencial (horário ou anti-horário).

    Returns:
        A área da curva fechada (polígono) como um float.
        Retorna 0.0 se a lista de pontos tiver menos de 3 pontos (não forma um polígono).
    """
    num_pontos = len(pontos)
    if num_pontos < 3:
        print("Para formar uma curva fechada (polígono), são necessários pelo menos 3 pontos.")
        return 0.0

    soma_primeiro_termo = 0
    soma_segundo_termo = 0

    for i in range(num_pontos):
        x1, y1 = pontos[i]
        x2, y2 = pontos[(i + 1) % num_pontos]  # O operador % garante que o último ponto se conecta ao primeiro

        soma_primeiro_termo += x1 * y2
        soma_segundo_termo += y1 * x2

    area = 0.5 * abs(soma_primeiro_termo - soma_segundo_termo)
    return area
        
def resample_closed_curve(points, num_points):
    points = np.asarray(points)
    
    # Close the curve if needed
    if not np.allclose(points[0], points[-1]):
        points = np.vstack([points, points[0]])

    # Compute cumulative distances (arc lengths)
    deltas = np.diff(points, axis=0)
    segment_lengths = np.linalg.norm(deltas, axis=1)
    arc_lengths = np.concatenate([[0], np.cumsum(segment_lengths)])
    
    # Total length of the curve
    total_length = arc_lengths[-1]
    
    # Create uniform arc lengths
    target_lengths = np.linspace(0, total_length, num_points, endpoint=False)
    
    # Interpolate in x and y separately
    interp_x = interp1d(arc_lengths, points[:, 0], kind='cubic')
    interp_y = interp1d(arc_lengths, points[:, 1], kind='cubic')
    
    resampled_x = interp_x(target_lengths)
    resampled_y = interp_y(target_lengths)
    
    return np.stack([resampled_x, resampled_y], axis=1)

def verifica_curva_contida(curva_interna_pontos, curva_externa_pontos):
    """
    Verifica se uma curva fechada está completamente contida dentro da área de outra curva fechada.

    Args:
        curva_interna_pontos (list of tuples): Uma lista de tuplas (x, y) representando os pontos
                                               da curva que se deseja verificar se está contida.
                                               A ordem dos pontos deve formar uma curva fechada.
        curva_externa_pontos (list of tuples): Uma lista de tuplas (x, y) representando os pontos
                                               da curva que representa a área externa.
                                               A ordem dos pontos deve formar uma curva fechada.

    Returns:
        bool: True se a curva interna estiver completamente contida na curva externa, False caso contrário.
    """
    if len(curva_interna_pontos) < 3 or len(curva_externa_pontos) < 3:
        return False

    # Cria objetos Polygon a partir dos pontos.
    # Shapely automaticamente fecha o polígono se o último ponto não for igual ao primeiro.
    poligono_externo = Polygon(curva_externa_pontos)
    poligono_interno = Polygon(curva_interna_pontos)

    # Verifica se o polígono interno está contido dentro do polígono externo
    if (poligono_externo.contains(poligono_interno)):
        return True
    else:
        return False

data_dir = os.getcwd()
data_list = sorted(os.listdir(f'{data_dir}/input'))
section = ['ED', 'ES']

for data in data_list:
    os.makedirs(os.path.join(f'{data_dir}/output/{data}', 'contours-txt'), exist_ok=True)
    os.makedirs(os.path.join(f'{data_dir}/output/{data}', 'contours-png'), exist_ok=True)
    try:
        mat_path = f'{data_dir}/input/{data}/{data}.mat'
        mat = loadmat(mat_path)
        print(f'  Reading {mat_path} ...')
    except FileNotFoundError:
        print('  Erro: Arquivo .MAT não encontrado.')

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
        endox = np.full((80,1,Z), np.nan)
        endoy = np.full((80,1,Z), np.nan)
        rvendox = np.full((80,1,Z), np.nan)
        rvendoy = np.full((80,1,Z), np.nan)
        rvepix = np.full((80,1,Z), np.nan)
        rvepiy = np.full((80,1,Z), np.nan)
        nan = np.full((80,1,1), np.nan)
        endo = [[] for _ in range(17)]
        rvendo = [[] for _ in range(17)]
        rvepi = [[] for _ in range(17)]
        
        for frame in range(Z):
            # Convertendo a imagem para 2D
            slice_2d = image[:, :, frame]
            """ # Definindo tamanho da plotagem
            fig, ax = plt.subplots(figsize=(Y / 25, X / 25)) """

            # Endo
            mask = (slice_2d == 1)
            # Aplicando técnica de dilatação e erosão
            mask = smooth_image(mask, 3)
            # Encontrando os contornos na máscara
            contours = find_contours(mask, level=0.5) # Retorno: um array numpy [N, (y, x)]
            # Salvando os contornos em um array numpy
            generate_closed_curve(contours, endox, endoy, frame, endo)
            # Plotando os contornos com os pontos originais
            """ for contour in contours:
                ax.plot(contour[:, 0], contour[:, 1], linewidth=0.5, color='gray') """
                
            # RVEndo
            mask = (slice_2d == 3)
            # Aplicando técnica de dilatação e erosão
            mask = smooth_image(mask, 3)
            # Encontrando os contornos na máscara
            contours = find_contours(mask, level=0.5)
            # Salvando os contornos em um array numpy
            generate_closed_curve(contours, rvendox, rvendoy, frame, rvendo)
            # Plotando os contornos com os pontos originais
            """ for contour in contours:
                ax.plot(contour[:, 0], contour[:, 1], linewidth=0.5, color='gray') """

            # RVEpi
            # Técnica de dilatação
            kernel_size = 3
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            mask = dilate(mask.astype(np.uint8), kernel, iterations=1)
            # Subtraindo interseção entre a máscara 2
            mask = mask & ~(slice_2d == 2)
            # Adicionando padding na imagem original
            slice_2d += mask
            # Aplicando técnica de dilatação e erosão
            slice_2d = smooth_image(slice_2d, 3)
            # Extraindo contornos da segmentação completa com adição do padding
            contours = find_contours(slice_2d, level=0.1)
            # Salvando os contornos em um array numpy
            generate_closed_curve(contours, rvepix, rvepiy, frame, rvepi)
            # Plotando os contornos com os pontos originais
            """ for contour in contours:
                ax.plot(contour[:, 0], contour[:, 1], linewidth=0.5, color='gray') """

            """ # Salvando os contornos em imagens .jpg
            ax.plot(endoy[:,0,frame], endox[:,0,frame], linewidth=0.5, color='black')
            ax.plot(rvendoy[:,0,frame], rvendox[:,0,frame], linewidth=0.5, color='red')
            ax.plot(rvepiy[:,0,frame], rvepix[:,0,frame], linewidth=0.5, color='blue')
            contour_image_path = os.path.join(f'{data_dir}/output/{data}/contours-png', f'{fr}_{frame}.png')
            plt.axis('off')
            plt.savefig(contour_image_path, pad_inches=0)
            plt.close(fig) """

        # Removendo contorno do endocárdio das primeiras fatias
        """ frame = 2
        while (frame >= 0):
            if np.isnan(endox[0,0,frame]) or np.isnan(rvendox[0,0,frame]):
                if not np.isnan(rvepix[0,0,frame]):
                    for ind in range(80):
                        endox[ind,0,frame] = endoy[ind,0,frame] = rvendox[ind,0,frame] = rvendoy[ind,0,frame] = np.nan
                    frame = frame - 1
                    while (frame >= 0):
                        if not np.isnan(endox[0,0,frame]) or not np.isnan(rvendox[0,0,frame]) or not np.isnan(rvepix[0,0,frame]):
                            for ind in range(80):
                                endox[ind,0,frame] = endoy[ind,0,frame] = rvendox[ind,0,frame] = rvendoy[ind,0,frame] = rvepix[ind,0,frame] = rvepiy[ind,0,frame] = np.nan
                        frame = frame - 1
            frame = frame - 1 """
        
        # Fatias iniciais
        for frame in range(Z):
            if not np.isnan(rvepix[0,0,frame]):
                endox[:,0,frame] = endoy[:,0,frame] = rvendox[:,0,frame] = rvendoy[:,0,frame] = nan[:,0,0]
                break

        # Fatias finais
        frame = int(Z/2)
        while frame < Z:
            if (verifica_curva_contida(endo[frame], rvepi[frame]) == False or verifica_curva_contida(rvendo[frame], rvepi[frame]) == False):
                endox[:,0,frame] = endoy[:,0,frame] = rvendox[:,0,frame] = rvendoy[:,0,frame] = rvepix[:,0,frame] = rvepiy[:,0,frame] = nan[:,0,0]
                while (frame < Z):
                    endox[:,0,frame] = endoy[:,0,frame] = rvendox[:,0,frame] = rvendoy[:,0,frame] = rvepix[:,0,frame] = rvepiy[:,0,frame] = nan[:,0,0]
                    frame += 1
            frame += 1
        
        for frame in range(Z):
            # Salvando os contornos em arquivos .txt 
            contour_data = [("Endo", endox, endoy),("RVEndo", rvendox, rvendoy), ("RVEpi", rvepix, rvepiy)]
            for prefix, x_array, y_array in contour_data:
                txt_path = os.path.join(f'{data_dir}/output/{data}/contours-txt', f'{fr}_{frame}_{prefix}.txt')
                with open(txt_path, 'w') as txt_file:
                    for ind in range(80):
                        if np.isnan(x_array[ind, 0, frame]):
                            txt_file.write('0 0 0\n')
                        else:
                            txt_file.write(f'{x_array[ind,0,frame]:.6f} {y_array[ind,0,frame]:.6f} 0\n')

            # Salvando os contornos em imagens .jpg
            fig, ax = plt.subplots(figsize=(Y / 25, X / 25))
            ax.plot(endoy[:,0,frame], endox[:,0,frame], linewidth=0.5, color='black')
            ax.plot(rvendoy[:,0,frame], rvendox[:,0,frame], linewidth=0.5, color='red')
            ax.plot(rvepiy[:,0,frame], rvepix[:,0,frame], linewidth=0.5, color='blue')
            contour_image_path = os.path.join(f'{data_dir}/output/{data}/contours-png', f'{fr}_{frame}.png')
            plt.axis('off')
            plt.savefig(contour_image_path, pad_inches=0)
            plt.close(fig)

        # Reescrevendo arquivo .MAT
        if 'setstruct' in mat: 
            mat['setstruct']['EndoX'][0][0] = endox
            mat['setstruct']['EndoY'][0][0] = endoy
            mat['setstruct']['RVEndoX'][0][0] = rvendox
            mat['setstruct']['RVEndoY'][0][0] = rvendoy
            mat['setstruct']['RVEpiX'][0][0] = rvepix
            mat['setstruct']['RVEpiY'][0][0] = rvepiy
            savemat(f'{data_dir}/output/{data}/{data}_{fr}_editado.mat', mat)
            print('  Salvando arquivo .MAT ...')
        else:
            print('  Erro: Variável "setstruct" não encontrada no arquivo .MAT')

print ('  Find Contours done.')
