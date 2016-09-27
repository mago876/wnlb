import numpy as np
from numpy import linalg as la
from numpy import concatenate as conc
from numpy.lib import pad
from numpy.matlib import repmat
import pywt
import multiprocessing as mp
from sklearn.neighbors import NearestNeighbors


###################### Basic functions ######################

def aumentar_imagen(im, n):
    '''Aumenta n pixeles hacia cada lado a la imagen im por reflexion'''
    
    return pad(im, ((n,n),(n,n)), mode='reflect')


def recortar_imagen(im, n):
    '''Recorta n pixeles de cada lado de la imagen im'''
    
    return im[n:-n, n:-n]


def gen_listpat2(im, t):
    '''Genera bloque 3D de patches de la imagen im
    Se usa en: NLMeans2, NLBayes2'''

    N1, N2 = im.shape  # Tamanho de la imagen
    ncol = N1-t+1  # Cantidad de patches en cada columna de la imagen
    nfil = N2-t+1  # Cantidad de patches en cada fila de la imagen
    listpat = np.zeros([ncol, nfil, t**2])  # Bloque de patches
    
    for i in xrange(ncol):
        for j in xrange(nfil):
            pat = np.reshape(im[i:i+t, j:j+t], t**2)  # Patch de la imagen
            listpat[i, j, :] = pat
    
    return listpat


# BUSQUEDA LOCAL DE PATCHES SIMILARES (solo en nlbayes2)
def find_neighbors2(indpat, listpat, t, k, method, weights=[1,1]):
    '''Find the k nearest neighbors of the patch with indices indpat
    between the rows of listpat.
    It uses weighted band coefficients in wnlbayes1 (if provided)'''
    
    N1, N2 = listpat.shape[0:2]
    
    sW = 7 * t  # The size of the LOCAL search window
    i, j = indpat
    pat = listpat[i, j, :]
    
    if i<N1-4*t:
        fil = max(0, i-3*t)  # Beginning of search window
    else:
        fil = N1-7*t
    
    if j<N2-4*t:
        col = max(0, j-3*t)  # Beginning of search window
    else:
        col = N2-7*t
    
    searchlist = listpat[fil:fil+6*t+1, col:col+6*t+1, :].reshape(((sW-t+1)**2, t**2))
    
    if method=='sklearn':
        # kNN search: Using scikit-learn kNN
        nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(searchlist)
        indices = nbrs.kneighbors(pat)[1]
    elif method=='flann':
        # kNN search: Using FLANN
        flann = FLANN()
        params = flann.build_index(searchlist, algorithm="kmeans", target_precision=1, log_level = "info")
        indices = flann.nn_index(pat, k, checks=params["checks"])[0]
    
    return searchlist[indices, :][0]


# BUSQUEDA LOCAL DE PATCHES SIMILARES (solo en wnlbayes2)
def find_neighbors3(indpat, listsearch, listpatches, t, k, tau=0):
    '''Find the k nearest neighbors of the patch with indices indpat
    between the rows of listpat.
    It uses weighted band coefficients in wnlbayes2 (if provided)'''
    
    n = 4*t**2  # Lenght of the patches in wavelet domain (n = 4xtxt)
    N1, N2 = listpatches.shape[0:2]
    
    Nt = 7
    sW = Nt * t  # The size of the LOCAL search window
    i, j = indpat
    pat = listsearch[i, j, :, 0]
    
    # Top-left index of search window:
    if i<N1-(Nt+1)/2*t:
        fil = max(0, i-(Nt-1)/2*t)  # Beginning of search window
    else:
        fil = N1-Nt*t
    
    if j<N2-(Nt+1)/2*t:
        col = max(0, j-(Nt-1)/2*t)  # Beginning of search window
    else:
        col = N2-Nt*t
    
    searchlist = conc([listsearch[fil:fil+(Nt-1)*t+1, col:col+(Nt-1)*t+1, :, dim].reshape(((sW-t+1)**2, n)) for dim in xrange(4)])
    
    # kNN search: Using scikit-learn kNN
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='brute').fit(searchlist)
    distances, indices = nbrs.kneighbors(pat)
    
    indices = indices[0]
    distances = distances[0]
    
    if tau:  # Second step: keep variable number of neighbors
        kmin = 60  # Minimum number of neighbors to be used
        distances /= n
        aux = (distances < tau)  # Keep the neighbors closer to tau
        if aux.sum() > kmin:
            indices = indices[aux]
        else:
            indices = indices[:kmin]
    
    listpat = conc([listpatches[fil:fil+(Nt-1)*t+1, col:col+(Nt-1)*t+1, :, dim].reshape(((sW-t+1)**2, n)) for dim in xrange(4)])
    
    return listpat[indices, :]


def inv_semidef_projection(A, epsilon=0):
    # Returns the projection of matrix A on positive semidefinite subspace
    
    n = A.shape[1]

    ev, V = la.eigh(A)  # ev: eigvalues, V: eigvectors
    if np.sum(ev<epsilon) > 0:
        #print 'Covariance correction!'
        neweigs = np.fmax(epsilon, ev)  # Correct all eigvalues below epsilon 
        #neweigs = ev + epsilon  # Correct eigvalues
        w = 1. / neweigs
        B = np.dot(np.dot(V, np.diag(w)), la.inv(V))  # Inverse of projected matrix
        det = np.prod(neweigs)
        return B, det
    else:
        det = np.prod(ev)
        return la.inv(A), det
    #return la.inv(A), 1


def psnr(im1, im2, dynamic_range=255):
    '''Calculate de PSNR between images im1 and im2'''
    
    # Calculate MSE, mean square error:
    mseImage = (im1.astype(float) - im2.astype(float)) ** 2
    rows, columns = im1.shape
    mse = (mseImage / (rows * columns)).sum()

    # Calculate PSNR (Peak Signal to noise ratio)
    return 10 * np.log10( dynamic_range**2 / mse)


def calc_cov(list):
    '''Evaluate the covariance of the list'''
    M = list.size
    return np.sqrt( M/(M-1)*((list**2).sum()/M - (list.sum()/M)**2) )


###################### Non-Local Bayes ######################

def restore_patch_nlbayes1(p, t, sigma, vecinos, flat):
    ''' Restaura patch con NLBayes SIN USAR ORACULO (CORRECCION DE COVARIANZA)
    vecinos: los k vecinos mas cercanos a utilizar para restaurar el patch
    conf: confianza de la estimacion pres (valor del funcional de NlBayes a minimizar)'''

    pbar = np.mean(vecinos, 0)  # Neighbors mean
    if flat:
        pixels = np.reshape(vecinos, (1,-1))
        sigmaP = calc_cov(pixels)
        if sigmaP < 1.1*sigma:
            return pbar
    
    Sigma = np.cov(vecinos.T) # Covariance matrix
    invSigma, det = inv_semidef_projection(Sigma)
    pres = pbar + np.dot(np.dot(Sigma-(0.8*sigma**2)*np.eye(t**2), invSigma), p - pbar)
    conf = np.sum((pres-p)**2)/sigma**2 + np.dot(np.dot(p-pbar, invSigma), p-pbar) + np.log(det)/2
    conf = np.exp(-conf)

    return pres#, conf


def restore_patch_nlbayes2(p, t, sigma, vecinos, flat):
    '''Restaura patch con NLBayes USANDO ORACULO (SIN CORRECCION DE COVARIANZA)
    vecinos: los k vecinos mas cercanos a utilizar para restaurar el patch'''

    pbar = np.mean(vecinos, 0)  # Neighbors mean
    if flat:
        pixels = np.reshape(vecinos, (1,-1))
        sigmaP = calc_cov(pixels)
        if sigmaP < 1*sigma:
            return pbar
    
    Sigma = np.cov(vecinos.T) # Covariance matrix
    invSigma, det = inv_semidef_projection(Sigma + (1*sigma**2)*np.eye(t**2))
    pres = pbar + np.dot(np.dot(Sigma, invSigma), p - pbar)
    conf = np.sum((pres-p)**2)/sigma**2 + np.dot(np.dot(p-pbar, invSigma), p-pbar) + np.log(det)/2
    conf = np.exp(-conf)
    
    return pres#, conf
   

# BUSQUEDA LOCAL DE PATCHES SIMILARES
def nlbayes2(im, t, sigma, knn='sklearn', flat=True, borde=8, oracle=np.zeros((1,))):
    '''Restore image 'im' using NLBayes with noise sigma and patch size txt
    If oracle is provided (in second step), it's used for build the gaussian model'''
    
    im = aumentar_imagen(im, borde)
    
    # Parameters:
    N1, N2 = im.shape  # Dimensions of im

    # List of patches of im:
    listpat = gen_listpat2(im, t)
    
    if not oracle.any():  # No oracle provided
        k = 90  # Fijo para comparar con otros metodos
        use_oracle = False        
        listsearch = listpat
    else:  # An oracle is provided
        k = 90  # Fijo para comparar con otros metodos
        use_oracle = True
        oracle = aumentar_imagen(oracle, borde)
        listsearch = gen_listpat2(oracle, t)

    # Restore image:
    imres = np.zeros([N1, N2])
    
    for i in xrange(N1-t+1):
        for j in xrange(N2-t+1):
            
            p = listpat[i, j, :]
            vecinos = find_neighbors2([i,j], listsearch, t, k, knn)
            
            if not use_oracle:  # No provided oracle
                pres = restore_patch_nlbayes1(p, t, sigma, vecinos, flat)
            else:  # If oracle is provided, it's used for the gaussian patch model:
                pres = restore_patch_nlbayes2(p, t, sigma, vecinos, flat)
            imres[i:i+t, j:j+t] += np.reshape(pres, (t, t))  # Aggregation

    # Normalizing aggregation
    u = conc(([range(1,t)], t*np.ones([1,N1-2*t+2]), [range(t-1,0,-1)]),1)
    v = conc(([range(1,t)], t*np.ones([1,N2-2*t+2]), [range(t-1,0,-1)]),1)
    mask = u.T*v  # 'mask' contains number of estimations of each pixel
    imres = np.divide(imres, mask)
    
    if borde:
        imres = recortar_imagen(imres, borde)

    return imres


###################### Wavelet-Non-Local Bayes ######################

# To learn how to use pywavelets, see http://www.pybytes.com/pywavelets/ref/2d-dwt-and-idwt.html
# More details of Wavelets: http://wavelets.pybytes.com/wavelet/db8/

def wav_analysis(im, wavelet, niv):
    '''Analisis de la imagen im en niv niveles
    size(im) debe ser multiplo de 2**niv en cada dimension'''
    
    n, m = im.shape
    W = im.copy()
    
    for i in range(niv):
        LL, (HL, LH, HH) = pywt.dwt2(W[:n,:m], wavelet, mode='per')
        #size(LL), size(HL), size(LH), size(HH)
        W[:n,:m] = conc((conc((LL, HL),1), conc((LH, HH), 1)))
        n /= 2
        m /= 2
    
    return W


def wav_synthesis(W, wavelet, niv):
    '''Sintesis de los coef W en niv niveles'''
    
    n, m = np.array(W.shape)/2**(niv-1)
    imagen = W.copy()
    
    for i in xrange(niv):
        LL = imagen[:n/2,:m/2]
        HL = imagen[:n/2,m/2:m]
        LH = imagen[n/2:n,:m/2]
        HH = imagen[n/2:n,m/2:m]
        imagen[:n,:m] = pywt.idwt2((LL,(HL,LH,HH)), wavelet, mode='per')
        n *= 2
        m *= 2
    
    return imagen


def traslaciones_wav(W, n):
    '''Coeficientes de las 3 traslaciones (de n pixeles c/u)
    W tiene que tener solo 1 nivel
    Wh: traslacion horizontal
    Wv: traslacion vertical
    Wd: traslacion diagonal'''
    
    #% Reconstruimos imagen (1 nivel):
    imagen = wav_synthesis(W, 'db8', 1)
    
    # Trasladamos imagen espacial:
    imH = pad(imagen, ((0,0),(n,0)), mode='reflect')[:,:-n]
    imV = pad(imagen, ((n,0),(0,0)), mode='reflect')[:-n,:]
    imD = pad(imagen, ((n,0),(n,0)), mode='reflect')[:-n,:-n]
    
    # Coeficientes de las traslaciones:
    Wh = wav_analysis(imH, 'db8', 1)
    Wv = wav_analysis(imV, 'db8', 1)
    Wd = wav_analysis(imD, 'db8', 1)
    
    Wblock = np.zeros(conc((W.shape,(4,))))
    Wblock[:,:,0] = W
    Wblock[:,:,1] = Wh
    Wblock[:,:,2] = Wv
    Wblock[:,:,3] = Wd

    return Wblock


def gen_listpat_3D2(Wblock,t):
    ''' Genera bloque de patches (como filas de listpat3D) de la imagen W
    Wblock contiene W, Wh, Wv, Wd apiladas (en 3ra dimension)
    W son los coef wavelets originales
    Wh, Wv, Wd son los coef de las traslaciones horizontal, vertical y
        diagonal respectivamente
    Filas de listpat3D(:,:,i) son los patches de la traslacion i
    i=1 -> Imagen original, i=2 -> Wh, i=3 -> Wv, i=4 -> Wd
    Se usa en: WNLBayes2'''
    
    N1, N2 = Wblock[:,:,0].shape  # Shape of the image
    N1 /= 2; N2 /= 2  # Size of each band
    ncol = N1-t+1  # Cantidad de patches en cada columna de c/banda
    nfil = N2-t+1  # Cantidad de patches en cada fila de c/banda

    listpat3D = np.zeros([ncol, nfil, 4*t**2, 4])  # Bloque de patches (filas)
    
    for dim in xrange(4):
        Wx = Wblock[:,:,dim]  # Traslacion x in {0, h, v, d}
        for i in xrange(ncol):
            for j in xrange(nfil):
                pat0 = np.reshape(Wx[i:i+t,j:j+t],t**2)  # Patch del resumen
                patH = np.reshape(Wx[i:i+t,N2+j:N2+j+t],t**2)  # Patch de la banda H
                patV = np.reshape(Wx[N1+i:N1+i+t,j:j+t],t**2)  # Patch de la banda V
                patD = np.reshape(Wx[N1+i:N1+i+t,N2+j:N2+j+t],t**2)  # Patch de la banda D
                listpat3D[i, j, :, dim] = conc((pat0,patH,patV,patD))
    
    return listpat3D
    
# PRIMER PASADA, SIN ORACULO (NOISY PATCHES)
def restore_patch_wnlbayes1(p, t, sigma, vecinos, flat, umb=[0.1, 0.1, 0.1, 0.1]):
    ''' Restaura patch con WNLBayes
    vecinos: los k vecinos mas cercanos a utilizar para restaurar el patch
    The flat patch trick is done separately in each band'''

    pbar = np.mean(vecinos, 0)  # Neighbors mean
    if flat:
        n = t**2  # Size of the summary
        pixRES = np.reshape(vecinos[:,:n], -1)  # All coefficients used for flat area estimation
        pixH  = np.reshape(vecinos[:,n:2*n], -1)
        pixV  = np.reshape(vecinos[:,2*n:3*n], -1)
        pixD  = np.reshape(vecinos[:,3*n:], -1)
        
        sigmaRES = calc_cov(pixRES)
        sigmaH = calc_cov(pixH)
        sigmaV = calc_cov(pixV)
        sigmaD = calc_cov(pixD)
        
        if (sigmaRES < umb[0]*sigma**2) and (sigmaH < umb[1]*sigma**2) and (sigmaV < umb[2]*sigma**2) and (sigmaD < umb[3]*sigma**2):
            return pbar, 1

    Sigma = np.cov(vecinos.T) # Covariance matrix
    beta1 = 1
    invSigma, det = inv_semidef_projection(Sigma, epsilon=250)
    pres = pbar + np.dot(np.dot(Sigma-(beta1*sigma**2)*np.eye(4*t**2), invSigma), p - pbar)
    #pres = p - beta1*sigma**2*np.dot(invSigma, p - pbar)
    conf = np.sum((pres-p)**2)/sigma**2 + np.dot(np.dot(p-pbar, invSigma), p-pbar) + np.log(det)/2
    conf = np.exp(-conf/100)
    
    return pres, conf


# SEGUNDA PASADA, CON ORACULO (CLEAN PATCHES)
def restore_patch_wnlbayes2(p, t, sigma, vecinos, flat):
    ''' Restaura patch con WNLBayes
    vecinos: los k vecinos mas cercanos a utilizar para restaurar el patch
    The flat patch trick is done separately in each band'''

    pbar = np.mean(vecinos, 0)  # Neighbors mean
    if flat:
        n = t**2  # Size of the summary
        #pixels = np.reshape(vecinos[:,n], (1,-1))  # Only summary is used for variance estimation
        pixels = np.reshape(vecinos, -1)
        pixels[n:] *= 1
        sigmaP = calc_cov(pixels)
        if sigmaP < 1.1*sigma**2:
            return pbar, 1
    
    Sigma = np.cov(vecinos.T) # Covariance matrix
    beta2 = 1
    invSigma, det = inv_semidef_projection(Sigma + (beta2*sigma**2)*np.eye(4*t**2), epsilon=0)
    pres = pbar + np.dot(np.dot(Sigma, invSigma), p - pbar)
    conf = np.sum((pres-p)**2)/sigma**2 + np.dot(np.dot(p-pbar, invSigma), p-pbar) + np.log(det)/2
    conf = np.exp(-conf/100)
    
    return pres, conf


# Agregacion simple (promedio de restauraciones)
def wnlbayes_1nivel2(W, sigma, t, l, denoise_sum, flat, weights, Woracle=np.zeros((1,))):
    '''Restaura la imagen correspondiente a W (que tiene solo 1 nivel) usando
    nlbayes en dominio wavelet con ruido sigma y tamanho t de cada patch
    l es el largo de la traslacion
    Si denoise_sum == True, se filtra resumen, en caso contrario se deja el
    original'''
    
    N1, N2 = W.shape  # Dimensiones de la imagen
    Wres = np.zeros((N1, N2))  # Coeficientes restaurados
    N1 /= 2
    N2 /= 2  # Dimensiones de cada banda
    tau = 15
    
    # Generamos traslaciones de la imagen original:
    Wblock = traslaciones_wav(W, l)
    listpat3D = gen_listpat_3D2(Wblock, t)
    
    if not Woracle.any():  # No oracle provided
        k = 90  # Fijo para comparar con otros metodos
        use_oracle = False        
        listpatches = listpat3D
        listsearch = softthreshold_bandas(listpatches, weights, t)
    else:  # An oracle is provided
        k = 90  # Fijo para comparar con otros metodos
        use_oracle = True
        WblockOR = traslaciones_wav(Woracle, l)
        listpatches = gen_listpat_3D2(WblockOR, t)
        listsearch = listpatches
    
    #listsearch = softthreshold_bandas(listpatches, weights, t)
    #listsearch = ponderar_bandas(listpatches, weights, t)

    for i in xrange(N1-t+1):
        for j in xrange(N2-t+1):
            p = listpat3D[i,j,:,0]  # Patch a restaurar
            
            if not use_oracle:
                vecinos = find_neighbors3([i,j], listsearch, listpatches, t, k)
                pres, conf = restore_patch_wnlbayes1(p, t, sigma, vecinos, flat)  # WNLBayes                
            else:
                vecinos = find_neighbors3([i,j], listsearch, listpatches, t, k)#, tau=tau)
                pres, conf = restore_patch_wnlbayes2(p, t, sigma, vecinos, flat)  # WNLBayes
            
            # Aggregation
            f1 = range(i, i+t)        # Intevalo de filas de la posicion de los patches en resumen y banda H
            f2 = range(N1+i, N1+i+t)  # Intevalo de filas de la posicion de los patches en banda V y banda D
            c1 = range(j, j+t)        # Intevalo de columnas de la posicion de los patches en resumen y banda V
            c2 = range(N2+j, N2+j+t)  # Intevalo de columnas de la posicion de los patches en resumen y banda V
    
            if denoise_sum:
                Wres[np.ix_(f1,c1)] += np.reshape(pres[:t**2], (t,t)) # Agregamos patch restaurado al resumen
    
            Wres[np.ix_(f1,c2)] += np.reshape(pres[t**2:2*t**2], (t,t)) # Agregamos patch restaurado a banda H
            Wres[np.ix_(f2,c1)] += np.reshape(pres[2*t**2:3*t**2], (t,t)) # Agregamos patch restaurado a banda V
            Wres[np.ix_(f2,c2)] += np.reshape(pres[3*t**2:], (t,t)) # Agregamos patch restaurado a banda D
    
    # Normalizing aggregation
    u = conc(([range(1, t)], t*np.ones([1, N1-2*t+2]), [range(t-1, 0, -1)]), 1)
    v = conc(([range(1, t)], t*np.ones([1, N2-2*t+2]), [range(t-1, 0, -1)]), 1)
    mask = repmat(u.T*v,2,2); # Mask contiene cant de estimaciones de cada pixel
    Wres = np.divide(Wres, mask)
    if not denoise_sum:
        Wres[:N1,:N2] = W[:N1,:N2]
    
    return Wres


# Agregacion ponderada por la confianza en la restauracion del patch
def wnlbayes_1nivel2b(W, sigma, t, l, denoise_sum, flat, weights, Woracle=np.zeros((1,))):
    '''Restaura la imagen correspondiente a W (que tiene solo 1 nivel) usando
    nlbayes en dominio wavelet con ruido sigma y tamanho t de cada patch
    l es el largo de la traslacion
    Si denoise_sum == True, se filtra resumen, en caso contrario se deja el
    original'''
    
    N1, N2 = W.shape  # Dimensiones de la imagen
    Wres = np.zeros((N1, N2))  # Coeficientes restaurados
    mask = np.zeros((N1, N2))  # Confianza de cada patch restaurado (agregacion)
    N1 /= 2
    N2 /= 2  # Dimensiones de cada banda
    tau = 15
    
    # Generamos traslaciones de la imagen original:
    Wblock = traslaciones_wav(W, l)
    listpat3D = gen_listpat_3D2(Wblock, t)
    
    if not Woracle.any():  # No oracle provided
        k = 90  # Fijo para comparar con otros metodos
        use_oracle = False        
        listpatches = listpat3D
        listsearch = softthreshold_bandas(listpatches, weights, t)
    else:  # An oracle is provided
        k = 90  # Fijo para comparar con otros metodos
        use_oracle = True
        WblockOR = traslaciones_wav(Woracle, l)
        listpatches = gen_listpat_3D2(WblockOR, t)
        listsearch = listpatches
    
    #listsearch = softthreshold_bandas(listpatches, weights, t)
    #listsearch = ponderar_bandas(listpatches, weights, t)

    for i in xrange(N1-t+1):
        for j in xrange(N2-t+1):
            p = listpat3D[i,j,:,0]  # Patch a restaurar
            
            if not use_oracle:
                vecinos = find_neighbors3([i,j], listsearch, listpatches, t, k)
                pres, conf = restore_patch_wnlbayes1(p, t, sigma, vecinos, flat)  # WNLBayes                
            else:
                vecinos = find_neighbors3([i,j], listsearch, listpatches, t, k)#, tau=tau)
                pres, conf = restore_patch_wnlbayes2(p, t, sigma, vecinos, flat)  # WNLBayes
            
            # Aggregation
            f1 = range(i, i+t)        # Intevalo de filas de la posicion de los patches en resumen y banda H
            f2 = range(N1+i, N1+i+t)  # Intevalo de filas de la posicion de los patches en banda V y banda D
            c1 = range(j, j+t)        # Intevalo de columnas de la posicion de los patches en resumen y banda V
            c2 = range(N2+j, N2+j+t)  # Intevalo de columnas de la posicion de los patches en resumen y banda V
    
            if denoise_sum:
                Wres[np.ix_(f1,c1)] += conf * np.reshape(pres[:t**2], (t,t)) # Agregamos patch restaurado al resumen
            mask[np.ix_(f1,c1)] += conf
    
            Wres[np.ix_(f1,c2)] += conf * np.reshape(pres[t**2:2*t**2], (t,t)) # Agregamos patch restaurado a banda H
            Wres[np.ix_(f2,c1)] += conf * np.reshape(pres[2*t**2:3*t**2], (t,t)) # Agregamos patch restaurado a banda V
            Wres[np.ix_(f2,c2)] += conf * np.reshape(pres[3*t**2:], (t,t)) # Agregamos patch restaurado a banda D
            mask[np.ix_(f1,c2)] += conf
            mask[np.ix_(f2,c1)] += conf
            mask[np.ix_(f2,c2)] += conf
    
    # Normalizing aggregation
    Wres = np.divide(Wres, mask)
    if not denoise_sum:
        Wres[:N1,:N2] = W[:N1,:N2]
    
    return Wres


def ponderar_bandas(listpatches, weights, t):
    listsearch = listpatches.copy()
    
    if weights!=[1,1]:
        n = 4*t**2  # Lenght of the patches in wavelet domain (n = 4xtxt)
        auxHV = range(  n/4, 3*n/4)  # Positions of WH and WV bands
        auxD  = range(3*n/4,   n  )  # Positions of WD band
        listsearch[:,:,auxHV,:] *= weights[0]
        listsearch[:,:,auxD ,:] *= weights[1]
    
    return listsearch


def hardthreshold_bandas(listpatches, umbrales, t):
    listsearch = listpatches.copy()
    n = 4*t**2  # Lenght of the patches in wavelet domain (n = 4xtxt)
    auxHV = range(  n/4, 3*n/4)  # Positions of WH and WV bands
    auxD  = range(3*n/4,   n  )  # Positions of WD band
    HVbands = listsearch[:,:,auxHV,:]
    listsearch[:,:,auxHV,:] = pywt.threshold(HVbands, umbrales[0], 'hard')
    
    Dbands  = listsearch[:,:,auxD ,:]
    listsearch[:,:,auxD,:] = pywt.threshold(Dbands, umbrales[1], 'hard')
    
    return listsearch


def softthreshold_bandas(listpatches, umbrales, t):
    listsearch = listpatches.copy()
    n = 4*t**2  # Lenght of the patches in wavelet domain (n = 4xtxt)
    auxHV = range(  n/4, 3*n/4)  # Positions of WH and WV bands
    auxD  = range(3*n/4,   n  )  # Positions of WD band
    HVbands = listsearch[:,:,auxHV,:]
    listsearch[:,:,auxHV,:] = pywt.threshold(HVbands, umbrales[0], 'soft')
    
    Dbands  = listsearch[:,:,auxD ,:]
    listsearch[:,:,auxD,:] = pywt.threshold(Dbands, umbrales[1], 'soft')
    
    return listsearch


# PROMEDIA DENOISING DE TRASLACIONES PARA REDUCIR BIAS --> NO RESUELVE PSEUDO-GIBBS
def wnlbayes_1nivel_tras(Wn, sigma, t, l, wavelet, denoise_sum, knn='sklearn', weights=[1,1], flat=True, Woracle=np.zeros((1,))):
    '''Apica wnlbayes_1nivel2 a la imagen y sus 4 traslaciones y luego
    las promedia (para eliminar bias introducido por cada una)'''
    
    traslations = [1]  # Largo de las traslaciones para cada imagen
    
    imn = wav_synthesis(Wn, wavelet, 1)
    
    imres = np.zeros(Wn.shape)
    cant_tras = 0
    
    for n in traslations:
        cant_tras += 1
        imnH = pad(imn, ((0,0),(n,0)), mode='reflect')[:,:-n]
        imnV = pad(imn, ((n,0),(0,0)), mode='reflect')[:-n,:]
        imnD = pad(imn, ((n,0),(n,0)), mode='reflect')[:-n,:-n]
        
        WnH = wav_analysis(imnH, wavelet, 1)
        WnV = wav_analysis(imnV, wavelet, 1)
        WnD = wav_analysis(imnD, wavelet, 1)
        
        if Woracle.any():
            oracle = wav_synthesis(Woracle, wavelet, 1)
            oracleH = pad(oracle, ((0,0),(n,0)), mode='reflect')[:,:-n]
            oracleV = pad(oracle, ((n,0),(0,0)), mode='reflect')[:-n,:]
            oracleD = pad(oracle, ((n,0),(n,0)), mode='reflect')[:-n,:-n]
            
            WoracleH = wav_analysis(oracleH, wavelet, 1)
            WoracleV = wav_analysis(oracleV, wavelet, 1)
            WoracleD = wav_analysis(oracleD, wavelet, 1)
        else:
            WoracleH = Woracle
            WoracleV = Woracle
            WoracleD = Woracle
        
        Wres = wnlbayes_1nivel2(Wn, sigma, t, l, denoise_sum, flat, weights, Woracle)
        imWNLB1 = wav_synthesis(Wres, wavelet, 1)
        
        WresH = wnlbayes_1nivel2(WnH, sigma, t, l, denoise_sum, flat, weights, WoracleH)
        imWNLBH = wav_synthesis(WresH, wavelet, 1)
        
        WresV = wnlbayes_1nivel2(WnV, sigma, t, l, denoise_sum, flat, weights, WoracleV)
        imWNLBV = wav_synthesis(WresV, wavelet, 1)
        
        WresD = wnlbayes_1nivel2(WnD, sigma, t, l, denoise_sum, flat, weights, WoracleD)
        imWNLBD = wav_synthesis(WresD, wavelet, 1)
        
        imWNLBH = pad(imWNLBH, ((0,0),(0,n)), mode='reflect')[:,n:]
        imWNLBV = pad(imWNLBV, ((0,n),(0,0)), mode='reflect')[n:,:]
        imWNLBD = pad(imWNLBD, ((0,n),(0,n)), mode='reflect')[n:,n:]
        imres += (imWNLB1 + imWNLBH + imWNLBV + imWNLBD) / 4
        
    return imres/cant_tras


# APLICA TRASLACIONES PARA REDUCIR BIAS EN 2DO NIVEL
def wnlbayes3(im, sigma, wavelet, borde=16, flat=False, oracle=np.zeros((1,))):
    '''Restaura la imagen correspondiente a W (con 3 niveles) usando
    nlbayes en dominio wavelet con ruido sigma'''

    im = aumentar_imagen(im, borde)
    W = wav_analysis(im, wavelet, 3)
    
    N1, N2 = np.array(W.shape)/4  # Dimensiones de la imagen del 3er nivel
    imres = W.copy()
    
    if not oracle.any():  # No oracle provided
        use_oracle = False
        Woracle = oracle
    else:  # An oracle is provided
        use_oracle = True
        oracle = aumentar_imagen(oracle, borde)
        WOR = wav_analysis(oracle, wavelet, 3)
        Woracle = WOR[:N1, :N2]  # First oracle to be used

    for n in xrange(3, 0, -1):  # Para cada nivel
        l = 1  #2**(n-1)  # Largo de las traslaciones
        
        if n == 3:
            if use_oracle:
                denoise_sum = 1
                t = 2  # Ventana variable segun nivel
            else:
                denoise_sum = 1
                t = 2  # Ventana variable segun nivel
            flat2 = False
        elif n == 2:
            if use_oracle:
                denoise_sum = 0
                t = 3  # Ventana variable segun nivel
            else:
                denoise_sum = 0
                t = 3  # Ventana variable segun nivel
            flat2 = False
        else:
            if use_oracle:
                denoise_sum = 0
                t = 3  # Ventana variable segun nivel
            else:
                denoise_sum = 0
                t = 3  # Ventana variable segun nivel
            flat2 = flat
        
        weights = [sigma,sigma]
        
        Waux = wnlbayes_1nivel2(imres[:N1, :N2], sigma, t, l, denoise_sum, flat2, weights, Woracle)
        imres[:N1, :N2] = wav_synthesis(Waux, wavelet, 1)
 
        if use_oracle:
            WOR[:N1, :N2] = wav_synthesis(WOR[:N1, :N2], wavelet, 1)
            Woracle = WOR[:2*N1, :2*N2]
            
        N1 *= 2
        N2 *= 2
    
    if borde:
        imres = recortar_imagen(imres, borde)
    
    return imres