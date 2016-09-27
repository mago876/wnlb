import numpy as np
from scipy import ndimage as ndi
from scipy.misc import imsave
import matplotlib.pyplot as plt
#from skimage.measure import compare_psnr
from wnlbayes_func import *
from time import time
#from skimage.restoration import nl_means_denoising as nlmski


###############################################################################
#
#  Inicializamos datos
#
##############################################################################

    
# Parameters:
sigma = 20
#wavelet = 'db8'  # Orthogonal wavelet
wavelet = 'bior4.4'  # Biorthogonal wavelet

# Load image:
im = ndi.imread('images/lenna_1024.png').astype('float64')
#im = ndi.imread('images/imtest2.png').astype('float64')
#im = 128*np.ones((256,256))  # Flat image

# Add noise:
#imn = im + sigma*np.random.standard_normal(im.shape)
# or load a noisy image:
imn = ndi.imread('images/lenna_1024_noisy20.png').astype('float64')


# Show images
#plt.figure()
#psnr1 = psnr(im, imn)
#title1 = 'Noisy image - PSNR = ' + "{0:.3f}".format(psnr1)
#plt.subplot(121), plt.imshow(im, cmap='gray', interpolation='nearest'), plt.title('Original image')
#plt.subplot(122), plt.imshow(imn, cmap='gray', interpolation='nearest'), plt.title(title1)
#plt.show()



###############################################################################
#
#  Algoritmos de denoising
#
###############################################################################


######### NLmeans ##########

#t=5
#
#tic = time()
#imNLM = nlmeans(imn, t, knn='sklearn')
#toc = time()
#print 'Time of NLMeans (global):', toc-tic
#
#tic = time()
#imNLM2 = nlmeans2(imn, t, knn='sklearn')
#toc = time()
#print 'Time of NLMeans2 (local):', toc-tic
#
##Results:
#plt.figure()
#psnr2 = psnr(im, imNLM)
#psnr22 = psnr(im, imNLM2)
#title2 = 'NLMeans (global) - PSNR = ' + str(psnr2)
#title22 = 'NLMeans (local)- PSNR = ' + str(psnr22)
##plt.subplot(131), plt.imshow(im, cmap='gray', interpolation='nearest'), plt.title('Original image')
##plt.subplot(132), plt.imshow(imn, cmap='gray', interpolation='nearest'), plt.title(title1)
#plt.subplot(121), plt.imshow(imNLM, cmap='gray', interpolation='nearest'), plt.title(title2)
#plt.subplot(122), plt.imshow(imNLM2, cmap='gray', interpolation='nearest'), plt.title(title22)
#plt.show()


############ NLMeans de Scikit-image ##########

#t=5
#tic = time()
#imNLM2 = nlmski(imn, t, h=sigma, fast_mode=False)
#toc = time()
#print 'Time of NLMeans (skimage):', toc-tic


########## NLBayes ##########

#t=5
#
## Primer pasada para usar como oraculo
#tic = time()
#imNLB = nlbayes2(imn, t, sigma, knn='sklearn', flat=True)
#toc = time()
#print 'Time of NLBayes (1ra pasada):', toc-tic
#
#tic = time()
#imNLB2 = nlbayes2(imn, t, sigma, knn='sklearn', flat=False, oracle=imNLB)
#toc = time()
#print 'Time of NLBayes (2da pasada):', toc-tic
#
## Results:
#plt.figure()
#psnr3 = psnr(im, imNLB)
#psnr4 = psnr(im, imNLB2)
#title3 = 'NLBayes (1ra pasada) - PSNR = ' + "{0:.3f}".format(psnr3)
#title4 = 'NLBayes (2da pasada) - PSNR = ' + "{0:.3f}".format(psnr4)
#plt.subplot(121), plt.imshow(im, cmap='gray', interpolation='nearest'), plt.title('Original image')
#plt.subplot(122), plt.imshow(imn, cmap='gray', interpolation='nearest'), plt.title(title1)
#plt.subplot(121), plt.imshow(imNLB, cmap='gray', interpolation='nearest'), plt.title(title3)
#plt.subplot(122), plt.imshow(imNLB2, cmap='gray', interpolation='nearest'), plt.title(title4)
#plt.show()


############## WNLBayes (1 nivel) ##########

nivel = 1
Wn = wav_analysis(imn, wavelet, nivel)

t=3  # Tamanho patch 1ra pasada
tic = time()
Wres = wnlbayes_1nivel2b(Wn, sigma, t, 1, denoise_sum=True, flat=False, weights=[sigma,sigma])
imres = wav_synthesis(Wres, wavelet, nivel)
toc = time()
print 'Time of WNLBayes (1 nivel, 1ra pasada):', toc-tic

t=5 # Tamanho patch 2da pasada
tic = time()
Wres2 = wnlbayes_1nivel2b(Wn, sigma, t, 1, denoise_sum=True, flat=False, weights=[0,0], Woracle=Wres)
imres2 = wav_synthesis(Wres2, wavelet, nivel)
toc = time()
print 'Time of WNLBayes (1 nivel, 2da pasada):', toc-tic

# Results:
plt.figure()
psnr4 = psnr(im, imres)
psnr5 = psnr(im, imres2)
title4 = 'WNLBayes (1 nivel, 1ra pasada) - PSNR = ' + "{0:.3f}".format(psnr4)
title5 = 'WNLBayes (1 nivel, 2da pasada) - PSNR = ' + "{0:.3f}".format(psnr5)
#plt.subplot(131), plt.imshow(im, cmap='gray', interpolation='nearest'), plt.title('Original image')
#plt.subplot(121), plt.imshow(imn, cmap='gray', interpolation='nearest'), plt.title(title1)
plt.subplot(121), plt.imshow(imres, cmap='gray', interpolation='nearest'), plt.title(title4)
plt.subplot(122), plt.imshow(imres2, cmap='gray', interpolation='nearest'), plt.title(title5)
plt.show()


############# WNLBayes (1 nivel) - CYCLE SPINNING ##########

#nivel = 1
#Wn = wav_analysis(imn, wavelet, nivel)
#
#t=3  # Tamanho patch 1ra pasada
#tic = time()
#Wres = wnlbayes_1nivel_tras(Wn, sigma, t, 1, wavelet, denoise_sum=True, flat=False, weights=[sigma,sigma])
##imres = wav_synthesis(Wres, wavelet, nivel)
#imres = Wres
#toc = time()
#print 'Time of WNLBayes (1 nivel, 1ra pasada):', toc-tic
#
#t=5 # Tamanho patch 2da pasada
#tic = time()
#Wres = wav_analysis(imres, wavelet, 1)
#Wres2 = wnlbayes_1nivel_tras(imn, sigma, t, 1, wavelet, denoise_sum=True, flat=False, weights=[0,0], Woracle=Wres)
##imres2 = wav_synthesis(Wres2, wavelet, nivel)
#imres2 = Wres2
#toc = time()
#print 'Time of WNLBayes (1 nivel, 2da pasada):', toc-tic
#
## Results:
#plt.figure()
#psnr4 = psnr(im, imres)
#psnr5 = psnr(im, imres2)
#title4 = 'WNLBayes (1 nivel, 1ra pasada) - PSNR = ' + "{0:.3f}".format(psnr4)
#title5 = 'WNLBayes (1 nivel, 2da pasada) - PSNR = ' + "{0:.3f}".format(psnr5)
##plt.subplot(131), plt.imshow(im, cmap='gray', interpolation='nearest'), plt.title('Original image')
##plt.subplot(121), plt.imshow(imn, cmap='gray', interpolation='nearest'), plt.title(title1)
#plt.subplot(121), plt.imshow(imres, cmap='gray', interpolation='nearest'), plt.title(title4)
#plt.subplot(122), plt.imshow(imres2, cmap='gray', interpolation='nearest'), plt.title(title5)
#plt.show()


######## WNLBayes (3 niveles) ##########

tic = time()
imWNLB3 = wnlbayes3(imn, sigma, wavelet, flat=False)
toc = time()
print 'Time of WNLBayes (3 niveles, 1ra pasada):', toc-tic

tic = time()
imWNLB4 = wnlbayes3(imn, sigma, wavelet, flat=False, oracle=imWNLB3)
toc = time()
print 'Time of WNLBayes (3 niveles, oraculo=im):', toc-tic

# Results:
plt.figure()
psnr6 = psnr(im, imWNLB3)
psnr7 = psnr(im, imWNLB4)
#psnr8 = psnr(im, imWNLB5)
title6 = 'WNLBayes (3 niveles, 1ra pasada) - PSNR = ' + "{0:.3f}".format(psnr6)
title7 = 'WNLBayes (3 niveles, 2da pasada) - PSNR = ' + "{0:.3f}".format(psnr7)
#title8 = 'WNLBayes (3 niveles, 2da pasada por nivel) - PSNR = ' + "{0:.3f}".format(psnr8)
#plt.subplot(221), plt.imshow(im, cmap='gray', interpolation='nearest'), plt.title('Original image')
#plt.subplot(222), plt.imshow(imn, cmap='gray', interpolation='nearest'), plt.title(title1)
plt.subplot(121), plt.imshow(imWNLB3, cmap='gray', interpolation='nearest'), plt.title(title6)
plt.subplot(122), plt.imshow(imWNLB4, cmap='gray', interpolation='nearest'), plt.title(title7)
#plt.subplot(133), plt.imshow(imWNLB5, cmap='gray', interpolation='nearest'), plt.title(title8)
plt.show()


######### WNLBayes (por nivel: 1) ##########
#
#W = wav_analysis(im, wavelet, 3)
#Wn = wav_analysis(imn, 'db8', 3)
## Primer nivel de la descomposicion:
#W1 = W[:128,:128]
#Wn1 = Wn[:128,:128]
#
#t=2  # Tamanho patch 1ra pasada
#tic = time()
#Wres = wnlbayes_1nivel2(Wn1, sigma, t, 1, denoise_sum=True, flat=False, weights=[sigma,sigma])
#imres1 = wav_synthesis(Wres, wavelet, 1)
#toc = time()
#print 'Time of WNLBayes (nivel 1, 1ra pasada):', toc-tic
#
#t=2 # Tamanho patch 2da pasada
#tic = time()
#Wres2 = wnlbayes_1nivel2(Wn1, sigma, t, 1, denoise_sum=True, flat=False, weights=[0,0], Woracle=Wres)
#imres2 = wav_synthesis(Wres2, wavelet, 1)
#toc = time()
#print 'Time of WNLBayes (nivel 1, 2da pasada):', toc-tic
#
## Results:
#im1 = wav_synthesis(W1, wavelet, 1)  # Primer nivel sin ruido
#psnr6 = psnr(im1, imres1)
#psnr7 = psnr(im1, imres2)
#title6 = 'WNLBayes (nivel 1, 1ra pasada) - PSNR = ' + "{0:.3f}".format(psnr6)
#title7 = 'WNLBayes (nivel 1, 2da pasada) - PSNR = ' + "{0:.3f}".format(psnr7)
#plt.figure()
#plt.subplot(121), plt.imshow(imres1, cmap='gray', interpolation='nearest'), plt.title(title6)
#plt.subplot(122), plt.imshow(imres2, cmap='gray', interpolation='nearest'), plt.title(title7)
#plt.show()

######## WNLBayes (por nivel: 2) ##########

#W = wav_analysis(im, wavelet, 3)
#Wn = wav_analysis(imn, 'db8', 3)
## Segundo nivel de la descomposicion:
#W2 = W[:256,:256]
#Wn2 = Wn[:256,:256].copy()
#Wn2[:128,:128] = imres1  # Primer nivel restaurado
#
#t=3  # Tamanho patch 1ra pasada
#tic = time()
#Wres2 = wnlbayes_1nivel2b(Wn2, sigma, t, 1, denoise_sum=False, flat=False, weights=[sigma,sigma])
#imres3 = wav_synthesis(Wres2, wavelet, 1)
#toc = time()
#print 'Time of WNLBayes (nivel 2, 1ra pasada):', toc-tic
#
##t=2 # Tamanho patch 2da pasada
##tic = time()
##Wres2 = wnlbayes_1nivel2(Wn1, sigma, t, 1, denoise_sum=True, flat=False, weights=[0,0], Woracle=Wres2)
##imres2 = wav_synthesis(Wres2, wavelet, 1)
##toc = time()
##print 'Time of WNLBayes (nivel 1, 2da pasada):', toc-tic
#
## Results:
#im2 = wav_synthesis(W2, wavelet, 2)  # Segundo nivel sin ruido
#psnr6 = psnr(im2, imres3)
#title6 = 'WNLBayes (nivel 2, 1ra pasada) - PSNR = ' + "{0:.3f}".format(psnr6)
#plt.figure()
#plt.subplot(121), plt.imshow(im2, cmap='gray', interpolation='nearest'), plt.title('Original')
#plt.subplot(122), plt.imshow(imres3, cmap='gray', interpolation='nearest'), plt.title(title6)
#plt.show()


########## WNLBayes (3 niveles)  -  Band weights grid search ##########

#wavelet = 'db8'
#nivel = 3
#print 'WNLBayes (3 niveles) (Bands weights grid search):'
#results = np.zeros((6,6))
#for p1 in xrange(1,6):
#    for p2 in xrange(6):
#        wei = [1, 0.4, 0, 1, 0.2*p1, 0.2*p2]
#        tic = time()
#        imWNLB = wnlbayes3(imn, sigma, weights=wei, flat=True)#, oracle=imWNLB3)
#        toc = time()
#        ps = psnr(im, imWNLB)
#        results[p1,p2] = ps
#        print 'Weights:', wei
#        print 'Time:', toc-tic
#        print 'PSNR:', "{0:.3f}".format(ps), '\n'