"""
Module to implement 2D Wavelet on images

Created: Feb 19, 2019
Author: Cristina MT
"""

import numpy as np

class Wavelet2D_gauss():
	"""
	Class with some of the most used wavelets from the gaussian family.
	All the functions are symetrical.

	Note that the functions are all defined for a particular scale,
	and the Wavelet Transform hasn't been normalised (The norm could
	vary depending on the application)

	They are constructed to be used with the BuildFFT function 
	"""

	def firstder(a_scale, xg, yg, fft_img ):
		"""
		Function to compute the 2D Wavelet Transform using the first
		derivative of a gaussian as a mother wavelet.

		INPUT:
			a_scale: float, scale or dilation for the gaussian envelop, 
					it works better when it's a multiple of 2
			xg: 	numpy array, grid of x values. Output of BuildFFT.img2D
			yg: 	numpy array, grid of y values. Output of BuildFFT.img2D
			fft_img: numpy array, 2D FFT of the image
		OUTPUT:
			WT_mod : numpy array, modulus of the WT, same size as the input image
			WT_arg : numpy array, argument of the WT in degrees
		"""

		# Define the gaussian function in real space. Note that x*x is better than x**2
		phi = np.exp(-((xg/a_scale)*(xg/a_scale)+(yg/a_scale)*(yg/a_scale))/2)
		# Define the first derivative in x and in y
		phi_x = -(xg/a_scale)*phi
		phi_y = -(yg/a_scale)*phi
		# Normalise according to the scale
		phi_xg = 1/(a_scale**2)*phi_x
		phi_yg  = 1/(a_scale**2)*phi_y
		## Compute the 2D FFT, shifted
		fft_phi_x=np.fft.fftshift(np.fft.fft2(phi_xg))
		fft_phi_y=np.fft.fftshift(np.fft.fft2(phi_yg))
		# Compute the gradient in x and in y, and inverse the FFT
		WT_x=np.fft.ifftshift(np.fft.ifft2(fft_phi_x*fft_img))
		WT_y=np.fft.ifftshift(np.fft.ifft2(fft_phi_y*fft_img))
		# Compute the modulus (not normalized) and the argument in degrees
		WT_mod=np.sqrt(np.abs(WT_x)**2+np.abs(WT_y)**2)
		WT_arg = np.angle(WT_x + 1j*WT_y, deg = True)

		return WT_mod, WT_arg
    
	def secder(a_scale, xg, yg, fft_img):
		"""
		Function to compute the 2D Wavelet Transform using the second
		derivative of a gaussian as a mother wavelet.

		INPUT:
			a_scale: float, scale or dilation for the gaussian envelop, 
					it works better when it's a multiple of 2
			xg: 	numpy array, grid of x values. Output of BuildFFT.img2D
			yg: 	numpy array, grid of y values. Output of BuildFFT.img2D
			fft_img: numpy array, 2D FFT of the image
		OUTPUT:
			WT_mod : numpy array, modulus of the WT, same size as the input image
			WT_arg : numpy array, argument of the WT in degrees
		"""

		# Define the gaussian kernel
		phi = np.exp(-((xg/a_scale)*(xg/a_scale)+(yg/a_scale)*(yg/a_scale))/2)
		# Compute the first derivative in x and in y
		phi_x = (xg/a_scale)*(xg/a_scale)*phi
		phi_y = (yg/a_scale)*(yg/a_scale)*phi
		# Compute the second derivative in x and in y, normalised
		phi_xg = 1/(a_scale**2)*phi_x
		phi_yg  = 1/(a_scale**2)*phi_y
		# Compute the 2D FFT, shifted
		fft_phi_x=np.fft.fftshift(np.fft.fft2(phi_xg))
		fft_phi_y=np.fft.fftshift(np.fft.fft2(phi_yg))
		# Compute the WT in x and in y
		WT_x=np.fft.ifftshift(np.fft.ifft2(fft_phi_x*fft_img))
		WT_y=np.fft.ifftshift(np.fft.ifft2(fft_phi_y*fft_img))
		# Compute the modulus and the argument in degrees
		WT_mod=np.sqrt(np.abs(WT_x)**2+np.abs(WT_y)**2)
		WT_arg = np.angle(WT_x + 1j*WT_y, deg = True)
        
		return WT_mod, WT_arg
		
	def gauss(a_scale, xg, yg, fft_img):
		"""
		Function to compute the 2D Wavelet Transform using a gaussian as a mother wavelet.

		INPUT:
			a_scale: float, scale or dilation for the gaussian envelop, 
					it works better when it's a multiple of 2
			xg: 	numpy array, grid of x values. Output of BuildFFT.img2D
			yg: 	numpy array, grid of y values. Output of BuildFFT.img2D
			fft_img: numpy array, 2D FFT of the image
		OUTPUT:
			WT_mod : numpy array, modulus of the WT, same size as the input image
		"""

		# Construct gaussian kernel
		phi = np.exp(-((xg/a_scale)*(xg/a_scale)+(yg/a_scale)*(yg/a_scale))/2)
		# Normalise 
		phi_xg = phi/np.sum(phi.flatten())
		# Compute the 2D FFT of the kernel
		fft_phi_x=np.fft.fftshift(np.fft.fft2(phi_xg))
		# Compute the Wavelet Transform
		WT_x=np.fft.ifftshift(np.fft.ifft2(fft_phi_x*fft_img))
		# Compute the WT modulus
		WT_mod = np.abs(WT_x)

		return WT_mod