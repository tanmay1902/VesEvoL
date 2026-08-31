"""
Module to build FFT -related arrays of 2D images 
to be used in multiscale analysis or multi-operations
and reduce computation time

Created: Feb 19, 2019
Author: Cristina MT
"""

import numpy as np

class BuildFFT():
	"""
	Builds the 2D FFT of an image and the base grid matrix
	to be used in convolution computations.

	It is particularly useful when multiple convolutions
	are required on the same image, as it saves time
	in the computation by taking the FFT from 'memory'
	instead of computing it each time a convolution is desired.
	"""

	def img2D(mat_img):
		"""
		Function to compute the 2D FFT of an image, 
		together with the grid matrix for position

		INPUT: 
		mat_img : numpy array, 2D
		OUTPUT:
		xg, yg : numpy arrays with a 'grating' structure for x and y axis
						they are used to construct the filter function with the same
						dimensions and coordinates as the images
		fft_img : numpy array with the fourier transform in 2D of the image, 
						with shifted quadrants
		
		"""
		# Construct x and y vectors, with zero at the middle
		x = np.arange(-mat_img.shape[1]/2, mat_img.shape[1]/2)
		y = np.arange(-mat_img.shape[0]/2, mat_img.shape[0]/2)
		# Construct the matrix as a stack of vectors
		xg = np.tile(x, (len(y),1))
		yg = np.tile(y, (len(x), 1)).transpose()

		# Compute the 2D FFT of the image, shifted
		fft_img = np.fft.fftshift(np.fft.fft2(mat_img))

		return xg, yg, fft_img