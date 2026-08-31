"""
Module to detect edges of 2D images 

Created: Feb 19, 2019
Author: Cristina MT
"""


import numpy as np

class EdgeDet():
	
	def canny_wt(WT_mod, WT_arg):
		"""
		Function to implement the edge detection based on a modified
		canny detector with the WT.

		Note that no threshold is used with this version.

		INPUT:
			WT_mod : numpy array, WT modulus, same size as image
			WT_arg : numpy array, WT argument in degrees
		OUTPUT
			mask_edge : numpy array, binary mask with the detected edges
						same size as image 
		"""

		# Discretise the argument to follow the directions
		# where we can look for the gradient
		rWT_arg = 45.*np.round(WT_arg/45.)

		# Define matrices with each direction to look for the 
		# maxima with matrix operations instead of loops
		dir_sa=[0.,45.,90.,135.,-180.,-135.,-90.,-45.]
		# Initialise direction matrix
		WT_mod_0=np.zeros(WT_mod.shape)
		WT_mod_45=np.zeros(WT_mod.shape)
		WT_mod_90=np.zeros(WT_mod.shape)
		WT_mod_135=np.zeros(WT_mod.shape)
		WT_mod_m180=np.zeros(WT_mod.shape)
		WT_mod_m135=np.zeros(WT_mod.shape)
		WT_mod_m90=np.zeros(WT_mod.shape)
		WT_mod_m45=np.zeros(WT_mod.shape)
		# Get the values from the WT modulus
		WT_mod_0[:,:-1]=WT_mod[:,1:]
		WT_mod_m180[:,1:]=WT_mod[:,: -1]
		WT_mod_90[:-1,:]=WT_mod[1:,:]
		WT_mod_m90[1:,:]=WT_mod[:-1,:]
		WT_mod_45[:-1,:-1]=WT_mod[1:,1:]
		WT_mod_m135[1:,1:]=WT_mod[:-1,:-1]
		WT_mod_135[:-1,1:]=WT_mod[1:,:-1]
		WT_mod_m45[1:,:-1]=WT_mod[:-1,1:]

		# Construct the matrices to compare with the pixel i+1
		# and with the pixel i-1
		WT_mod_ap=[WT_mod_0,WT_mod_45,
					WT_mod_90,WT_mod_135,
					WT_mod_m180,WT_mod_m135,
					WT_mod_m90,WT_mod_m45]
		WT_mod_am=[WT_mod_m180,WT_mod_m135,
					WT_mod_m90,WT_mod_m45,
					WT_mod_0,WT_mod_45,
					WT_mod_90,WT_mod_135]

		# Initialize matrix for the edge mask
		mask_edge=np.zeros(WT_mod.shape)
		# Loop through each direction and compare the corresponding matrices
		# keeping the maxima
		for idir in range(len(dir_sa)):
			mask_angle=rWT_arg==dir_sa[idir]
			mask_edge[mask_angle*np.greater(WT_mod,WT_mod_ap[idir])*np.greater(WT_mod,WT_mod_am[idir])]=1
			# Normalize the modulus
			nWT_mod=WT_mod/np.max(WT_mod.flatten())

		return mask_edge