# -*- coding: utf-8 -*-
#Oscar Gerardo Lazo Arjona

#References ('cause in software it's cool to put them at the beginning)
#[1] Olivero, J.J.; R.L. Longbothum (February 1977). 
#   "Empirical fits to the Voigt line width: A brief review". 
#   Journal of Quantitative Spectroscopy and Radiative Transfer
#   17 (2): 233–236. Bibcode:1977JQSRT..17..233O. 
#   doi:10.1016/0022-4073(77)90161-3. ISSN 0022-4073. 
#   Retrieved 2009-04-01.

from lmfit.models import LorentzianModel, VoigtModel
from matplotlib import pyplot
from random import random
from numpy import array
from math import pi,sqrt,exp,erf,log
import numpy as np
from scipy.optimize import curve_fit
import subprocess

def model_lorentzian(n=1):
	s='lambda x,'
	for i in range(1,n+1): s+=' A'+str(i)+','
	for i in range(1,n+1): s+=' x0'+str(i)+','
	for i in range(1,n+1): s+=' gamma'+str(i)+','
	s=s[:-1]
	s+=' :'
	for i in range(1,n+1): s+=' A'+str(i)+'*gamma'+str(i)+'**2/(4*(x-x0'+str(i)+')**2+gamma'+str(i)+'**2) +'
	s=s[:-1]
	return eval(s)

def fit_lorentizans(x,y,p1,n=1,make_fit=True):
	sub_models=[]
	for i in range(n):
		prefix='l'+str(i+1)+'_'
		li = LorentzianModel(prefix=prefix)
		if i==0:
			parameters=li.make_params()
		else:
			parameters.update( li.make_params())

		A=p1[i]
		sigma=p1[2*n+i]
		center=p1[n+i]

		sigmap=sigma/2
		Ap=A*pi*sigmap

		parameters[prefix+'center'   ].set(center)
		parameters[prefix+'amplitude'].set(Ap)
		parameters[prefix+'sigma'    ].set(sigmap)

		sub_models+=[li]

	mod=sub_models[0]
	for i in range(1,n): mod+= sub_models[i]

	init=mod.eval(parameters,x=x)
	if not make_fit: return p1,[0 for i in range(3*n)],init

	out  = mod.fit(y, parameters, x=x)

	#print out.result.params.var_names

	var_names=out.result.var_names
	error_bars0={}
	for i in var_names:
		ii=var_names.index(i)
		error_bar=sqrt(out.result.covar[ii][ii])
		error_bars0.update({i:error_bar})

	result=out.values

	fit=[0 for i in range(3*n)]
	error_bars=[0 for i in range(3*n)]
	for i in range(n):
		Ap=result['l'+str(i+1)+'_amplitude']
		sigmap=result['l'+str(i+1)+'_sigma']
		center=result['l'+str(i+1)+'_center']

		Ap_bar=error_bars0['l'+str(i+1)+'_amplitude']
		center_bar=error_bars0['l'+str(i+1)+'_center']
		sigmap_bar=error_bars0['l'+str(i+1)+'_sigma']
		
		A=Ap/pi/sigmap
		sigma=2*sigmap

		covar=out.result.covar[n*i][n*i+2]
		A_bar= A*sqrt( (Ap_bar/Ap)**2 + (sigmap_bar/sigmap)**2 -2*covar/Ap/sigmap)/pi
		sigma_bar=2*sigmap_bar

		
		fit[i]=A
		fit[n+i]=center
		fit[2*n+i]=sigma
		
		error_bars[i]=A_bar
		error_bars[n+i]=center_bar
		error_bars[2*n+i]=sigma_bar

	fitted_curve=out.best_fit
	return fit,error_bars,fitted_curve

def convolve_with_gaussian(x,f,sigma):
    #We will calculate with data from a zero-centered normalized gaussian distribution
    #such that the steps between data are the same as the original signal.
    N=len(f); step=(x[1]-x[0])
    a=(x[-1]-x[0])/2.0

    x_gaussian=[-a +i*step for i in range(N)]
    gaussian=[0.398942280401433*exp(-xi**2/(2*sigma**2))/sigma for xi in x_gaussian]
    
    #We calculate N points of the convolution.
    fg=np.convolve(f,gaussian,mode='same')

    #We correct for border effects.
    stepcor=step*float(N)/float(N+1)
    xfg=[x[0] +i*stepcor for i in range(len(fg))]

    #We give the convolution it's correct units.
    fg=[fg[i]*step for i in range(len(fg))]
    return np.array(xfg),np.array(fg)

def fit_vogte(x,y,p1,n=1,equal_widths=False,make_fit=True):
	x=array(x); y=array(y)
	sub_models=[]
	
	for i in range(n):
		prefix='v'+str(i+1)+'_'
		vi = VoigtModel(prefix=prefix)
		if i==0:
			parameters=vi.make_params()
		else:
			parameters.update( vi.make_params())

		center = p1[0][i]
		A      = p1[1][i]
		gamma  = p1[2][i]
		if equal_widths: gamma=p1[2][0]
		
		#We correct in order to use our own convention
		gammap=gamma/2.0
		Ap=A*pi*gammap
		if len(p1)==4:
			#We use the suggested sigma if it is provided.
			sigma=p1[3][i]
			if equal_widths: sigma=p1[3][0]
			Ap=A /(-1/2.0*sqrt(2.0))
			Ap=Ap/((erf(1/2.0*sqrt(2.0)*gammap/sigma) - 1.0))
			Ap=Ap/(exp(1/2.0*gammap**2/sigma**2)/(sqrt(pi)*sigma))
		else: sigma=0.00001

		parameters[prefix+'center'   ].set(center)
		parameters[prefix+'amplitude'].set(Ap,min=0.0)
		if equal_widths and i !=0:
			parameters[prefix+'sigma'    ].set(sigma, expr='v1_sigma')
			parameters[prefix+'gamma'].set(value=gammap, vary=True, expr='v1_gamma')
		else:
			parameters[prefix+'sigma'    ].set(sigma)
			parameters[prefix+'gamma'].set(value=gammap, vary=True)

		sub_models+=[vi]

	
	mod=sub_models[0]
	for i in range(1,n): mod+= sub_models[i]

	#We decide wether to make the fit or return the initial guess.
	if not make_fit:
		init=mod.eval(parameters,x=x)
		return p1+[[0 for i in range(n)] for j in range(1)],[[0 for i in range(n)] for j in range(5)],init

	#We make the fit.
	def my_aborting_function(params, iteration, resid, *args, **kws):
		max_iter=2000
		if iteration>max_iter:
			raise ValueError,'The maximum number of iterations was reached: '+str(max_iter)

	out  = mod.fit(y, parameters, x=x,iter_cb=my_aborting_function)

	var_names=out.result.var_names
	error_bars0={}
	for i in var_names:
		ii=var_names.index(i)
		error_bar=sqrt(out.result.covar[ii][ii])
		error_bars0.update({i:error_bar})

	result=out.values

	#We gather the results to return them in the format
	#[[x1,x2,x3,...],[A1,A2,A3,...],[gamma1,gamma2,gamma3,...],
	# [sigma1,sigma2,sigma3,...],[fwhm1,fwhm2,fwhm3,...]]
	fit       =[[0 for i in range(n)] for i in range(5)]
	error_bars=[[0 for i in range(n)] for i in range(5)]
	for i in range(n):

		center=result['v'+str(i+1)+'_center']
		Ap    =result['v'+str(i+1)+'_amplitude']
		gammap=result['v'+str(i+1)+'_gamma']
		sigma =result['v'+str(i+1)+'_sigma']

		center_bar    =error_bars0['v'+str(i+1)+'_center']
		Ap_bar        =error_bars0['v'+str(i+1)+'_amplitude']
		if equal_widths:
			gammap_bar    =error_bars0['v1_gamma']
			sigma_bar     =error_bars0['v1_sigma']
		else:
			gammap_bar    =error_bars0['v'+str(i+1)+'_gamma']
			sigma_bar     =error_bars0['v'+str(i+1)+'_sigma']
		#We will calculate the error bar of the fwhm conservatively:
		fwhm_bar=2*gammap_bar+sigma_bar
		
		#We correct in order to use our own convention
		A = -1/2.0*sqrt(2.0)*Ap*(erf(1/2.0*sqrt(2.0)*gammap/sigma) - 1)
		A = A*exp(1/2.0*gammap**2/sigma**2)/(sqrt(pi)*sigma)
		gamma=2*gammap

		#We correct the convention for the error bars. The error bar
		#for the amplitude should be calculated with error-propagating formulae
		#but since it is not of great interest to us, this will suffice for now.
		A_bar = -1/2.0*sqrt(2.0)*Ap_bar*(erf(1/2.0*sqrt(2.0)*gammap/sigma) - 1)
		A_bar = A_bar*exp(1/2.0*gammap**2/sigma**2)/(sqrt(pi)*sigma)
		gamma_bar=2*gammap_bar
		
		#We calculate the FWHM according to [1]
		fg=2.0*sigma*sqrt(2.0*log(2.0))
		fl=gamma
		fwhm= 0.5346*abs(fl) +sqrt(0.2166*fl**2 + fg**2)

		#We save the results.
		fit[0][i]=center
		fit[1][i]=A
		fit[2][i]=gamma
		fit[3][i]=sigma
		fit[4][i]=fwhm
		
		error_bars[0][i]=center_bar
		error_bars[1][i]=A_bar
		error_bars[2][i]=gamma_bar
		error_bars[3][i]=sigma_bar
		error_bars[4][i]=fwhm_bar

	fitted_curve=out.best_fit
	
	return fit,error_bars,fitted_curve

def remove_offset(chan):
	off=(sum(chan[:10])+sum(chan[-10:]))/20
	#off=(chan[-1]+chan[0])/2.0
	return [ dat-off for dat in chan]

def rescale_data(data,rescale_to,rescale_from=None,rescale_together=False):
	"""This function recieves a list of data lists and returns them rescaled so
	that each of their maxima becomes 'rescale_to'. If 'rescale_from' is given
	then the data value 'rescale_from' becomes 'rescale_to'."""
	
	if rescale_together:
		rescale_from=max([max(yi) for yi in data])
	
	rescaled_data=[]
	for d in data:
		if rescale_from == None:
			m=max(d)
		else:
			m=rescale_from
		rescaled_data+=[ [i*rescale_to/m for i in d] ]
	return rescaled_data

def model_lorentzian(n=1):
	s='lambda x,'
	for i in range(1,n+1): s+=' A'+str(i)+','
	for i in range(1,n+1): s+=' x0'+str(i)+','
	for i in range(1,n+1): s+=' gamma'+str(i)+','
	s=s[:-1]
	s+=' :'
	for i in range(1,n+1): s+=' A'+str(i)+'*gamma'+str(i)+'**2/(4*(x-x0'+str(i)+')**2+gamma'+str(i)+'**2) +'
	s=s[:-1]
	return eval(s)

def model_gaussian(n=1):
	s='lambda x,'
	for i in range(1,n+1): s+=' A'+str(i)+','
	for i in range(1,n+1): s+=' x0'+str(i)+','
	for i in range(1,n+1): s+=' sigma'+str(i)+','
	s=s[:-1]
	s+=' :'
	for i in range(1,n+1): s+=' A'+str(i)+'*np.exp(-(x-x0'+str(i)+')**2/(2*sigma'+str(i)+'**2)) +'
	s=s[:-1]
	return eval(s)

def fit_lorentizans(x,y,p0=None,n=1,return_p0=False):
	'''Fits a lorentzian curve using p0=[A1,A2,...,x1,x2,...,gamma1,gamma2,...] as an initial guess.
	It returns a curve with N_points.'''
	
	lorentzians=model_lorentzian(n)
	N=len(x)
	if return_p0:
		fit=p0;pcov=[[0.0 for j in range(3*n)] for i in range(3*n)]
	else:
		fit,pcov = curve_fit(lorentzians,x,y,p0=p0)
	s=''.join([',fit['+str(i)+']' for i in range(3*n)])
	s='[lorentzians(xi'+s+') for xi in x]'
	fitted_curve=eval(s)
	
	for i in range(n):
		fit[-i-1]=abs(fit[-i-1])
		
	fit_ord=sorted([(fit[n+i],fit[i],fit[2*n+i]) for i in range(n)])
	fit=[fit_ord[i][1] for i in range(n)]+[fit_ord[i][0] for i in range(n)]+[fit_ord[i][2] for i in range(n)]

	return fit,pcov,fitted_curve

def fit_gaussians(x,y,p0=None,n=1,return_p0=False):
	'''Fits a gaussian curve using p0=[A1,A2,...,x1,x2,...,sigma1,sigma2,...] as an initial guess.
	It returns a curve with N_points.'''
	
	gaussians=model_gaussian(n)
	N=len(x)
	if return_p0:
		fit=p0;pcov=None
	else:
		try:
			fit,pcov = curve_fit(gaussians,x,y,p0=p0)
		except:
			print 'WARNING: it was impossible to fit the model.'
			fit=[0]*(2*n)+[1]*n; pcov=None
	s=''.join([',fit['+str(i)+']' for i in range(3*n)])
	s='[gaussians(xi'+s+') for xi in x]'
	fitted_curve=eval(s)
	return fit,pcov,fitted_curve

def three_peak_indices(pmt):
	peakind = signal.find_peaks_cwt(pmt, np.arange(1,50))
	lp=[(pmt[i],i) for i in peakind]
	lp=list(reversed(sorted(lp)))[:3]
	lp=sorted([i[1] for i in lp])
	return lp

def read_steps(name,column=2):
	f=file(name,'r')
	l=f.readlines()
	f.close()

	steps=[]
	for li in l[2:]:
		d=[di for di in li[:-1].split(',') if di!='']
		steps+=[float(d[column-1])]
	return steps

def get_file_names(by_pairs=True):
	p = subprocess.Popen(['ls'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
	out, err = p.communicate()
	data_files=[i for i in out.split('\n') if i[-4:]=='.dat']
	data_files=[i for i in data_files if i not in ['calib87.dat']]
	if by_pairs:
		N=len(data_files)
		if N%2!=0:
			raise ValueError,'the number of data files is odd!'
		pairs=[(data_files[2*i],data_files[2*i+1]) for i in range(N/2)]
		return pairs
	else:
		return data_files

def read_chans(file_name):
	f=file(file_name,'r')
	d=f.readlines()
	f.close()
	d=[l.split() for l in d]
	d=[[float(i) for i in l] for l in d]
	chans=[]
	Nchans=len(d[0])
	Ndat=len(d)
	for i in range(Nchans):
		chan=[d[j][i] for j in range(Ndat)]
		chans+=[chan]
		
	return [[ chans[i][j] for j in range(Ndat/4,3*Ndat/4)] for i in range(Nchans)]

def calibrate(v,v_base,m):
	return [m*(vi-v_base) for vi in v]

#~ a=-10.0; b=10.0; N=1000
#~ step=(b-a)/(N-1)
#~ x=array([a+i*step for i in range(N)])
#~ 
#~ n=3
#~ 
#~ mod0=model_lorentzian(n)
#~ noise=0.1
#~ p0=[10,2,1, -5,0,5, 0.25,0.25,0.25]
#~ y0=array([mod0(xi,*p0) for xi in x])
#~ #y =array([mod0(xi,*p0) + noise*(random()-0.5) for xi in x])
#~ 
#~ x,y0=convolve_with_gaussian(x,y0,0.75)
#~ y =array([y0[i] + noise*(random()-0.5) for i in range(len(y0))])
#~ 
#~ p1=[[-6, 0.2, 5.1], [1.5,0.2,0.1],  [0.125,0.125,0.125]]
#~ 
#~ fit,error_bars,init=fit_vogte(x,y,p1,n,make_fit=False)
#~ fit,error_bars,fitted_curve=fit_vogte(x,y,p1,n)
#~ 
#~ 
#~ pyplot.plot(x,init,'k-')
#~ pyplot.plot(x,y,'b+')
#~ pyplot.plot(x,y0,'c-',linewidth=4.0)
#~ pyplot.plot(x,fitted_curve,'r-',linewidth=1.0)
#~ pyplot.savefig('voigt_fit.png',bbox_inches='tight')
