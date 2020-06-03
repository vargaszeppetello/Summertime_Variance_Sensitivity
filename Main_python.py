#/usr/bin/python

from netCDF4 import Dataset
import numpy as np
#from scipy.stats import pearsonr
import matplotlib.pyplot as plt
#import ensemble_play

def write_netcdf(SET_NAME,var_name,LAT,LON,ARRAY):

	new_set = Dataset(SET_NAME,'w',format='NETCDF3_64BIT')
	new_set.createDimension('lat',len(LAT))
	new_set.createDimension('lon',len(LON))	
	new_set.createDimension('n',len(ARRAY[:,0,0]))
	
	MAIN_VAR = new_set.createVariable(var_name,'f4',('n','lat','lon',))
	LAT_VAR  = new_set.createVariable('lat','f4',('lat',))
	LON_VAR  = new_set.createVariable('lon','f4',('lon',))

	MAIN_VAR[:,:,:] = ARRAY
	LAT_VAR[:] = LAT
	LON_VAR[:] = LON
	new_set.close()

def write_list(SET_NAME,X):
	new_set = Dataset(SET_NAME,'w',format='NETCDF3_64BIT')
	new_set.createDimension('n',len(X))

	Xvar = new_set.createVariable('dT','f4',('n', ))
#	Yvar = new_set.createVariable('dTvar','f4',('n', ))

	Xvar[:] = X
#	Yvar[:] = Y

	new_set.close()


def hor_interp(array,X_initial,Y_initial,X_final,Y_final):
	from scipy.interpolate import griddata
	x,y = np.meshgrid(X_initial,Y_initial)
	x_final,y_final = np.meshgrid(X_final,Y_final)
	final_array = griddata((x.ravel(),y.ravel()),array.ravel(),(x_final,y_final))
	return(final_array)

def q_s(T_array):

	e_s = 6.11*10**(7.5*T_array/(237.5+T_array))
	return(e_s*0.61/1000.)

def dq_dT(T_array):
	
	q_i = q_s(T_array-0.05)
	q_f = q_s(T_array+0.05)
	return((q_f - q_i)/0.1)

def New_toy(R,P,C,SM,T,Q,qs,gamma,nu,r_s,mu_fact):

	mean_temp = T

	i = 0
	while i < len(SM[:,0]):
		j = 0
		while j < len(SM[0,:]):
			if SM[i,j] > 1:
				SM[i,j] =1
			j = j + 1
		i = i + 1

	nlats = len(SM[:,0])
	nlons = len(SM[0,:])

	L = 2.25e6
	rho_a = 1.2

	VPD = qs - Q

	precip_std = (P/(L**2))**0.5
	mu = precip_std*mu_fact

	beta = (rho_a*VPD/r_s)
	Full_damp = nu + (L*rho_a*SM*gamma/r_s)*(1 - (1/(mu/beta + 1)))

	Trad = (1/Full_damp**2)*R
	Tcovar = -(2/Full_damp**2)*C/((mu/beta) + 1)
	Tprecip = (1/Full_damp**2)*P/((mu/beta)+1)**2

	return(Tprecip + Tcovar + Trad)

def dsigma_dT(R,P,C,SM,T,Q,qs,gamma,double_gamma,Tvar,nu,r_s,mu_fact,RH_sensitivity):

	mean_temp = T
	i = 0
	while i < len(SM[:,0]):
		j = 0
		while j < len(SM[0,:]):
			if SM[i,j] > 1:
				SM[i,j] =1
			j = j + 1
		i = i + 1

	nlats = len(SM[:,0])
	nlons = len(SM[0,:])

	L = 2.25e6
	rho_a = 1.2
	

	VPD = qs - Q
	RH  = Q/qs

	precip_std = (P/(L**2))**0.5
	mu = precip_std*mu_fact

	beta = (rho_a*VPD/r_s)
	zeta = (1/(1 + (mu/beta)))

	i = 0
	while i < nlats:
		j = 0
		while j < nlons:
			if zeta[i,j] < 0:
				zeta[i,j] = 0
			j = j +1
		i = i + 1

	Full_damp = nu + (L*rho_a*SM*gamma/r_s)*(1 - (1/(mu/beta + 1)))


	alpha = mu*r_s/rho_a

	dz_dT = alpha/((VPD + alpha)**2)*(gamma*(1-RH) - qs*RH_sensitivity)
	dG_dT = (L*rho_a*SM/r_s)*(double_gamma*(1-zeta) - gamma*dz_dT)

#	dsigma_dT = (2/(Full_damp**2))*(zeta*dz_dT*P - C*dz_dT - Full_damp*dG_dT*Tvar)

	dsigma_dT_precip = (2/(Full_damp**2))*(zeta*dz_dT*P)
	dsigma_dT_covar  = -(2/(Full_damp**2))*(C*dz_dT)
	dsigma_dT_Tvar   = -(2/(Full_damp**2))*(Full_damp*dG_dT*Tvar)


	return(dsigma_dT_precip)

def skew(skew_F,skew_P,cross1,cross2,P,SM,Q,qs,gamma,nu,r_s,mu_fact):

	nlats = len(SM[:,0])
	nlons = len(SM[0,:])

	L = 2.25e6
	rho_a = 1.2

	VPD = qs - Q

	precip_std = (P/(L**2))**0.5
	mu = precip_std*mu_fact

	beta = (rho_a*VPD/r_s)
	zeta = (1/(1 + (mu/beta)))

	i = 0
	while i < nlats:
		j = 0
		while j < nlons:
			if zeta[i,j] < 0:
				zeta[i,j] = 0
			j = j +1
		i = i + 1

	Full_damp = nu + (L*rho_a*SM*gamma/r_s)*(1 - (1/(mu/beta + 1)))

	fullcross1 = 3*zeta*cross1
	fullcross2 = 3*(zeta**2)*cross2

	skew1 = (1/(Full_damp**3))*(skew_F)
	skew2 = (1/(Full_damp**3))*(-fullcross1)
	skew3 = (1/(Full_damp**3))*(fullcross2)
	skew4 = (1/(Full_damp**3))*(-skew_P*(zeta**3))

	return(skew1+skew2+skew3+skew4)

def New_toy_LHF(R,P,C,mean_sm,mean_temp,Q,nu,r_s,mu):

	L = 2.25e6
	rho_a = 1.2

	gamma = dq_dT(mean_temp)
	qs = q_s(mean_temp)

	VPD = qs - Q

	precip_std = (P/(L**2))**0.5
	mu = precip_std*mu	

	beta = (rho_a*VPD/r_s)

	Gamma = nu + (L*rho_a*mean_sm*gamma/r_s)*(1 - (beta/(mu+beta)))
#	FACT = L*rho_a*mean_sm*gamma/(r_s*Gamma) 
	zeta = (1/(1 + (mu/beta)))

#	pcoef = 1 - (mu/(mu + beta))*(1 + (beta*FACT)/(mu + beta))
#	rcoef = (FACT*mu)/(mu + beta)

	pcoef = zeta
	rcoef = (Gamma/nu) - 1
	fact = (nu/Gamma)

	LEprecip = (fact**2)*P*(pcoef**2)
	LEcovar  = (fact**2)*2*C*pcoef*rcoef
	LErad	 = (fact**2)*R*(rcoef**2)

	return(LErad,LEcovar,LEprecip)

#def plotting(X,Y,pred,pred_noRH,gcme):
def plotting(X,Y,Z):

	import matplotlib.pyplot as plt
	import matplotlib.colors as colors
	import matplotlib as mpl

###############################################################################################
	class MidpointNormalize(colors.Normalize):
		def __init__(self, vmin=None, vmax=None,midpoint=None,clip=False):
			self.midpoint = midpoint
			colors.Normalize.__init__(self,vmin,vmax,clip)

		def __call__(self,value,clip=None):
			x, y = [self.vmin,self.midpoint,self.vmax], [0,0.5,1]
			return np.ma.masked_array(np.interp(value,x,y)) 

##############################################################################################

	lsm_set = Dataset('era5_lsm.nc')
	lsm = lsm_set['lsm'][0,:,:]
	lsm_lat = lsm_set['latitude'][:]
	lsm_lon = lsm_set['longitude'][:]
	interp_lsm = hor_interp(lsm,lsm_lon,lsm_lat,X,Y)

#	weird_lon = np.ones(len(X))
#
#	i = 0
#	while i < len(X):
#		if X[i] > 55:
#			weird_lon[i] = 0
#		if X[i] > 349:
#			weird_lon[i] = 1
#
#		i = i + 1
#
############################## GREAT LAKES ##################################################3

#	Z = np.nanmean(Array,axis=0)
#	from scipy.stats import pearsonr
#	O = []
#	G = []
	agree = np.zeros(shape=(len(Y),len(X)))
#
#	i = 0
#	while i < len(Y):
#		j = 0
#		while j < len(X):
#			if sum(np.sign(Z_full[:,i,j])) > 12 or sum(np.sign(Z_full[:,i,j])) < -12:
#				agree[i,j] = 1
#			j = j +1
#		i = i + 1
#
#	Z = np.nanmean(Z_full,axis=0)

	i = 0
	while i < len(Y):
		j = 0
		while j < len(X):
	
#			O.append(pred[i,j])
#			G.append(gcme[i,j])

			if interp_lsm[i,j] < 0.01:  # FOR OBS GRID
				agree[i,j] = np.nan
#				gcme[i,j] = np.nan
#				pred_noRH[i,j] = np.nan
				Z[i,j] = np.nan
###### GREENLAND

			if Y[i] < 75 and Y[i] > 60 and X[j] > 300 and X[j] < 335:
				agree[i,j] = np.nan	
#				gcme[i,j] = np.nan
#				pred_noRH[i,j] = np.nan
				Z[i,j] = np.nan

			if Y[i] < 75 and Y[i] > 70 and X[j] > 300 and X[j] < 350:
				agree[i,j] = np.nan	
#				pred[i,j] = np.nan
#				pred_noRH[i,j] = np.nan
				Z[i,j] = np.nan
##################
			if Y[i] < 25 and Y[i] > 75:
#				O.append(Z[i,j])
#				pred[i,j] = np.nan
#				gcme[i,j] = np.nan
#				pred_noRH[i,j] = np.nan
				Z[i,j] = np.nan
	
#			if weird_lon[j] == 0:
#			if X[j] < 194 or X[j] >304:
#				
#				pred[i,j] = np.nan
#				gcme[i,j] = np.nan
#				pred_noRH[i,j] = np.nan
###
#			if pred[i,j] < 0 or pred[i,j] > 3:
#				pred[i,j] = np.nan
#				gcme[i,j] = np.nan
#				pred_noRH[i,j] = np.nan

			j = j + 1
		i = i + 1

#	return(Z)
#	return(pred,pred_noRH,gcme)
	vmin = -1
	vmax = 1
	midpoint = 0
	bounds = np.linspace(vmin,vmax,11)
#	bounds = [0.25,0.5,0.75,.85,0.95,1.05,1.1,1.25,1.5,2,3,4]
#	bounds = [-2,-1,-.5,-.25,.25,.5,1,1.5,2]
#	bounds = [-1,-0.5,-0.25,-.1,.1,.25,.5,.75,1]
#	bounds = [-.5,-.4,-.3,-.2,-.1,.1,.2,.3,.4,0.5]
	cmap = 'RdBu_r'
	
	Zfin = plot_prep(Z)
	newstiples = plot_prep(agree)
	LSM = plot_prep(lsm)

	fig = plt.figure(figsize=(16,6))
	ax1 = fig.add_axes([0.1,0.2,0.8,0.65])

	ax1.contourf(X,Y,Zfin,norm=MidpointNormalize(vmin=vmin,vmax=vmax,midpoint=midpoint),cmap=cmap,levels=bounds,extend='both')
#	ax1.contourf(X,Y,newstiples,[0.75,1],colors='none',hatches=['..'])
	ax1.contour(lsm_lon,lsm_lat,LSM,[0.5],colors='k')
	ax1.set_ylim(25,75)
#	ax1.set_xlim(195,304)

	ax1_cbar = fig.add_axes([0.2,0.12,0.6,.02])
	cb1 = mpl.colorbar.ColorbarBase(ax1_cbar,norm=MidpointNormalize(vmin=vmin,vmax=vmax,midpoint=midpoint),spacing='uniform',
		cmap=cmap,boundaries=bounds,ticks=bounds,orientation='horizontal')
	
#	plt.savefig('covar_rat.pdf')
	plt.show()

#	print(max(O),np.percentile(O,95),np.percentile(O,90))

def scatter(pred,pred_noRH,gcme):

	from scipy.stats import pearsonr
	O = pred.flatten()
	G = gcme.flatten()
	noRH = pred_noRH.flatten()

	newO = np.zeros(1)
	newG = np.zeros(1)
	newN = np.zeros(1)
	i = 0

	while i < len(O):
		if str(O[i]) != 'nan':
			newO = np.concatenate((newO,np.ones(1)*O[i]),axis=0)
			newG = np.concatenate((newG,np.ones(1)*G[i]),axis=0)
			newN = np.concatenate((newN,np.ones(1)*noRH[i]),axis=0)
		i = i + 1

	newO = newO[1:]
	newG = newG[1:]
	newN = newN[1:]

	r_one = pearsonr(newG,newO)
	r_two = pearsonr(newG,newN)
	print(np.nanmean(newO),np.nanmean(newG),np.nanmean(newN))

	print(r_one[0],np.nanstd(newG),np.nanstd(newO))
	print(r_two[0],np.nanstd(newG),np.nanstd(newN))

	slope_wRH = r_one[0]*np.nanstd(newO)/np.nanstd(newG)
	slope_woRH = r_two[0]*np.nanstd(newN)/np.nanstd(newG)
	plt.scatter(newG,newN,s=5)
	plt.scatter(newG,newO,s=5)
	plt.plot(np.linspace(-5,5,11),np.linspace(-5,5,11),'-',color='k')
	plt.plot(np.linspace(-5,5,11),np.linspace(-5,5,11)*slope_woRH,'--')
	plt.plot(np.linspace(-5,5,11),np.linspace(-5,5,11)*slope_wRH,'--')
	plt.xlim(-0.25,3)
	plt.ylim(-.25,3)
#	plt.savefig('Scatter_agreement_Eurasia.png')
	plt.show()

def Model_scatter(X,Y,TEMP,HUM):

	nmodels = len(TEMP[:,0,0])
	nlats = len(TEMP[0,:,0])
	nlons = len(TEMP[0,0,:])

	lsm_set = Dataset('era5_lsm.nc')
	lsm = lsm_set['lsm'][0,:,:]
	lsm_lat = lsm_set['latitude'][:]
	lsm_lon = lsm_set['longitude'][:]
	interp_lsm = hor_interp(lsm,lsm_lon,lsm_lat,X,Y)

	Sx = np.zeros(nmodels)
	Sy = np.zeros(nmodels)

	weird_lon = np.ones(len(X))

	i = 0
	while i < len(X):
		if X[i] > 55:
			weird_lon[i] = 0
		if X[i] > 349:
			weird_lon[i] = 1

		i = i + 1

	i = 0
	while i < nmodels:	
		j = 0
		while j < nlats:
			k = 0
			while k < nlons:
				if interp_lsm[j,k] < 0.1:
					TEMP[i,j,k] = np.nan
					HUM[i,j,k] = np.nan
				if Y[j] < 35 or Y[j] > 65:
					TEMP[i,j,k] = np.nan
					HUM[i,j,k] = np.nan
				if X[k] < 194 or X[k] > 304:
		#		if weird_lon[k] == 0:
					TEMP[i,j,k] = np.nan
					HUM[i,j,k] = np.nan

				k = k + 1
			j = j + 1
		Sx[i] = np.nanmean(TEMP[i,:,:])
		Sy[i] = np.nanmean(HUM[i,:,:])
		i = i + 1

	
	Labels = ['ACCESS-CM2','ACCESS-ESM1-5','CanESM5','CESM2','CESM2-WACCM',
		'CNRM-CM6-1','CNRM-ESM2-1','CNRM-CM6-1-HR','EC-Earth3','EC-Earth3-Veg',
		'GFDL-ESM4','GISS-E2-1-G','HadGEM3-GC31-LL','INM-CM4-8','INM-CM5-0','IPSL-CM6A-LR',
		'MIROC6','MIROC-ES2L','MPI-ESM1-2-HR','MPI-ESM1-2-LR','MRI-ESM2-0','NorESM2-MM','NorESM2-LM','UKESM1-0-LL']

	c = ['darkorange','red','fuchsia','forestgreen','darkviolet','black','brown','deepskyblue','sandybrown','chartreuse','teal',
		'khaki','olive','maroon','tomato','limegreen','powderblue','blue','yellow','darkviolet','dodgerblue','royalblue','darkgoldenrod','mediumvioletred']


	size_carbonclimate = [10,-21.1,15.95,-21.6,10,10,-83.11,10,10,10,-80.06,10,10,10,10,-8.67,10,-69.57,10,-5.17,10,10,-20.95,-38.4]
	size_carbonconcentration = [10,.37,1.28,.9,10,10,1.36,10,10,10,.93,10,10,10,10,.62,10,1.12,10,.71,10,10,.85,.75]
#	shape = ['s','o','o','o','s','s','o','s','s','s','o','s','s','s','s','o','s','o','s','o','s','s','o','o']

#	fig, ax = plt.subplots()
#	scatter = ax.scatter(Sx,Sy,c=c)
#	legend1 = ax.legend(c,labels=Labels,ncol=6,loc='lower left',title="Models")
#	plt.show()

	X = []
	Y = []

	i = 0
	while i < nmodels:
		plt.scatter(Sx[i],Sy[i],s=100,color=c[i],label=Labels[i],edgecolors='k')
		if size_carbonconcentration[i] != 10:
			X.append(Sy[i])
			Y.append(size_carbonclimate[i])

#			plt.scatter(Sy[i],size_carbonclimate[i],s=size_carbonclimate[i],marker=shape[i],color=c[i],label=Labels[i])

		i = i + 1
	from scipy.stats import pearsonr
	
	rstuff = pearsonr(X,Y)
	print(rstuff[0])
	rstuff = pearsonr(Sx,Sy)
	print(rstuff[0])
#	f = breakhere
#	plt.xlim(2,100)

#	plt.legend(loc='upper right',ncol=3,fontsize='small')
	plt.savefig('US_model_scat.pdf')
	plt.show()

	

def plot_prep(Array):

	split = int(len(Array[0,:])/2)
	the_east = Array[:,:split]
	the_west = Array[:,split:]

	new_array = np.concatenate((the_west,the_east),axis=1)

	return(new_array)

def CERES_Summer():

	Rad_set =Dataset('CERES_EBAF-Surface_Ed4.0_Subset_200003-201712.nc')
	Rad_data = Rad_set['sfc_net_sw_all_mon'][:,:,:]

	Rad_lat = Rad_set['lat'][:]
	Rad_lon = Rad_set['lon'][:]

	Rad_Junes = np.linspace(3,17*12 + 3,18)
	Rad_Julys = np.linspace(4,17*12 + 4,18)
	Rad_Augs  = np.linspace(5,17*12 + 5,18)

	Rad_dates = np.concatenate((Rad_Junes,Rad_Julys,Rad_Augs),axis=0)
	Rad_dates = np.sort(Rad_dates)
	Rad_dates = Rad_dates.astype(int)

	Summer_rad = Rad_data[Rad_dates,:,:]

	return(Summer_rad,Rad_lat,Rad_lon)

def OBS_anoms(array,nyears):

	nlats = len(array[0,:,0])
	nlons = len(array[0,0,:])

	anoms = np.zeros(shape=(nyears*3,nlats,nlons))
#	Junes = np.linspace(5,(nyears-1)*12 + 5,nyears)
#	Julys = np.linspace(6,(nyears-1)*12 + 6,nyears)
#	Augs  = np.linspace(7,(nyears-1)*12 + 7,nyears)
#
#	Junes = Junes.astype(int)
#	Julys = Julys.astype(int)
#	Augs  = Augs.astype(int)

	dum_Junes = np.linspace(0,(nyears*3) - 3,nyears)
	dum_Julys = np.linspace(1,(nyears*3) - 2,nyears)
	dum_Augs  = np.linspace(2,(nyears*3) - 1,nyears)

	dum_Junes = dum_Junes.astype(int)
	dum_Julys = dum_Julys.astype(int)
	dum_Augs  = dum_Augs.astype(int)

	anoms[dum_Junes,:,:] 	= array[dum_Junes,:,:] - np.mean(array[dum_Junes,:,:],axis=0)
	anoms[dum_Julys,:,:] 	= array[dum_Julys,:,:] - np.mean(array[dum_Julys,:,:],axis=0)
	anoms[dum_Augs,:,:] 	= array[dum_Augs,:,:] - np.mean(array[dum_Augs,:,:],axis=0)

#	Mean = np.nanmean(array[Junes,:,:] + array[Julys,:,:] + array[Augs,:,:],axis=0)/3.

	return(anoms)

def GCME_skew():

	L = 2257000.
	
	skew_Fset = Dataset('HISTORICAL_spread_summer_third_moment_rsds.nc','r')
	skew_Pset  = Dataset('HISTORICAL_spread_summer_third_moment_pr.nc','r')

	skew_cross1 = Dataset('HISTORICAL_spread_summer_cross1.nc','r')	
	skew_cross2 = Dataset('HISTORICAL_spread_summer_cross2.nc','r')

	lon = skew_Fset['lon'][:]
	lat = skew_Fset['lat'][:]

	skew_F = np.nanmean(skew_Fset['rsds'][:,:,:],axis=0)
	skew_P  = np.nanmean(skew_Pset['pr'][:,:,:],axis=0)*(L**3)

	cross1 = np.nanmean(skew_cross1['cross1'][:,:,:],axis=0)*L
	cross2 = np.nanmean(skew_cross2['cross2'][:,:,:],axis=0)*(L**2)

	pr_var = Dataset('HISTORICAL_spread_summer_variance_pr.nc','r')
	PVAR = np.nanmean(pr_var['pr'][:,:,:],axis=0)
	
#	Max fact = 26.565, 95% percentile = 11.3333 90% = 9.28 
	Fact = 26.565
	mean_pr = Dataset('HISTORICAL_spread_summer_mean_pr.nc','r')
	MEAN_PR = np.nanmean(mean_pr['pr'][:,:,:],axis=0)
	
	print(np.percentile(MEAN_PR,90))
	print(np.percentile(MEAN_PR,95))
	print(max(MEAN_PR))
	f = breakhere

	SM = MEAN_PR*(60*60*24*30)/(10.*Fact2)  # in cm/month normalized by 95 percentile

	nlat = len(lat)
	nlon = len(lon)

	i = 0
	while i < nlat:
		j = 0
		while j < nlon:
			if SM[i,j] >1:
				SM[i,j] = 1
			j = j + 1
		i = i + 1

	Tmean_set = Dataset('HISTORICAL_spread_summer_mean_tas.nc','r')
	Tmean = np.nanmean(Tmean_set['tas'][:,:,:],axis=0) - 273.15

	RH_set = Dataset('HISTORICAL_spread_summer_mean_hurs.nc','r')
	RH = np.nanmean(RH_set['hurs'][:,:,:],axis=0)/100.

	QS_mean = q_s(Tmean)
	Q_mean = RH*QS_mean
	gamma = dq_dT(Tmean)

	Tskew = skew(skew_F,skew_P,cross1,cross2,PVAR,SM,Q_mean,QS_mean,gamma,14,75,5)

	return(lon,lat,Tskew)

def GCME_var(c,scenario):

	L = 2257000.
	
	rad_var = Dataset(scenario+'_spread_summer_variance_rsds.nc','r')
	RVAR = np.nanmean(rad_var['rsds'][:,:,:],axis=0)
	lat = rad_var['lat'][:]
	lon = rad_var['lon'][:]
	nlat = len(lat)
	nlon = len(lon)

	pr_var = Dataset(scenario+'_spread_summer_variance_pr.nc','r')

	if scenario == 'HISTORICAL':
		PVAR = np.nanmean(pr_var['pr'][:,:,:],axis=0)
	else:
		PVAR = np.nanmean(pr_var['pr'][:,:,:],axis=0)*(L**2)

	covar = Dataset(scenario+'_spread_summer_covar.nc','r')
	CVAR = np.nanmean(covar['covar'][:,:,:],axis=0)

#	Max fact = 26.565, 95% percentile = 11.3333 90% = 9.28 
#	Fact = 26.565

	mean_pr = Dataset(scenario+'_spread_summer_mean_pr.nc','r')
	MEAN_PR = np.nanmean(mean_pr['pr'][:,:,:],axis=0)

	if scenario == 'HISTORICAL':
		Fact = 26.565
	else:
		Fact = 30

	SM = MEAN_PR*(60*60*24*30)/(10.*Fact)  # in cm/month normalized by 95 percentile
	
	i = 0
	while i < nlat:
		j = 0
		while j < nlon:
			if SM[i,j] >1:
				SM[i,j] = 1
			j = j + 1
		i = i + 1

	Tmean_set = Dataset(scenario+'_spread_summer_mean_tas.nc','r')
	Tmean = np.nanmean(Tmean_set['tas'][:,:,:],axis=0) - 273.15

	RH_set = Dataset(scenario+'_spread_summer_mean_hurs.nc','r')
	RH = np.nanmean(RH_set['hurs'][:,:,:],axis=0)/100.

	QS_mean = q_s(Tmean)
	Q_mean = RH*QS_mean
	gamma = dq_dT(Tmean)
	
	A = dq_dT(Tmean + 0.05)
	B = dq_dT(Tmean - 0.05)
	double_gamma = (A - B)/0.1

	RH_sensitivity_set = Dataset('RH_sensitivity.nc','r')
	RH_sensitivity = c*RH_sensitivity_set['dRH_dT'][:,:]/100.

	delta_T_set = Dataset('delta_T_mean.nc','r')
	delta_T_init = delta_T_set['dT'][:,:]

	Tvar_init = New_toy(RVAR,PVAR,CVAR,SM,Tmean,Q_mean,QS_mean,gamma,14,75,5)
	Tsensitivity = dsigma_dT(RVAR,PVAR,CVAR,SM,Tmean,Q_mean,QS_mean,gamma,double_gamma,Tvar_init,14,75,5,RH_sensitivity)

	return(lon,lat,Tvar_init,Tsensitivity*delta_T_init)

def sensitivity_hist(no_hum):
	no_hum_hist,bins = np.histogram(no_hum,bins=30,range=(-0.2,1.5))
	my_bins = (bins[1:] + bins[0:len(bins)-1])/2.
	plt.plot(my_bins,no_hum_hist/sum(no_hum_hist))
#	plt.show()

def cmip_get_summer(array):

# Assume all grids the same

	Junes = np.linspace(5,19*12 + 5,20)
	Julys = np.linspace(6,19*12 + 6,20)
	Augs  = np.linspace(7,19*12 + 7,20)
		
	All_sums = np.concatenate((Junes,Julys,Augs),axis=0)
	All_sums = np.sort(All_sums)
	All_sums = All_sums.astype(int)

#	Junes = Junes.astype(int)
#	Julys = Julys.astype(int)
#	Augs  = Augs.astype(int)
#
#	June_anoms = array[Junes,:,:] - np.nanmean(array[Junes,:,:],axis=0)
#	July_anoms = array[Julys,:,:] - np.nanmean(array[Julys,:,:],axis=0)
#	Aug_anoms  = array[Augs,:,:]  - np.nanmean(array[Augs,:,:],axis=0)

#	Full_anoms = np.concatenate((June_anoms,July_anoms,Aug_anoms),axis=0)


#	June_anoms2 = array2[Junes,:,:] - np.nanmean(array2[Junes,:,:],axis=0)
#	July_anoms2 = array2[Julys,:,:] - np.nanmean(array2[Julys,:,:],axis=0)
#	Aug_anoms2  = array2[Augs,:,:]  - np.nanmean(array2[Augs,:,:],axis=0)
#		
#	Full_anoms2 = np.concatenate((June_anoms2,July_anoms2,Aug_anoms2),axis=0)

#	skew1 = np.nanmean(Full_anoms**3,axis=0)
#	skew2 = np.nanmean(Full_anoms2**3,axis=0)
	summers_only = np.nanmean(array[All_sums,:,:],axis=0)
	
#	return(Full_anoms,Variance)

#	cross1 = np.nanmean((Full_anoms**2)*Full_anoms2,axis=0)
#	cross2 = np.nanmean((Full_anoms*(Full_anoms2**2)),axis=0)

#	covar = np.nanmean(Full_anoms*Full_anoms2,axis=0)

	return(summers_only)

def Claire_CMIP6(var,scenario):

	from netCDF4 import MFDataset

	path = '/home/disk/eos9/czarakas/Data/CMIP6/'

	model_list = ['CanESM5','CESM2','CNRM-ESM2-1',
		'GFDL-ESM4','IPSL-CM6A-LR',
		'MIROC-ES2L','UKESM1-0-LL']

	nmodels = len(model_list)
	
	Final_lat = np.linspace(-90,90,128)
	Final_lon = np.linspace(0,359,256)

	BEG_MEAN = np.zeros(shape=(nmodels,128,256))
	END_MEAN = np.zeros(shape=(nmodels,128,256))

	i = 0
	while i < nmodels:

		try:
			name = model_list[i]
			var_set = MFDataset(path+name+'/'+var+'/'+var+'_Amon_'+name+'_'+scenario+'_r1*.nc')		
			begin = var_set[var][:20*12,:,:]
			end   = var_set[var][120*12:140*12,:,:]	

			begin_mean = cmip_get_summer(begin)
			end_mean   = cmip_get_summer(end)

			print('No issue with ')
			print(name)

		except ValueError:
			name = model_list[i]
			print('There was an issue with')
			print(name)
			if name == 'CESM2': 
				var_set = Dataset(path+name+'/'+var+'/'+var+'_Amon_'+name+'_'+scenario+'_r1i1p1f1_gn_000101-005012.nc')	
				ivar_set = Dataset(path+name+'/'+var+'/'+var+'_Amon_'+name+'_'+scenario+'_r1i1p1f1_gn_010101-015012.nc')		
				begin = var_set[var][:20*12,:,:]
				end   = ivar_set[var][20*12:40*12,:,:]	
			if name == 'CNRM-ESM2-1':	
				if scenario == '1pctCO2':
					var_set = Dataset(path+name+'/'+var+'/'+var+'_Amon_'+name+'_'+scenario+'_r1i1p1f2_gr_185001-199912.nc')	
				else:
	
					var_set = Dataset(path+name+'/'+var+'/'+var+'_Amon_'+name+'_'+scenario+'_r1i1p1f2_gr_185001-198912.nc')	
				begin = var_set[var][:20*12,:,:]
				end = var_set[var][120*12:140*12,:,:]

				
			begin_mean = cmip_get_summer(begin)
			end_mean   = cmip_get_summer(end)

			print('But It"s OK')

		mod_lat = var_set['lat'][:]
		mod_lon = var_set['lon'][:]
		nlats = len(mod_lat)
		nlons = len(mod_lon)
		
		mlon = np.linspace(0,359,nlons)
		mlat = np.linspace(-90,90,nlats)

		REGRID_BEG = hor_interp(begin_mean,mlon,mlat,Final_lon,Final_lat)
		REGRID_END = hor_interp(end_mean,mlon,mlat,Final_lon,Final_lat)

		BEG_MEAN[i,:,:] = REGRID_BEG
		END_MEAN[i,:,:] = REGRID_END

		i = i + 1

	write_netcdf('1pctCO2-rad_spread_end_mean_tas.nc','tas',Final_lat,Final_lon,END_MEAN)
	write_netcdf('1pctCO2-rad_spread_beginning_mean_tas.nc','tas',Final_lat,Final_lon,BEG_MEAN)

def CMIP6(var,va,scenario):

	from netCDF4 import MFDataset

	path = '/home/disk/eos9/cmip6/'+scenario+'/'
#	path = '/home/disk/eos10/lvz7/hurs_'+scenario+'/'

	model_list = ['ACCESS-CM2','ACCESS-ESM1-5','CanESM5','CESM2','CESM2-WACCM',
		'CNRM-CM6-1','CNRM-ESM2-1','CNRM-CM6-1-HR','EC-Earth3','EC-Earth3-Veg',
		'GFDL-ESM4','GISS-E2-1-G','HadGEM3-GC31-LL','INM-CM4-8','INM-CM5-0','IPSL-CM6A-LR',
		'MIROC6','MIROC-ES2L','MPI-ESM1-2-HR','MPI-ESM1-2-LR','MRI-ESM2-0','NorESM2-MM','NorESM2-LM','UKESM1-0-LL']

	nmodels = len(model_list)
	print(nmodels)

	Final_lat = np.linspace(-90,90,128)
	Final_lon = np.linspace(0,359,256)

#	BIG_COVAR = np.zeros(shape=(nmodels,128,256))
	BIG_VAR1 = np.zeros(shape=(nmodels,128,256))
	BIG_VAR2 = np.zeros(shape=(nmodels,128,256))
#	BIG_MEAN = np.zeros(shape=(nmodels,128,256))

	L = 2257000. # J/kg H2O

	i = 0

	while i < nmodels:
		print(model_list[i])
		name = model_list[i]
		try:

#			if name == 'NorESM2-LM':
#
#				var_set = MFDataset(path+name+'/'+var+'/'+var+'_Amon_'+scenario+'_'+name+'_r1i1p1f*.nc')
#				var_data = var_set[var][144*12:164*12,:,:]

#				var2_set = Dataset(path+name+'/'+va+'/'+va+'_Amon_'+scenario+'_'+name+'_r1i1p1f1_gn_199001-199912.nc')		
#				jvar_set = Dataset(path+name+'/'+va+'/'+va+'_Amon_'+scenario+'_'+name+'_r1i1p1f1_gn_200001-200912.nc')		
#				kvar_set = Dataset(path+name+'/'+va+'/'+va+'_Amon_'+scenario+'_'+name+'_r1i1p1f1_gn_201001-201412.nc')
#
#				iv = var2_set[va][4*12:,:,:]
#				jv = jvar_set[va][:,:,:]
##				kv = kvar_set[va][:,:,:]
#				var2_data = np.concatenate((iv,jv,kv),axis=0)


			if name == 'GISS-E2-1-G':

				var_set = MFDataset(path+name+'/'+var+'/'+var+'_Amon_'+name+'_'+scenario+'_r1i1p*.nc')		
#				var2_set = MFDataset(path+name+'/'+va+'/'+va+'_Amon_'+name+'_'+scenario+'_r1i1p*.nc')		
#				var_set = MFDataset(path+var+'_Amon_'+name+'_'+scenario+'_r1i1p*.nc')		
	
#				var_data = var_set[var][144*12:164*12,:,:] # For historical
#				var2_data = var2_set[va][144*12:164*12,:,:]

				var_data = var_set[var][64*12:84*12,:,:]
#				var2_data = var2_set[va][64*12:84*12,:,:]

			else:

#				var_set = MFDataset(path+var+'_Amon_'+name+'_'+scenario+'_r1i1p*.nc')		
				var_set = MFDataset(path+name+'/'+var+'/'+var+'_Amon_'+name+'_'+scenario+'_r1i1p1f*.nc')		
#				var2_set = MFDataset(path+name+'/'+va+'/'+va+'_Amon_'+name+'_'+scenario+'_r1i1p1f*.nc')		

#				var_data = var_set[var][144*12:164*12,:,:]
#				var2_data = var2_set[va][144*12:164*12,:,:] # For historical

				var_data = var_set[var][64*12:84*12,:,:]
#				var2_data = var2_set[va][64*12:84*12,:,:]

#				var_data = var_set[var][3432-240:,:,:]

				print(var_data.shape)


			print("NO ISSUE WITH....")
#			print(model_list[i])

#			lat = var_set['lat'][:]
#			lon = var_set['lon'][:]
			summers_only= cmip_get_summer(var_data)


#		
#			return(lon,lat,np.nanmean(summers_only,axis=0))


#			covar = np.nanmean(anoms1*anoms2*L,axis=0)

		except ValueError:

			print("THERE WAS SOME ISSUE WITH...")
			print(model_list[i])
			if name == 'CESM2':			

				var_set = Dataset(path+name+'/'+var+'/'+var+'_Amon_'+name+'_'+scenario+'_r1i1p1f1_gn_206501-210012.nc')		
#				var2_set = Dataset(path+name+'/'+va+'/'+va+'_Amon_'+name+'_'+scenario+'_r1i1p1f1_gn_206501-210012.nc')		

#				var_set = Dataset(path+name+'/'+var+'/'+var+'_Amon_'+name+'_'+scenario+'_r1i1p1f1_gn_185001-201412.nc')		
#				var2_set = Dataset(path+name+'/'+va+'/'+va+'_Amon_'+name+'_'+scenario+'_r1i1p1f1_gn_185001-201412.nc')		
#				var_set = Dataset(path+var+'_Amon_'+name+'_'+scenario+'_r11i1p1f1_gn_195001-199912.nc')		
#				jvar_set = Dataset(path+var+'_Amon_'+name+'_'+scenario+'_r11i1p1f1_gn_200001-201412.nc')		
#				iv = var_set[var][45*12:,:,:]
#				jv = jvar_set[var][:,:,:]
#				var_data = np.concatenate((iv,jv),axis=0)

#				var_data = var_set[var][144*12:164*12,:,:]					
#				var2_data = var2_set[va][144*12:164*12,:,:]					

				var_data = var_set[var][14*12:34*12,:,:]
#				var2_data = var2_set[va][14*12:34*12,:,:]

			elif name == 'CNRM-ESM2-1' or name == 'CNRM-CM6-1':	

				print(name)
#				var_set = Dataset(path+name+'/'+var+'/'+var+'_Amon_'+name+'_'+scenario+'_r1i1p1f2_gr_185001-201412.nc')	
#				var2_set = Dataset(path+name+'/'+va+'/'+va+'_Amon_'+name+'_'+scenario+'_r1i1p1f2_gr_185001-201412.nc')	
				var_set = Dataset(path+name+'/'+var+'/'+var+'_Amon_'+name+'_'+scenario+'_r2i1p1f2_gr_201501-210012.nc')		
#				var2_set = Dataset(path+name+'/'+va+'/'+va+'_Amon_'+name+'_'+scenario+'_r2i1p1f2_gr_201501-210012.nc')		

#				var_data = var_set[var][144*12:164*12,:,:]
#				var2_data = var2_set[va][144*12:164*12,:,:]

				var_data = var_set[var][64*12:84*12,:,:]
#				var2_data = var2_set[va][64*12:84*12,:,:]

			elif name == 'CNRM-CM6-1-HR':	

				print(name)
#				var_set = Dataset(path+name+'/'+var+'/'+var+'_Amon_'+name+'_'+scenario+'_r1i1p1f2_gr_185001-201412.nc')	
#				var2_set = Dataset(path+name+'/'+va+'/'+va+'_Amon_'+name+'_'+scenario+'_r1i1p1f2_gr_185001-201412.nc')	
				var_set = Dataset(path+name+'/'+var+'/'+var+'_Amon_'+name+'_'+scenario+'_r1i1p1f2_gr_201501-210012.nc')		
#				var2_set = Dataset(path+name+'/'+va+'/'+va+'_Amon_'+name+'_'+scenario+'_r1i1p1f2_gr_201501-210012.nc')		
#				var_set = Dataset(path+var+'_Amon_'+name+'_'+scenario+'_r1i1p1f2_gr_185001-201412.nc')		
#				var_set = Dataset(path+var+'_Amon_'+name+'_'+scenario+'_r1i1p1f2_gr_201501-210012.nc')		

#				var_data = var_set[var][144*12:164*12,:,:]
#				var2_data = var2_set[va][144*12:164*12,:,:]

				var_data = var_set[var][64*12:84*12,:,:]
#				var2_data = var2_set[va][64*12:84*12,:,:]
			elif name == 'IPSL-CM6A-LR':


#				var_set = Dataset(path+name+'/'+var+'/'+var+'_Amon_'+name+'_'+scenario+'_r1i1p1f1_gr_185001-201412.nc')	
#				var2_set = Dataset(path+name+'/'+va+'/'+va+'_Amon_'+name+'_'+scenario+'_r1i1p1f1_gr_185001-201412.nc')	
				var_set = Dataset(path+name+'/'+var+'/'+var+'_Amon_'+name+'_'+scenario+'_r1i1p1f1_gr_201501-210012.nc')		
#				var2_set = Dataset(path+name+'/'+va+'/'+va+'_Amon_'+name+'_'+scenario+'_r1i1p1f1_gr_201501-210012.nc')		
#				var_data = var_set[var][65*12:85*12,:,:]
#				var_set = Dataset(path+var+'_Amon_'+name+'_'+scenario+'_r1i1p1f1_gr_185001-201412.nc')		

#				var_data = var_set[var][144*12:164*12,:,:]
#				var2_data = var2_set[va][144*12:164*12,:,:]

				var_data = var_set[var][64*12:84*12,:,:]
#				var2_data = var2_set[va][64*12:84*12,:,:]
#
			elif name == 'CESM2-WACCM':
				
				var_set = Dataset(path+name+'/'+var+'/'+var+'_Amon_'+name+'_'+scenario+'_r1i1p1f1_gn_201501-210012.nc')		
#				var_set = Dataset(path+var+'_Amon_'+name+'_'+scenario+'_r1i1p1f1_gn_185001-201512.nc')		
#				var_set = Dataset(path+name+'/'+var+'/'+var+'_Amon_'+name+'_'+scenario+'_r1i1p1f1_gn_185001-201412.nc')		
#				var2_set = Dataset(path+name+'/'+va+'/'+va+'_Amon_'+name+'_'+scenario+'_r1i1p1f1_gn_185001-201412.nc')		
#				var_data = var_set[var][64*12:84*12,:,:]
#				var_data = var_set[var][144*12:164*12,:,:]
#				var2_data = var2_set[va][144*12:164*12,:,:]

#				var_set = Dataset(path+name+'/'+var+'/'+var+'_Amon_'+name+'_'+scenario+'_r1i1p1f1_gn_201501-210012.nc')		
##				var2_set = Dataset(path+name+'/'+va+'/'+va+'_Amon_'+name+'_'+scenario+'_r1i1p1f1_gn_201501-210012.nc')		
				var_data = var_set[var][64*12:84*12,:,:]
#				var2_data = var2_set[va][64*12:84*12,:,:]


			print("BUT It'S OK")
			

#			var_data = var_set[var][144*12:164*12,:,:] # for historical
			summers_only = cmip_get_summer(var_data)
#
#			covar = np.nanmean(anoms1*anoms2*L,axis=0)



		mod_lat = var_set['lat'][:]
		mod_lon = var_set['lon'][:]
		nlats = len(mod_lat)
		nlons = len(mod_lon)
	
		mlon = np.linspace(0,359,nlons)
		mlat = np.linspace(-90,90,nlats)
	
#		REGRID_COVAR = hor_interp(covar,mlon,mlat,Final_lon,Final_lat)
		REGRID_VAR1  = hor_interp(summers_only,mlon,mlat,Final_lon,Final_lat)
#		REGRID_VAR2  = hor_interp(skew2,mlon,mlat,Final_lon,Final_lat)

#		print(np.nanmean(REGRID_VAR1))
#		print(np.nanmean(REGRID_VAR2))

#		mean2 = np.nanmean(var2_data,axis=0)

#		REGRID_MEAN  = hor_interp(mean,mlon,mlat,Final_lon,Final_lat)
#		print(np.nanmean(REGRID_COVAR),np.nanmean(REGRID_VAR1),np.nanmean(REGRID_VAR2))

#		BIG_MEAN[i,:,:] = REGRID_MEAN
#		BIG_COVAR[i,:,:] = REGRID_COVAR	
		BIG_VAR1[i,:,:] = REGRID_VAR1
#		BIG_VAR2[i,:,:] = REGRID_VAR2

		i = i + 1
#	return(Final_lon,Final_lat,REGRID_VAR1)

#	f = breakher
#	MMM = np.mean(BIG_MEAN,axis=0)
#	write_netcdf('SSP585_spread_summer_mean_pr.nc','pr',Final_lat,Final_lon,BIG_VAR1)
#	write_netcdf('HISTORICAL_spread_summer_third_moment_tas.nc','tas',Final_lat,Final_lon,BIG_VAR1)
#	write_netcdf('HISTORICAL_spread_summer_cross2.nc','cross2',Final_lat,Final_lon,BIG_VAR2)

#	write_netcdf('HISTORICAL_spread_summer_mean_pr.nc','pr',Final_lat,Final_lon,BIG_MEAN)

def area_avg(lon,lat,Array):

	lsm_set = Dataset('era5_lsm.nc')
	lsm = lsm_set['lsm'][0,:,:]
	lsm_lat = lsm_set['latitude'][:]
	lsm_lon = lsm_set['longitude'][:]
	interp_lsm = hor_interp(lsm,lsm_lon,lsm_lat,lon,lat)

#	agree_set = Dataset('Agree_map.nc')
#	agree_data = agree_set['agree'][:,:]

	nlats = len(lat)
	nlons = len(lon)
	nmodels = len(Array[:,0,0])
	dT = np.zeros(nmodels)

	i = 0
	while i < nmodels:
		deltas = []
		weights = []
		j = 0
		while j < nlats:
			k = 0
			while k < nlons:
				if interp_lsm[j,k] > 0.5 and lat[j] > 30:# and agree_data[j,k] > 0.75:
					deltas.append(Array[i,j,k]*np.cos(lat[j]*np.pi/180.))
					weights.append(np.cos(lat[j]*np.pi/180.))
				k = k + 1
			j = j + 1
#
#		print(len(deltas),len(weights))
		dT[i] = np.sum(deltas)/np.sum(weights)
		i += 1
#
	print(dT)
	write_list('dTmean.nc',dT)				

def cor_maps():

#	hist_mean_set = Dataset('HISTORICAL_spread_summer_mean_tas.nc','r')
#	Hmean = hist_mean_set['tas'][:,:,:]
#
#	ssp_mean_set = Dataset('SSP585_spread_summer_mean_tas.nc','r')
#	Smean = ssp_mean_set['tas'][:,:,:]


	hist_var_set = Dataset('HISTORICAL_spread_summer_variance_tas.nc','r')
	Hvar = hist_var_set['tas'][:,:,:]
#
	ssp_var_set = Dataset('SSP585_spread_summer_variance_tas.nc','r')
	Svar = ssp_var_set['tas'][:,:,:]

	lat = ssp_var_set['lat'][:]
	lon = ssp_var_set['lon'][:]

#	agree_set = Dataset('Agree_map_tvar.nc')
#	agree_data = agree_set['agree'][:,:]
#	plotting(lon,lat,Svar-Hvar,agree_data)	
	return(lon,lat,np.nanmean((Svar-Hvar),axis=0))

#	plotting(lon,lat,Smean-Hmean)

def plant_physio(scenario):

	h_var = 'hurs'
	t_var = 'tas'

#	beg_set_h = Dataset(scenario+'_spread_beginning_mean_'+h_var+'.nc','r')
#	end_set_h = Dataset(scenario+'_spread_end_mean_'+h_var+'.nc','r')

	beg_set_t = Dataset(scenario+'_spread_beginning_mean_'+t_var+'.nc','r')
	end_set_t = Dataset(scenario+'_spread_end_mean_'+t_var+'.nc','r')

	lat = beg_set_t['lat'][:]
	lon = beg_set_t['lon'][:]

#	HBeginning = np.nanmean(beg_set_h[h_var][:,:,:],axis=0)
#	HEnd = np.nanmean(end_set_h[h_var][:,:,:],axis=0)
	
	hbegin = beg_set_t[t_var][:,:,:]
	hend   = end_set_t[t_var][:,:,:]

	return(lon,lat,np.nanstd(hend-hbegin,axis=0))

def master_hist():

	lon,lat,Tvar_pred = GCME_var(1,'HISTORICAL')

	Hist_T_set = Dataset('HISTORICAL_spread_summer_variance_tas.nc','r')
	T_hist = np.nanmean(Hist_T_set['tas'][:,:,:],axis=0)

	Z = plotting(lon,lat,Tvar_pred/T_hist)

	hist,bins = np.histogram(Z,bins=30,range=(0.1,10))
	mybins = (bins[0:len(bins)-1] + bins[1:])/2.

	Obs_hist = Dataset('Observations_histogram.nc','r')
	X_obs = Obs_hist['bin_value'][:]
	Y_obs = Obs_hist['hist_value'][:]

	ERA5_hist = Dataset('ERA5_histogram.nc','r')
	X_era = ERA5_hist['bin_value'][:]
	Y_era = ERA5_hist['hist_value'][:]

	plt.semilogx(mybins,hist/sum(hist),linewidth=3)
	plt.semilogx(X_obs,Y_obs,linewidth=3)
	plt.semilogx(X_era,Y_era,linewidth=3)
	plt.xlim(.25,4)
#	plt.savefig('Maser_histogram.pdf')
	plt.show()


#master_hist()

#CMIP6('pr','pr','ssp585')

# First METHOD, differences across realizations of the model

#lon,lat,Initial_var_pred,change = GCME_var(1,'HISTORICAL')
#lon,lat,Final_var_pred,change = GCME_var(1,'SSP585')

# Second Method, estimating directly from the models

#T_var_hist = Dataset('HISTORICAL_spread_summer_variance_tas.nc','r')
#T_var_ssp = Dataset('SSP585_spread_summer_variance_tas.nc','r')

# Third Method, use sensitivity equation with just temperature

#lon,lat,Tvar_init,change = GCME_var(1,'HISTORICAL')
#plotting(lon,lat,Final_var_pred-Initial_var_pred)

################################################################################################3

T_skew_hist = Dataset('HISTORICAL_spread_summer_skew_rsds.nc','r')
T_var_hist  = Dataset('HISTORICAL_spread_summer_variance_rsds.nc','r')

lon = T_skew_hist['lon'][:]
lat = T_skew_hist['lat'][:]

T_skew = np.nanmean(T_skew_hist['rsds'][:,:,:],axis=0)
T_var  = np.nanmean(T_var_hist['rsds'][:,:,:],axis=0)

plotting(lon,lat,T_skew/(T_var**1.5))

#i

######################### SCATTER 

#T_hist = Dataset('HISTORICAL_spread_summer_mean_tas.nc','r')
#T_ssp  = Dataset('SSP585_spread_summer_mean_tas.nc','r')

#RH_hist = Dataset('HISTORICAL_spread_summer_mean_hurs.nc','r')
#RH_ssp  = Dataset('SSP585_spread_summer_mean_hurs.nc','r')

#T_H = T_hist['tas'][:,:,:]
#T_S = T_ssp['tas'][:,:,:]

#RH_H = RH_hist['hurs'][:,:,:]
#RH_S = RH_ssp['hurs'][:,:,:]

#lon = RH_hist['lon'][:]
#lat = RH_hist['lat'][:]

#Model_scatter(lon,lat,T_S - T_H,RH_S - RH_H)

