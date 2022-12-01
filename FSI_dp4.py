'''
Created 22/11/2022
Calculates FSI for DePreSys

@vikki.thompson
'''

import math
import os
import glob
import subprocess
import iris
import iris.coord_categorisation as icc
from iris.coord_categorisation import add_season_membership
import numpy as np
import matplotlib.pyplot as plt

# Regions
lookup_lon = {'UK':[-10., 2.],'ewp':[-5.2, 1.6],'nwe':[-4.5,-2.2],'swe':[-5,-2.5],'see':[-2.5,1.6],'mid':[-2.5,-.5],'ea':[-.5, 2],'nee':[-2.5,0],'nee2':[-.5,0],'ws':[-6,-3.5],'es':[-3.5,-1.5],'ns':[-6.,-2.5],'ni':[-7.5,-5.5]}
lookup_lat = {'UK':[49., 59.],'ewp':[50.1, 54.4],'nwe':[52.8,54.9],'swe':[50.1,52.8],'see':[50.1,51.7],'mid':[51.7,54],'ea':[51.7,52.8],'nee':[54,55],'nee2':[52.8,54],'ws':[54.9,56.5],'es':[54.9,57],'ns':[56.5,59],'ni':[53.8,55]}



### LOAD IN DEPRESYS DATA, SINGLE HINDCAST
def list_to_uk(list):
    cube = list.concatenate_cube()
    return cube.intersection(longitude=[-10, 2], latitude = [49,59]) # UK (excludes Shetland?)

hindcast = 'ay395'
iris.load('/gws/nopw/j04/bris_climdyn/vikkitho/u-ay191/r001/ay191a.p619961101.pp')

# For 10 realizations of 'hindcast'
temp_ens = iris.cube.CubeList([]); prec_ens = iris.cube.CubeList([]); wind_ens = iris.cube.CubeList([]); rhum_ens = iris.cube.CubeList([])
for ens in range(1,4):  # ensembles 1-3 
    print(ens)
    ens_in = str(ens).zfill(3)
    filepath = '/gws/nopw/j04/bris_climdyn/vikkitho/u-'+hindcast+'/r'+ens_in+'/*'
    files = glob.glob(filepath)[1:]
    temp_list = iris.cube.CubeList([]); prec_list = iris.cube.CubeList([]); rhum_list = iris.cube.CubeList([]); wind_list = iris.cube.CubeList([])
    for each in files:
        print(each)
        cubes = iris.load(each)#, calendar='360_day')
        temp_list.append(cubes[0])
        prec_list.append(cubes[1])
        rhum_list.append(cubes[2])
        wind_list.append(cubes[3])
    temp_ens.append(list_to_uk(temp_list))
    prec_ens.append(list_to_uk(prec_list))
    rhum_ens.append(list_to_uk(rhum_list))
    wind_ens.append(list_to_uk(wind_list))

temp = temp_ens.merge_cube()
prec = prec_ens.merge_cube()
rhum = rhum_ens.merge_cube()
wind = wind_ens.merge_cube()





### CALCULATE FWI

class FWICLASS:
    def __init__(self,temp,rhum,wind,prcp):
        self.h = rhum
        self.t = temp
        self.w = wind
        self.p = prcp
    def FFMCcalc(self,ffmc0):
        mo = (147.2*(101.0 - ffmc0))/(59.5 + ffmc0) #*Eq. 1*#
        if (self.p > 0.5):
            rf = self.p - 0.5 #*Eq. 2*#
            if(mo > 150.0):
                mo = (mo+42.5*rf*np.exp(-100.0/(251.0-mo))*
                      (1.0 - np.exp(-6.93/rf))) + (.0015*(mo - 150.0)**2)*math.sqrt(rf) #*Eq. 3b*#
            elif mo <= 150.0:
                mo = mo+42.5*rf*np.exp(-100.0/(251.0-mo))*(1.0 - np.exp(-6.93/rf)) #*Eq. 3a*#
            if(mo > 250.0):
                mo = 250.0
        ed = .942*(self.h**.679) + (11.0*np.exp((self.h-100.0)/10.0))+0.18*(21.1-self.t)*(1.0 - 1.0/math.exp(.1150 * self.h)) #*Eq. 4*#
        if(mo < ed):
            ew = .618*(self.h**.753) + (10.0*np.exp((self.h-100.0)/10.0))+ .18*(21.1-self.t)*(1.0 - 1.0/np.exp(.115 * self.h)) #*Eq. 5*#
            if(mo <= ew):
                kl = .424*(1.0-((100.0-self.h)/100.0)**1.7)+(.0694*math.sqrt(self.w))*(1.0 - ((100.0 - self.h)/100.0)**8) #*Eq. 7a*#
                kw = kl * (.581 * np.exp(.0365 * self.t)) #*Eq. 7b*#
                m = ew - (ew - mo)/10.0**kw #*Eq. 9*#
            elif mo > ew:
                m = mo
        elif(mo == ed):
            m = mo
        elif mo > ed:
            kl =.424*(1.0-(self.h/100.0)**1.7)+(.0694*math.sqrt(self.w))*(1.0-(self.h/100.0)**8) #*Eq. 6a*#
            kw = kl * (.581*np.exp(.0365*self.t)) #*Eq. 6b*#
            m = ed + (mo-ed)/10.0 ** kw #*Eq. 8*#
        ffmc = (59.5 * (250.0 -m)) / (147.2 + m)#*Eq. 10*#
        if (ffmc > 101.0):
            ffmc = 101.0
        if (ffmc <= 0.0):
            ffmc = 0.0
        return ffmc 
    def DMCcalc(self,dmc0,mth):
        el = [6.5,7.5,9.0,12.8,13.9,13.9,12.4,10.9,9.4,8.0,7.0,6.0]
        t = self.t
        if (t < -1.1):
            t = -1.1
        rk = 1.894*(t+1.1) * (100.0-self.h) * (el[mth-1]*0.0001) #*Eqs. 16 and 17*#
        if self.p > 1.5:
            ra= self.p
            rw = 0.92*ra - 1.27 #*Eq. 11*#
            wmi = 20.0 + 280.0/np.exp(0.023*dmc0) #*Eq. 12*#
            if dmc0 <= 33.0:
                b = 100.0 /(0.5 + 0.3*dmc0) #*Eq. 13a*#
            elif dmc0 > 33.0:
                if dmc0 <= 65.0:
                    b = 14.0 - 1.3*math.log(dmc0) #*Eq. 13b*#
                elif dmc0 > 65.0:
                    b = 6.2 * math.log(dmc0) - 17.2 #*Eq. 13c*#
            wmr = wmi + (1000*rw) / (48.77+b*rw) #*Eq. 14*#
            pr = 43.43 * (5.6348 - math.log(wmr-20.0)) #*Eq. 15*#
        elif self.p <= 1.5:
            pr = dmc0
        if (pr<0.0):
            pr = 0.0
        dmc = pr + rk
        if(dmc<= 1.0):
            dmc = 1.0
        return dmc
    def DCcalc(self,dc0,mth):
        fl = [-1.6, -1.6, -1.6, 0.9, 3.8, 5.8, 6.4, 5.0, 2.4, 0.4, -1.6, -1.6]
        t = self.t
        if(t < -2.8):
            t = -2.8
        pe = (0.36*(t+2.8) + fl[mth-1] )/2 #*Eq. 22*#
        if pe <= 0.0:
            pe = 0.0
        if self.p > 2.8:
            ra = self.p
            rw = 0.83*ra - 1.27 #*Eq. 18*#
            smi = 800.0 * np.exp(-dc0/400.0) #*Eq. 19*#
            dr = dc0 - 400.0*math.log( 1.0+((3.937*rw)/smi) ) #*Eqs. 20 and 21*#
            if (dr > 0.0):
                dc = dr + pe
        elif self.p <= 2.8:
            dc = dc0 + pe
        return dc
    def ISIcalc(self,ffmc):
        mo = 147.2*(101.0-ffmc) / (59.5+ffmc) #*Eq. 1*#
        ff = 19.115*np.exp(mo*-0.1386) * (1.0+(mo**5.31)/49300000.0) #*Eq. 25*#
        isi = ff * np.exp(0.05039*self.w) #*Eq. 26*#
        return isi
    def BUIcalc(self,dmc,dc):
        if dmc <= 0.4*dc:
            bui = (0.8*dc*dmc) / (dmc+0.4*dc) #*Eq. 27a*#
        else:
            bui = dmc-(1.0-0.8*dc/(dmc+0.4*dc))*(0.92+(0.0114*dmc)**1.7) #*Eq. 27b*#
        if bui <0.0:
            bui = 0.0
        return bui
    def FWIcalc(self,isi,bui):
        if bui <= 80.0:
            bb = 0.1 * isi * (0.626*bui**0.809 + 2.0) #*Eq. 28a*#
        else:
            bb = 0.1*isi*(1000.0/(25. + 108.64/np.exp(0.023*bui))) #*Eq. 28b*#
        if(bb <= 1.0):
            fwi = bb #*Eq. 30b*#
        else:
            fwi = np.exp(2.72 * (0.434*np.log(bb))**0.647) #*Eq. 30a*#
        return fwi




def fwi_calc(temp_list, rhum_list, wind_list, prec_list, mth_list, day_list):
    ffmc_list = []; dmc_list = []; dc_list = []; isi_list = []; bui_list = []; fwi_list = []
    ffmc0 = 85.0    # default start-up. Need calibrating for location
    dmc0 = 6.0  # default
    dc0 = 15.0  # default
    for i in np.arange(len(temp_list)):  
        #print(i)
        mth = int(mth_list[i])
        day = day_list[i]
        tmp = temp_list[i]
        hum = rhum_list[i]
        wnd = wind_list[i]
        if wnd<0:
            wnd = abs(wnd)
        pcp = prec_list[i]
        if hum>100.0:
            hum = 100.0
        fwisystem = FWICLASS(tmp,hum,wnd,pcp)
        ffmc_list.append(fwisystem.FFMCcalc(ffmc0))
        dmc_list.append(fwisystem.DMCcalc(dmc0,mth))
        dc_list.append(fwisystem.DCcalc(dc0,mth))
        isi_list.append(fwisystem.ISIcalc(ffmc_list[i]))
        bui_list.append(fwisystem.BUIcalc(dmc_list[i],dc_list[i]))
        fwi_list.append(fwisystem.FWIcalc(isi_list[i],bui_list[i]))
        ffmc0 = ffmc_list[i]
        dmc0 = dmc_list[i]
        dc0 = dc_list[i]
    return ffmc_list, dmc_list, dc_list, isi_list, bui_list, fwi_list


def regional_ts(name, cube):
    cube = cube.intersection(longitude = lookup_lon[name], latitude = lookup_lat[name])
    return cube.collapsed(['latitude', 'longitude'], iris.analysis.MEAN)

# Calculate regional values
temp_see = regional_ts('see', temp)
prec_see = regional_ts('see', prec)
rhum_see = regional_ts('see', rhum)
wind_see = regional_ts('see', wind)

# correct units
temp_see = temp_see - 273.15 # to *C
prec_see = prec_see * 86400 / 1000 # to m
wind_see = wind_see * 3.6 # km/hr


## Calculates outputs, and saves in out_dir
# extra inputs (check if still needed)
iris.coord_categorisation.add_month_number(temp_see, 'time') # add month
iris.coord_categorisation.add_day_of_month(temp_see, 'time') # add day of month
mth_list = temp_see.coord('month_number').points
day_list = temp_see.coord('day_of_month').points
 
# make output arrays
isi_out = np.zeros(shape=np.shape(temp_see.data))
bui_out = np.zeros(shape=np.shape(temp_see.data))
fwi_out = np.zeros(shape=np.shape(temp_see.data))

for each in [temp_see, prec_see, rhum_see, wind_see]:
    iris.coord_categorisation.add_season(each, coord='time', name='season')
    iris.coord_categorisation.add_year(each, coord='time', name='year')

# fill outputs
for ens in range(0, 3):
    print(ens)
    Y1 = temp_see.coord('year').points[0]
    Y2 = temp_see.coord('year').points[-1]
    fwi_list = []
    for i, yr in enumerate(np.arange(Y1, Y2+1)):
        print(yr)
        temp_list = temp_see.extract(iris.Constraint(year=yr)).data[ens,:]
        rhum_list = rhum_see.extract(iris.Constraint(year=yr)).data[ens,:]
        prec_list = prec_see.extract(iris.Constraint(year=yr)).data[ens,:]
        wind_list = wind_see.extract(iris.Constraint(year=yr)).data[ens,:]
        _, _, _, isi_list, bui_list, fwi_list_yr = fwi_calc(temp_list, rhum_list, wind_list, prec_list, mth_list[:120], day_list[:120])
        fwi_list.extend(fwi_list_yr)
    #isi_out[ens,:] = isi_list
    #bui_out[ens,:] = bui_list
    fwi_out[ens,:] = fwi_list


# FWI to cube
fwi_see = temp_see.copy() 
fwi_see.data = fwi_out

'''
### SAVE OUTPUTS
out_dir = '/gws/nopw/j04/bris_climdyn/vikkitho/'
iris.save(temp_see, out_dir+hindcast+'_temp_see.nc')
iris.save(prec_see, out_dir+hindcast+'_prec_see.nc')
iris.save(rhum_see, out_dir+hindcast+'_rhum_see.nc')
iris.save(wind_see, out_dir+hindcast+'_wind_see.nc')
iris.save(fwi_see, out_dir+hindcast+'_fwi_see.nc')
'''


T = temp_see.extract(iris.Constraint(season='jja'))
P = prec_see.extract(iris.Constraint(season='jja'))
H = rhum_see.extract(iris.Constraint(season='jja'))
W = wind_see.extract(iris.Constraint(season='jja'))
FW = fwi_see.extract(iris.Constraint(season='jja'))

## PLOTTING
# Data load
plt.ion()
plt.show()

def data_plot1(cube, ax_val, label):
    ' Plot year v value ' 
    ax = plt.subplot2grid((5, 1), (ax_val, 0), colspan=1, rowspan=1)
    Y1 = cube.coord('year').points[0]
    Y2 = cube.coord('year').points[-1]
    for yr in np.arange(Y1, Y2):
        each_yr = cube.extract(iris.Constraint(year=yr)).data
        ax.plot(np.repeat(yr, len(each_yr)), each_yr, '+', color='salmon') 
    ax.set_xticks(np.arange(Y1, Y2, 5))
    ax.set_ylabel(label)


def data_plot2(cube, ax_val, ens, label):
    ' Plot year v value '
    ax = plt.subplot2grid((5, 1), (ax_val, 0), colspan=1, rowspan=1)
    Y1 = cube.coord('year').points[0]
    Y2 = cube.coord('year').points[-1]
    for i, yr in enumerate(np.arange(Y1, Y2)):
        each_yr = cube.extract(iris.Constraint(year=yr)).data
        for en in np.arange(ens):
            ax.plot(each_yr[en, :], color='red', alpha=(i/20+.05)) 
    ax.set_xticks([0, 30, 60])
    ax.set_xticklabels(['June 1st', 'July 1st', 'August 1st'])
    ax.set_ylabel(label)

### Make this one a histogram
def data_plot3(cube, ax_val, nbins, label):
    ' Plot year v value '
    ax = plt.subplot2grid((5, 1), (ax_val, 0), colspan=1, rowspan=1)
    ax.hist((cube.data).reshape((900*3)), bins=nbins, histtype='stepfilled', \
            color='k', alpha=0.5, label='ERA-5')
    plt.yticks([])
    ax.set_ylabel(label)


# plotted by year
fig = plt.figure(figsize=(7., 8.), dpi=80, num=None)
data_plot1(T, 0, 'Temp, *C')
data_plot1(P/1000, 1, 'Prec, mm')
data_plot1(H, 2, 'RHum, %')
data_plot1(W, 3, 'Wind, km/hr')
data_plot1(FW, 4, 'FWI')

# Plotted by day of year
fig = plt.figure(figsize=(7., 8.), dpi=80, num=None)
ens = 3
data_plot2(T, 0, ens, 'Temp, *C')
data_plot2(P/1000, 1, ens, 'Prec, mm')
data_plot2(H, 2, ens, 'RHum, %')
data_plot2(W, 3, ens, 'Wind, km/hr')
data_plot2(FW, 4, ens, 'FWI')

# Plotted as histograms
fig = plt.figure(figsize=(7., 8.), dpi=80, num=None)
data_plot3(T, 0, range(10, 30, 1), 'Temp, *C')
data_plot3(P/1000, 1, np.arange(0, 4, .5), 'Prec, mm')
data_plot3(H, 2, range(50, 100, 2), 'RHum, %')
data_plot3(W, 3, range(0, 60, 2), 'Wind, km/hr')
data_plot3(FW, 4, range(0, 70, 2), 'FWI')

