#!/usr/bin/env python
# coding: utf-8

# # OpenMC Program for BurnUp analysis and Benchmarking
# 

# In[1]:


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 11:40:23 2020

@author: feryantama
"""


# ## 1. Initialization
# 
# Initialize package we used in this program



# In[2]:
import numpy as np
import pandas as pd
from random import seed, random
import math
import os
import openmc.deplete
import openmc
import argparse
'''
from scipy.interpolate import interp2d
import matplotlib.gridspec as gridspec
from matplotlib.colorbar import Colorbar
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from IPython.display import Image
import matplotlib.pyplot as plt
from vtk.util import numpy_support
import vtk
import matplotlib.animation as animation
'''

# This block will create new directory for a burn steps and determine which DEM timesteps will be used for the simulation. All input stored as argparse to enable running from shell script.

# In[3]:


header=os.getcwd()

class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter,
                      argparse.RawDescriptionHelpFormatter):
    pass

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

parser=argparse.ArgumentParser(description=__doc__,
                               formatter_class=CustomFormatter)

##REQUIRED
parser.add_argument('stopstep', type=int,
                    help='steps of burn = ')
parser.add_argument('deltas', type=int,
                    help='how many steps of DEM for every days = ',default=200000)
parser.add_argument('initial', type=int,
                    help='initial steps of DEM')
parser.add_argument('batch', type=int,
                    help='Simulation batch including the skipped batch')
parser.add_argument('particle', type=int,
                    help='Particle simulated per batch')
parser.add_argument('cross_section',type=str,
                    help='Cross Section Data cross_sections.xml location')
parser.add_argument('--timeline', nargs="+", type=float,required=True,
                    help='list of {t(step-1),t(step),t(step+1)} separated by space')

##MODE
parser.add_argument('--control', type=bool,
                    help="Is it using control rods ?",default=False)
parser.add_argument('--burnup',type=bool,
                    help="run burn up mode ?",default=False)
parser.add_argument('--tally',type=bool,
                    help="run tally mode ?",default=False)
parser.add_argument('--post',default='Default',const='Default',
                    nargs='?',choices=['Post_Only','Run_Only','Default'],
                    help="post processing or other choices: Post_Only, Run_Only, Default")
parser.add_argument('--criticality', type=bool,
                    help="run criticality only", default=False)

##AUXILIARY (POWER & CONTROL PERC.)
parser.add_argument('--skipped', type=int,
                    help='Skipped Cycles',default=10)
parser.add_argument('--Burn_power', type=float,
                    help='Power in burn code in MW',default=10)
parser.add_argument('--controlstep', type=float, 
                    help="control rods position step 0-9 ",default=0)
#args = parser.parse_args()

args=parser.parse_args(args=(['17','10000','8600000','210','5000',
                              '/media/feryantama/Heeiya/openmc/TEST/data/lib80x_hdf5/cross_sections.xml',
                              '--timeline']+
                             [str(x) for x in range(0,35,7)]+
                             [str(x) for x in range(38,78,10)]+
                             [str(x) for x in range(98,378,30)]+
                             ['--burnup','True',
                              '--tally','True',
                              '--post','Post_Only']))

stp=args.stopstep
deltas=args.deltas
initial=args.initial

if args.burnup==True and args.tally==True:
    if not os.path.isdir('step_'+str(stp)+'-DEPLETE'):
        os.makedirs('step_'+str(stp)+'-DEPLETE',mode=0o777)
    os.chdir(header+'/step_'+str(stp)+'-DEPLETE')

if args.tally==True and args.control==False and args.criticality==False and args.burnup==False:
    if not os.path.isdir('step_'+str(stp)+'-TALLY'):
        os.makedirs('step_'+str(stp)+'-TALLY',mode=0o777)
    os.chdir(header+'/step_'+str(stp)+'-TALLY')

if args.tally==True and args.control==True:
    if not os.path.isdir('step_'+str(int(args.controlstep))+'-CONTROL'):
        os.makedirs('step_'+str(int(args.controlstep))+'-CONTROL',mode=0o777)
    os.chdir(header+'/step_'+str(int(args.controlstep))+'-CONTROL')
    
if args.tally==True and args.criticality==True:
    #if not os.path.isdir(header+'step_'+str(initial+(deltas*stp))+'-CRITICALITY'):
    #    os.makedirs('step_'+str(initial+(deltas*stp))+'-CRITICALITY',mode=0o777)
    os.chdir(header+'/step_'+str(initial+(deltas*stp))+'-CRITICALITY')
    
#os.environ['OPENMC_CROSS_SECTIONS']='/home/feryantama/Desktop/HTR2020/data/lib80x_hdf5/cross_sections.xml'
os.environ['OPENMC_CROSS_SECTIONS']=args.cross_section
chain = openmc.deplete.Chain.from_xml(header+"/chain_casl_pwr.xml")

if args.post=='Post_Only' or args.post=='Default':
    from scipy.interpolate import interp2d
    import matplotlib.gridspec as gridspec
    from matplotlib.colorbar import Colorbar
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    from IPython.display import Image
    import matplotlib.pyplot as plt
    from vtk.util import numpy_support
    import vtk
    import matplotlib.animation as animation



# ## 2. Function or Definition

# ### - Polar Angle Calculation
# This function used to determine $\theta$ (polar angle) of a cartesian coordinate relative to the centerline (0,0).

# In[4]:


def polar_angle(x,y):
    theta=math.atan(y/x)
    if x>=0 and y>=0:
        theta=theta
    if x<0 and y>=0:
        theta=theta+math.pi
    if x<0 and y<0:
        theta=theta+math.pi
    if x>=0 and y<0:
        theta=theta+math.pi*2
    return theta    


# ### - Make Pebble Dataframe from .vtk
# This function will read .vtk dump from LIGGGHTS or LAMMPS stored in the DEM_post directory under the working directory. Since .vtk dump file used cartesian coordinate to determine the center of a pebble, this function will also convert the coordinate into cylindrical coordinate ($\theta, r, z$) utilizing (**def polar_angle**) mentioned above. The pebble center coordinate will be stored as dataframe.

# In[5]:


def make_pebbledf(filepath):
    point_position=1e+20
    endpoint_position=2e+20
    id_position=1e+20
    endid_position=2e+20
    type_position=1e+20
    endtype_position=2e+20

    pointList=list()
    idList=list()
    typeList=list()
    
    with open(filepath) as fp:
        cnt=1
        for line in fp:
            for part in line.split():
                if "POINTS" in part:
                    point_data=line.replace('POINTS ','').replace(' float\n','')
                    point_data=int(point_data)
                    point_position=cnt
                    endpoint_position=math.ceil(point_position+point_data/3)
                elif "id" in part:
                    id_data=line.replace('id 1 ','').replace(' int\n','')
                    id_data=int(id_data)
                    id_position=cnt
                    endid_position=math.ceil(id_position+id_data/9)
                elif "type" in part:
                    type_data=line.replace('type 1 ','').replace(' int\n','')
                    type_data=int(type_data)
                    type_position=cnt
                    endtype_position=math.ceil(type_position+type_data/9)
            
            if (cnt>point_position and cnt<endpoint_position+1):
                pointList.extend(line.replace(' \n','').split(' '))
            elif (cnt>id_position and cnt<endid_position+1):
                idList.extend(line.replace(' \n','').split(' '))
            elif (cnt>type_position and cnt<endtype_position+1):
                typeList.extend(line.replace(' \n','').split(' '))
            cnt+=1

    pointarr=np.zeros([point_data,3])
    idarr=np.zeros([point_data])
    typearr=np.zeros([point_data])
    rho=np.zeros([point_data])
    theta=np.zeros([point_data])

    cnt=0
    for i in range (0,point_data):
        pointarr[i,0]=float(pointList[cnt*3+0])*100
        pointarr[i,1]=float(pointList[cnt*3+1])*100
        pointarr[i,2]=float(pointList[cnt*3+2])*100
        rho[i]=math.sqrt((pointarr[i,0]**2)+(pointarr[i,1]**2))
        theta[i]=(polar_angle(pointarr[i,0],pointarr[i,1]))
        typearr[i]=int(typeList[i])
        idarr[i]=int(idList[i])
        cnt+=1
    
    datasets=np.column_stack((pointarr,typearr,rho,theta))  
    Pebbledf=pd.DataFrame(data=datasets,index=idarr,columns=['x','y','z','type','rho','theta'])
    return(Pebbledf)


# ### - Grouping the Pebble According to its center cylindrical coordinate
# To simplify the material tracking algorithm (*explained later*), the pebble should be spatially grouped. This definition allows the user to group the pebble based on its axial ($z$) and radial ($r$) position. there are **ax_size** axial segment and **rd_size** radial segment thats makes up **ax_size X rd_size** ammount of group. as the pebble increase after timestep 0, the newly added pebble will grouped only by its radial position into **rd_size** group. 
# 
# overall there are:
# 
# **ax_size $\times$ rd_size + (stp $\times$ rd_size)** *(group of pebble)*
# 
# where **stp** is burn up step.
# Every pebble which has been labeled or grouped in past burn up step will retain its group even the position has changed. Therefore, this function will record the pebble grouping for each pebble number to ASCII file.

# In[15]:


def segmentation(Pebbledf,ax_size,rd_size,header,stp):
    print("========Start Segmentation========")
    if stp==0:
        count_list=np.zeros(len(Pebbledf))
        highest=0; lowest=0
        for i in range (0,len(Pebbledf)):
            if Pebbledf.iloc[i][3]==1 and Pebbledf.iloc[i][2]>=highest:
                highest=Pebbledf.iloc[i][2]

        dlt=(highest-lowest)/5
        axiallist=np.linspace(lowest+dlt,highest-dlt,ax_size-1)
        axiallist=np.insert(axiallist,0,-600,axis=0)
        axiallist=np.insert(axiallist,len(axiallist),600,axis=0)
        radiallist=np.linspace(25,90,rd_size)
        radiallist=np.insert(radiallist,0,0,axis=0)
        count=1
        for i in range (0,len(axiallist)-1):
            for j in range (0,len(radiallist)-1):
                for k in range (0,len(Pebbledf)):
                    if (Pebbledf.iloc[k][3]!=2 and 
                        Pebbledf.iloc[k][2]>axiallist[i] and
                        Pebbledf.iloc[k][2]<=axiallist[i+1] and
                        Pebbledf.iloc[k][4]>radiallist[j] and
                        Pebbledf.iloc[k][4]<=radiallist[j+1]):
                        count_list[k]=count
                    
                count+=1
        labdf=pd.DataFrame(data=count_list,columns=['label'],index=Pebbledf.index)
        labdf.to_csv('Prop.Label',index=True,header=True, sep='\t')
        print("========Finish Segmentation=======")
        Pebbledf=pd.concat([Pebbledf,labdf],axis=1)

    if stp!=0:
        labdf=pd.read_csv(header+'/step_'+str(stp-1)+'-DEPLETE'+'/Prop.Label',sep='\t',index_col='Unnamed: 0'	)
        print("=========Skip Segmentation========")
        Pebbledf=pd.concat([Pebbledf,labdf],axis=1)
        Pebbledf=Pebbledf.replace(np.nan, 'c', regex=True)
        count_list=list(Pebbledf.iloc[:,6])
        for i in range (0,len(Pebbledf)):
            if Pebbledf.iloc[i][6]=='c':
                if Pebbledf.iloc[i][3]==2:
                    count_list[i]=0
                if Pebbledf.iloc[i][3]==1:
                    radiallist=np.linspace(25,90,rd_size)
                    radiallist=np.insert(radiallist,0,0,axis=0)
                    for j in range(0,len(radiallist)-1):
                        if (Pebbledf.iloc[i][4]>radiallist[j] and 
                            Pebbledf.iloc[i][4]<=radiallist[j+1]):
                            count_list[i]=max(labdf['label'])+j+1
        labdf_new=pd.DataFrame(data=count_list,columns=['label'],index=Pebbledf.index)
        labdf_new.to_csv('Prop.Label',index=True,header=True, sep='\t')
        Pebbledf=Pebbledf.drop(columns='label')
        Pebbledf=pd.concat([Pebbledf,labdf_new],axis=1)
    depleted_mat=int(max(Pebbledf['label']))
    
    Segment_vol=np.zeros(depleted_mat)

    for j in range (0,len(Pebbledf)):
        for k in range(1,depleted_mat+1):
            if (Pebbledf.iloc[j][6]==k and Pebbledf.iloc[j][2]>=-258.182):
                Segment_vol[k-1]+=1

    Segment_vol*=8335*(4/3)*math.pi*(0.025**3)
    return(Pebbledf,depleted_mat,Segment_vol)


# ### - Sphere Parametric Equation
# This function define the radius of a point from the center point of sphere.

# In[7]:


def sphere(array,center):
    r=math.sqrt(((array[0]-center[0])**2)+((array[1]-center[1])**2)+((array[2]-center[2])**2))
    return (r)


# ### - Make TRISO dataframe
# Definition below will create random point inside the Pebble fuel zone and check if its overlaps with other random point. in total there are 8335 TRISO center point generated. Those center coordinate will be stored as dataframe.

# In[8]:


def make_TRISO():
    seed(123457)
    cnt=0
    rd=list()
    r=2.5-(0.0915/2)
    rm=0.2/2
    center=[0,0,0]
    trial=0
    print('======Start Randoming TRISO=======')
    while (cnt<8335):
        x=float((random()*r*2)-r)
        y=float((random()*r*2)-r)
        z=float((random()*r*2)-r)
        array=[x,y,z]
        rad=sphere(array,center)
        test=0
        if ((rad**2)<(r**2)):
            if cnt>0:
                for i in range (0,len(rd)):
                    if (sphere(array,rd[i])**2)>(rm**2):
                        test+=1
                        if test==len(rd):
                            rd.append(array)             
                            cnt+=1
                trial+=1
#                print(str(test)+'\t'+str(trial))
            else:
                cnt+=1
                rd.append(array)

    Trisodf=pd.DataFrame(rd,columns=['x','y','z'])
    return(Trisodf)


# ### - Kernel Material Definition
# 

# In[9]:


def fuel_material(stp,filename,header,depleted_mat,rd_size):
    Fueldf=pd.read_csv(header+'/Prop.fuel',sep='\t')
    Fuellist=list()
    #========================< FISSILE ELEMENT >==========================
    nuclide_correction=[['Tc99_m1','Tc100','Rh102_m1','Rh103_m1','Rh105_m1','Rh106','Rh106_m1','Ag109_m1',
                         'Cd115','Te127','Xe137','Pr144','Nd151','Sm155','Pa234','Np240','Np240_m1'],
                        ['Tc99','Ru100','Rh102','Rh103','Rh105','Pd106','Pd106','Ag109','Cd114','I127',
                         'Cs137','Nd144','Pm151','Eu155','U234','Pu240','Pu240']]
    if stp==0:
        for i in range (0,len(Fueldf)):
            if Fueldf.iloc[i][0]=='Kernel':
                fuel=list()
                for j in range(0,depleted_mat):
                    FulMat=openmc.Material(name=str(Fueldf.iloc[i][0])+'_'+str(j))
                    FulMat.set_density('g/cm3', Fueldf.iloc[i][1])
                    if Fueldf.iloc[i][2]>0:
                        FulMat.add_nuclide('U235',Fueldf.iloc[i][2]*0.1718,'ao')
                        FulMat.add_nuclide('U238',Fueldf.iloc[i][2]*0.8282,'ao')
                    if Fueldf.iloc[i][3]>0:
                        FulMat.add_nuclide('O16',Fueldf.iloc[i][3],'ao')
                    if Fueldf.iloc[i][4]>0:
                        FulMat.add_element('B',Fueldf.iloc[i][4],'ao')
                    if Fueldf.iloc[i][5]>0:
                        FulMat.add_element('C',Fueldf.iloc[i][5],'ao')
                    if Fueldf.iloc[i][6]>0:
                        FulMat.add_element('Si',Fueldf.iloc[i][6],'ao')
                    fuel.append(FulMat)
                Fuellist.append(fuel)
    if stp!=0:
        results=openmc.deplete.ResultsList.from_hdf5(filename+'/depletion_results.h5')
        nuclide=list(chain.nuclide_dict.keys())
        time, k=results.get_eigenvalue()
        time/=(24*3600)
        fuel=list() 
        matlistdf=pd.read_csv(filename+'/Fuel.Prop'+str(stp-1),index_col='Unnamed: 0')
        for i in range(0,depleted_mat):
            if i<(depleted_mat-rd_size):
                fuel.append(openmc.Material(name='Kernel_'+str(i)))
                fuel[i].set_density('atom/b-cm',matlistdf.sum(axis=1)[(2*i)+1]-matlistdf['time'].iloc[(2*i)+1]-matlistdf['material'].iloc[(2*i)+1])
                for n in range(0,len(nuclide)):
                    fuel[i].add_nuclide(nuclide[n],matlistdf[nuclide[n]].iloc[(2*i)+1],'ao')
                for j in range(0,len(nuclide_correction[0])):
                    ao=0
                    for nuc in fuel[i].nuclides:
                        if nuc[0]==nuclide_correction[0][j]:
                            ao+=nuc[1]
                        if nuc[0]==nuclide_correction[1][j]:
                            ao+=nuc[1]
                    fuel[i].remove_nuclide(nuclide_correction[0][j])
                    fuel[i].remove_nuclide(nuclide_correction[1][j])
                    fuel[i].add_nuclide(nuclide_correction[1][j],ao,'ao')
            if i>=depleted_mat-rd_size:
                for k in range (0,len(Fueldf)):
                    if Fueldf.iloc[k][0]=='Kernel':
                        FulMat=openmc.Material(name=str(Fueldf.iloc[k][0])+'_'+str(i))
                        FulMat.set_density('g/cm3', Fueldf.iloc[k][1])
                        if Fueldf.iloc[k][2]>0:
                            FulMat.add_nuclide('U235',Fueldf.iloc[k][2]*0.1718,'ao')
                            FulMat.add_nuclide('U238',Fueldf.iloc[k][2]*0.8282,'ao')
                        if Fueldf.iloc[k][3]>0:
                            FulMat.add_nuclide('O16',Fueldf.iloc[k][3],'ao')
                        if Fueldf.iloc[k][4]>0:
                            FulMat.add_element('B',Fueldf.iloc[k][4],'ao')
                        if Fueldf.iloc[k][5]>0:
                            FulMat.add_element('C',Fueldf.iloc[k][5],'ao')
                        if Fueldf.iloc[k][6]>0:
                            FulMat.add_element('Si',Fueldf.iloc[k][6],'ao')
                        fuel.append(FulMat)
        Fuellist.append(fuel)
    #=========================< TRISO ELEMENT >===========================
    for i in range(0,len(Fueldf)):
        if Fueldf.iloc[i][0]!='Kernel':
            FulMat=openmc.Material(name=str(Fueldf.iloc[i][0]))
            FulMat.set_density('g/cm3', Fueldf.iloc[i][1])
            if Fueldf.iloc[i][2]>0:
                FulMat.add_nuclide('U235',Fueldf.iloc[i][2]*0.1718,'ao')
                FulMat.add_nuclide('U238',Fueldf.iloc[i][2]*0.8282,'ao')
            if Fueldf.iloc[i][3]>0:
                FulMat.add_nuclide('O16',Fueldf.iloc[i][3],'ao')
            if Fueldf.iloc[i][4]>0:
                FulMat.add_element('B',Fueldf.iloc[i][4],'ao')
            if Fueldf.iloc[i][5]>0:
                FulMat.add_element('C',Fueldf.iloc[i][5],'ao')
            if Fueldf.iloc[i][6]>0:
                FulMat.add_element('Si',Fueldf.iloc[i][6],'ao')
            Fuellist.append(FulMat)
    return(Fuellist)


# ## 3. Material Definition

# ### - Pebble Material
# Whole line below, will call all the function above to create TRISO and Pebble position dataframe with its material detail. This simulation will used 5 axial segment and 3 radial segment for grouping the pebble. The physical representation of how the grouping works shown below this block.

# In[ ]:


Pebbledf,depleted_mat,Segment_vol=segmentation(
    make_pebbledf(header+'/DEM_post/pebble_'+str(int(initial+(deltas*args.timeline[stp])+stp*100000))+'.vtk'),
    5,3,header,stp)

kernel,bufer,ipyc,opyc,sic,matrix,dummy=fuel_material(stp, header+'/step_'+str(stp-1)+'-DEPLETE', header,depleted_mat,3)
'''
for i in range(0,len(kernel)):
    if stp==0:
        kernel[i].add_s_alpha_beta('c_O_in_UO2')
        kernel[i].add_s_alpha_beta('c_U_in_UO2')

sic.add_s_alpha_beta('c_Si_in_SiC')
sic.add_s_alpha_beta('c_C_in_SiC')

opyc.add_s_alpha_beta('c_Graphite')
ipyc.add_s_alpha_beta('c_Graphite')
matrix.add_s_alpha_beta('c_Graphite')
dummy.add_s_alpha_beta('c_Graphite')
bufer.add_s_alpha_beta('c_Graphite')
'''
materials_file=openmc.Materials([bufer,opyc,ipyc,sic,matrix,dummy])
for i in range (0,len(kernel)):
    kernel[i].volume=Segment_vol[i]
    materials_file.append(kernel[i])


# ### - Control Rod Material
# This line only work when the *control* argument is **True**. This define the control rod material for control rod worth experiment. Detail of the control rod worth benchmarking explained in this works below.
# 
# *link* : http://www-pub.iaea.org/MTCD/publications/PDF/te_1382_web/TE_1382_Part1.pdf

# In[ ]:


if args.control==True:
    Control_rod=openmc.Material(name='Control_rod')
    Control_rod.add_element('B',0.8,'ao')
    Control_rod.add_element('C',0.2,'ao')
    Control_rod.set_density('g/cm3', 1.7)
    
    Spacing=openmc.Material(name='Spacing')
    Spacing.add_element('Cr',0.18,'wo')
    Spacing.add_element('Fe',0.681,'wo')
    Spacing.add_element('Ni',0.1,'wo')
    Spacing.add_element('Si',0.01,'wo')
    Spacing.add_element('Mn',0.02,'wo')
    Spacing.add_element('C',0.001,'wo')
    Spacing.add_element('Ti',0.008,'wo')
    Spacing.set_density('g/cm3', 7.9)
    
    Caps=openmc.Material(name='Caps_n_Joints')
    Caps.add_element('Fe',1,'ao')
    Caps.set_density('atom/b-cm',0.04)
    
    materials_file.extend([Control_rod,Spacing,Caps])


# ### - Reflector Material
# THis line will read the reflector dataframe from **Prop.Reflector** file under the working directory. this dataframe contain the nuclide composition for each reflector block in HTR-10 obtained from NEA HTR-10 initial criticality calculation.
# 
# *link* : https://catatanstudi.files.wordpress.com/2009/10/2006-evaluation-of-the-initial-critical-configuration-of-the-htr-10-pbr_nea-oecd.pdf

# In[ ]:


#Reflector Properties
Reflectordf=pd.read_csv(header+'/Prop.Reflector',sep='\t')
Reflectorlist=list()
for i in range (0,len(Reflectordf)):
    RefMat=openmc.Material(name='Region'+str(int(Reflectordf.iloc[i][0])))
    RefMat.set_density('atom/b-cm', Reflectordf.iloc[i][1])
    if Reflectordf.iloc[i][2]>0:
        RefMat.add_element('N',Reflectordf.iloc[i][2],'ao')
    if Reflectordf.iloc[i][3]>0 and Reflectordf.iloc[i][6]>0:
        RefMat.add_element('O',Reflectordf.iloc[i][3]+Reflectordf.iloc[i][6],'ao')
    if Reflectordf.iloc[i][4]>0:
        RefMat.add_element('Ar',Reflectordf.iloc[i][4],'ao')
    if Reflectordf.iloc[i][5]>0:
        RefMat.add_element('H',Reflectordf.iloc[i][5],'ao')
    if Reflectordf.iloc[i][7]>0:
        RefMat.add_element('C',Reflectordf.iloc[i][7],'ao')
    if Reflectordf.iloc[i][8]>0:
        RefMat.add_element('B',Reflectordf.iloc[i][8],'ao')
    #RefMat.add_s_alpha_beta('c_Graphite')
    Reflectorlist.append(RefMat)
    Reflectorlist[i]
for i in range (0,len(Reflectorlist)):
    materials_file.append(Reflectorlist[i])
if args.post=='Default' or args.post=='Run_Only':
    materials_file.export_to_xml()


# ## 4. Geometry Building

# ### - TRISO Layer Definition
# TRISO consist of fissile meterialin the center and 4 different layer around it. In this calculation there are 
# 
# **ax_size $\times$ rd_size + (stp $\times$ rd_size)** $= 15+(stp\times3) $ (*TRISO Types*)
# 
# Representing each fuel pebble group mentioned before. TRISO sphere including all the layer defined as a universe.
# 

# In[ ]:

'''
#======================================================================
#------------------------------TRISO Definition------------------------
#======================================================================
'''
TRISOsurf=list()
TRISOmatlist=[kernel,bufer,ipyc,sic,opyc]
TRISOcelllist=list()
TRISOunivlist=list()
spherenum=[0.025,0.034,0.038,0.0415,0.0455]
for i in range(0,len(spherenum)):
    TRISOsurf.append(openmc.Sphere(name='Sph_'+str(i),r=spherenum[i]))

for j in range(0,depleted_mat):
    TRISOcell=list()
    for i in range(0,len(spherenum)):
        if i==0:
            TRISOcell.append(openmc.Cell(fill=TRISOmatlist[i][j],region=-TRISOsurf[i]))
        elif i==len(spherenum)-1:
            TRISOcell.append(openmc.Cell(fill=TRISOmatlist[i],region=+TRISOsurf[i-1]))
        else:
            TRISOcell.append(openmc.Cell(fill=TRISOmatlist[i],region=-TRISOsurf[i] & +TRISOsurf[i-1]))
    TRISOcelllist.append(TRISOcell)
    TRISOunivlist.append(openmc.Universe(cells=TRISOcell))

# ### - Pebble Definition
# There are two types of pebble in HTR-10 reactor; fuel and dummy. Fuel pebble consist of two region there are fuelzone (where the TRISO element scattered) and matrix (outer layer). This part will create the fuelzone universe which contain TRISO element scattered in certain coordinate from **TRISOdf**. Each fuelzone universe has one type of TRISO which will represent the pebble group. All the fuelzone stored as a Pebble universe along with the matrix.

# In[ ]:

'''
#======================================================================
#-----------------------------Pebble Definition------------------------
#======================================================================
'''
print('=============='+str(int(initial+(deltas*args.timeline[stp])+stp*100000))+'=============')
Fuel_zone=openmc.Sphere(r=2.5)
Graphite_zonecelllist=list()

Fuel_elementreg=-Fuel_zone
if stp==0:
    Trisodf=make_TRISO()
    Trisodf.to_csv('Trisodf.csv',index=True,header=True,sep='\t')
if stp!=0:
    Trisodf=pd.read_csv(header+'/step_'+str(0)+'-DEPLETE'+'/Trisodf.csv',sep='\t',index_col='Unnamed: 0')

Pebbleunivlist=list()
Fuel_elementcelllist=list()
Fuel_elementsurf=list()

for i in range (0,len(Trisodf)):
    Fuel_elementsurf.append(openmc.Sphere(r=0.0455,x0=Trisodf.loc[i]['x'],y0=Trisodf.loc[i]['y'],z0=Trisodf.loc[i]['z']))
    Fuel_elementreg=Fuel_elementreg & +Fuel_elementsurf[i]
    
for j in range (0,depleted_mat):
    Pebbleuniv=openmc.Universe()
    Graphite_zonecelllist.append(openmc.Cell(name='Graphite_Zone',fill=matrix,region= +Fuel_zone))
    Pebbleuniv.add_cell(Graphite_zonecelllist[j])
    Fuel_elementcell=list()
    for i in range (0,len(Trisodf)):
        Fuel_elementcell.append(openmc.Cell(fill=TRISOunivlist[j],region=-Fuel_elementsurf[i]))
        Fuel_elementcell[i].translation=(Trisodf.loc[i]['x'], Trisodf.loc[i]['y'], Trisodf.loc[i]['z'])
        Pebbleuniv.add_cell(Fuel_elementcell[i])

    Fuel_elementcell.append(openmc.Cell(name='Fuel_Zone'+str(j),fill=matrix,region=Fuel_elementreg))
    Pebbleuniv.add_cell(Fuel_elementcell[-1])

    Pebbleunivlist.append(Pebbleuniv)
    Fuel_elementcelllist.append(Fuel_elementcell)

# ### - Surface Definition
# This part will create the whole reactor surface. The bottom surface of bottom reflector will move downward 60cm for every steps after burnstep 0 to represent $1cm/day$ pebble movement to discharge.

# In[ ]:


'''
======================================================================
----------------------------Surface Definition------------------------
======================================================================
'''
levellist=list()
levelnum=[610.0,540.0,510.0,495.0,465.0,450.0,430.0,402.0,388.764,351.818,130.0,114.7,105.0,95.0,40.0,0]
for i in range (0,len(levelnum)):
    level=openmc.ZPlane(name='level_'+str(i),z0=351.818-levelnum[i])
    levellist.append(level)

radiallist=list()
radialnum=[25.0,41.75,70.75,90.0,95.6,108.6,140.6,148.6,167.793,190.0]
for i in range (0,len(radialnum)):
    radial=openmc.ZCylinder(name='cyl_'+str(i),x0=0.0,y0=0.0,r=radialnum[i])
    radiallist.append(radial)

cone_center=-(90.0/((90.0-25.0)/(388.764-351.818)))
cone=openmc.ZCone(x0=0.0,y0=0.0,z0=cone_center,r2=(90.0/-(cone_center))**2)

LS=round((min(Pebbledf['z'])-3),3)
if LS<=-258.182:
    LS=-258.182
low_surf=openmc.ZPlane(name='low_surf',z0=LS)
low_surf7=openmc.ZPlane(name='cell7low',z0=LS-(540-495))
low_surf81=openmc.ZPlane(name='cell81low',z0=LS-(610-495))
print('========Print to XML geoms========')
print('=============='+str(round(LS,3))+'============')

# ### - Core Definition
# The core consist of pebble and coolant. Each pebble modeled according to its group and its coordinate. The fuel pebble filled with **pebbleuniv** and dummy filled with graphite. Outside all the pebble modeled as coolant. The core modeled as universe.

# In[ ]:

'''
#======================================================================
#-------------------------------Core Definition------------------------
#======================================================================
'''

Pebblereg=-radiallist[radialnum.index(190)]
Pebblesurflist=list()
Pebblecelllist=list()
coreuniv=openmc.Universe()
ct=0
Fuel_tally_Filter=[]
Fuel_tally_Indexs=[]

for i in range (0,len(Pebbledf)):
    if Pebbledf.iloc[i][2]>=(-258.182-3):
        Pebblesurflist.append(openmc.Sphere(name='Pebble_'+str(int(Pebbledf.index[i])),
                                            r=3,x0=Pebbledf.iloc[i][0],y0=Pebbledf.iloc[i][1],z0=Pebbledf.iloc[i][2]))
        Pebblereg=Pebblereg & +Pebblesurflist[ct]
        if Pebbledf.iloc[i][3]==1:
            Pebblecelllist.append(openmc.Cell(name='Fuel_'+str(int(Pebbledf.index[i])),
                                              fill=Pebbleunivlist[int(Pebbledf.iloc[i][6]-1)], 
                                              region= -Pebblesurflist[ct]))
            Pebblecelllist[ct].translation=(Pebbledf.iloc[i][0], Pebbledf.iloc[i][1], Pebbledf.iloc[i][2])
            Fuel_tally_Filter.append(Pebblecelllist[ct])
            Fuel_tally_Indexs.append([Pebblecelllist[ct].id,Pebbledf.index[i],Pebbledf['label'].iloc[i]])
        else:
            Pebblecelllist.append(openmc.Cell(name='Dummy_'+str(int(Pebbledf.index[i])),
                                              fill=dummy,region= -Pebblesurflist[ct]))
        coreuniv.add_cell(Pebblecelllist[ct])
        ct+=1
    
Coolantcell=openmc.Cell(name='coolant',fill=Reflectorlist[5],region=Pebblereg) 
coreuniv.add_cell(Coolantcell)
coreboundreg=((-radiallist[radialnum.index(90.0)] & -levellist[levelnum.index(130.0)] & +levellist[levelnum.index(351.818)]) | 
              (-cone & -levellist[levelnum.index(351.818)] & +levellist[levelnum.index(388.764)]) |
              (-radiallist[radialnum.index(25.0)] & -levellist[levelnum.index(388.764)] & +low_surf ))

# ### - Reflector Definition
# This part will create reflector cell, boring (KLAK, Hot duct and Control) and Control Rod based on the **Prop.boring** file under the working directory. **Prop.boring** contain boring equation coefficient for each different boring. The KLAK channel modelled in oval shape and round shape same as the high fidelity model in HTR-10 benchmark by NEA and IAEA TecDoc. If the **args.control** is *True*, the Control Rod will be modeled. The Core filled with core universe in the line above.

# In[ ]:

'''
#======================================================================
#----------------------------Reflector Definition----------------------
#======================================================================
'''
Reactor_univ=openmc.Universe()
'''
#///////////////////////////////CORE////////////////////////////////////
'''
coreboundcell=openmc.Cell(fill=coreuniv,region=coreboundreg)
Reactor_univ.add_cell(coreboundcell)

'''
#////////////////////////////CONTROL ROD////////////////////////////////
'''
if args.control==True:
    Z_control=[0,4.5,53.2,56.8,105.5,109.1,157.8,161.4,210.1,213.7,262.4,264.7]
    C_control=[2.75,2.95,3,5.25,5.3,5.5,6.499]
    M_control=[Spacing,Reflectorlist[5],
               Control_rod,Reflectorlist[5],
               Spacing,Reflectorlist[5]]

    delt=(394.2-171.2)/9
    stepcont=args.controlstep
    Z_base=171.2+stepcont*delt

    control_surf=list()
    contc=0
    for i in range(0,len(Z_control)):
        control_surf.append(openmc.ZPlane(z0=round(351.818+Z_control[-1]-(Z_control[i]+Z_base),3)))
        contc+=1

    for i in range(0,len(C_control)-1):
        control_surf.append(openmc.ZCylinder(x0=0.0,y0=0.0,r=C_control[i]))
        contc+=1

    control_cell=list()
    for i in range(0,5):
        for j in range(0,len(C_control)-1):
            if j!=len(C_control)-2:
                reg_contr=+control_surf[len(Z_control)+j] & -control_surf[1+len(Z_control)+j] & -control_surf[1+i*2] & +control_surf[2+i*2]
            if j==len(C_control)-2:
                reg_contr=+control_surf[len(Z_control)+j] & -control_surf[1+i*2] & +control_surf[2+i*2]
            control_cell.append(openmc.Cell(fill=M_control[j],region=reg_contr))

  #=====================================================================
    for i in range(0,6):
        for j in range(0,2):
            if j==0:
                control_cell.append(openmc.Cell(fill=Caps,region=(+control_surf[len(Z_control)] & 
                                                                  -control_surf[len(Z_control)+5] &
                                                                  -control_surf[i*2] & 
                                                                  +control_surf[1+i*2])))
            if j!=0:
                control_cell.append(openmc.Cell(fill=Reflectorlist[5],region=(+control_surf[len(Z_control)+5] & 
                                                                              -control_surf[i*2] & 
                                                                              +control_surf[1+i*2])))
  
  #=====================================================================
    control_cell.append(openmc.Cell(fill=Reflectorlist[5],region=(-control_surf[len(Z_control)] &
                                                                  -control_surf[0] &
                                                                  +control_surf[len(Z_control)-1])))
    control_cell.append(openmc.Cell(fill=Reflectorlist[5],region=(+control_surf[0] | 
                                                                  -control_surf[len(Z_control)-1])))
    Control_univ=openmc.Universe()
    for cell in control_cell:
        Control_univ.add_cell(cell)

'''
#/////////////////////////////BORING////////////////////////////////////
'''

Boringdf=pd.read_csv(header+'/Prop.boring',sep='\t')
Boring_plane=[0.0,105.0,130.0,388.764,610.0,450.0]

Boring_surf=list()


boringreg=-radiallist[radialnum.index(190.0)]
Boring_cell=list()
for i in range (0,len(Boringdf)):
    Boringcyl=openmc.ZCylinder(name='boring_'+str(i),x0=Boringdf.iloc[i][2],y0=Boringdf.iloc[i][3],r=(Boringdf.iloc[i][1]/2)-0.001)
    Boring_surf.append(Boringcyl)    
    if Boringdf.iloc[i][0]=='control' and args.control==True:
        Boringcell=openmc.Cell(fill=Control_univ,region=-Boringcyl & 
                               -levellist[levelnum.index(Boringdf.iloc[i][4])]&
                               +levellist[levelnum.index(Boringdf.iloc[i][5])])
        Boringcell.translation=(Boringdf.iloc[i][2], Boringdf.loc[i][3], 0)
        Boring_cell.append(Boringcell)
    else:
        Boringcell=openmc.Cell(fill=Reflectorlist[5],region=-Boringcyl & 
                               -levellist[levelnum.index(Boringdf.iloc[i][4])]&
                               +levellist[levelnum.index(Boringdf.iloc[i][5])])
        Boring_cell.append(Boringcell)
    boringreg=boringreg & (+Boringcyl |
                           +levellist[levelnum.index(Boringdf.iloc[i][4])]|
                           -levellist[levelnum.index(Boringdf.iloc[i][5])])

Boring_surf.append(openmc.XCylinder(y0=0,z0=351.818-480,r=14.999))
Boring_surf.append(openmc.XPlane(x0=90))
Boring_surf.append(openmc.XPlane(x0=190))
Boring_cell.append(openmc.Cell(fill=Reflectorlist[5],name='Hotduct',region=(-Boring_surf[-3] & -Boring_surf[-1] & +Boring_surf[-2]) & +Boring_surf[0]))

boringreg=boringreg & (+Boring_surf[-3] | +Boring_surf[-1] | -Boring_surf[-2])

for i in range (0,len(Boring_cell)):
    Reactor_univ.add_cell(Boring_cell[i])

'''
#/////////////////////////////OVAL KLAK///////////////////////////////////
'''

KLAKdf=pd.read_csv(header+'/Prop.KLAK',sep='\t')
klakcirc1=list()
klakcirc2=list()
klakplane1=list()
klakplane2=list()
klakwall1=list()
klakwall2=list()
KLAKlist=list()
for i in range (0,len(KLAKdf)):
    klkcirc1=openmc.ZCylinder(x0=KLAKdf.iloc[i][9],y0=KLAKdf.iloc[i][10],r=KLAKdf.iloc[i][11])
    klkcirc2=openmc.ZCylinder(x0=KLAKdf.iloc[i][12],y0=KLAKdf.iloc[i][13],r=KLAKdf.iloc[i][14])
    klkplane1=openmc.Plane(a=KLAKdf.iloc[i][1],b=KLAKdf.iloc[i][2],c=KLAKdf.iloc[i][3],d=KLAKdf.iloc[i][4])
    klkplane2=openmc.Plane(a=KLAKdf.iloc[i][5],b=KLAKdf.iloc[i][6],c=KLAKdf.iloc[i][7],d=KLAKdf.iloc[i][8]) 
    klkwall1=openmc.Plane(a=KLAKdf.iloc[i][15],b=KLAKdf.iloc[i][16],c=KLAKdf.iloc[i][17],d=KLAKdf.iloc[i][18])
    klkwall2=openmc.Plane(a=KLAKdf.iloc[i][19],b=KLAKdf.iloc[i][20],c=KLAKdf.iloc[i][21],d=KLAKdf.iloc[i][22])
    klakcirc1.append(klkcirc1)
    klakcirc2.append(klkcirc2)
    klakplane1.append(klkplane1)
    klakplane2.append(klkplane2)
    klakwall1.append(klkwall1)
    klakwall2.append(klkwall2)
    KLKlist1=openmc.Cell(fill=Reflectorlist[5],
                         region=((-klakcirc1[i]| -klakcirc2[i]|
                                 (-klakplane1[i]& +klakplane2[i]& -klakwall1[i]& +klakwall2[i]))&
                                 (-levellist[levelnum.index(130.0)])&
                                 (+levellist[levelnum.index(388.764)])))
    boringreg=boringreg & (((+klakcirc1[i]& +klakwall1[i])|
                            (+klakcirc2[i]& -klakwall2[i])| +klakplane1[i]| -klakplane2[i])|
                            +levellist[levelnum.index(130.0)]|
                            -levellist[levelnum.index(388.764)])
    KLAKlist.append(KLKlist1)

for i in range (0,len(KLAKlist)):
    Reactor_univ.add_cell(KLAKlist[i])

'''
#/////////////////////////////REFLECTOR///////////////////////////////////
'''
Reflector_cell=list()

for i in range (0,len(Reflectordf)):
    if Reflectordf.iloc[i][0]==83:
        Reflreg=openmc.Cell(name='cell'+str(int(Reflectordf.iloc[i][0])),fill=Reflectorlist[i], region=boringreg & -levellist[levelnum.index(351.818)] & -radiallist[radialnum.index(90.0)] & +cone & +levellist[levelnum.index(388.764)] & +radiallist[radialnum.index(25.0)])
        Reflector_cell.append(Reflreg)
    if Reflectordf.iloc[i][0]==48:
        Reflreg=openmc.Cell(name='cell'+str(int(Reflectordf.iloc[i][0]))+'a',fill=Reflectorlist[i], region=boringreg & -levellist[levelnum.index(40)] & -radiallist[radialnum.index(167.793)] & +levellist[levelnum.index(388.764)] & +radiallist[radialnum.index(148.6)])
        Reflector_cell.append(Reflreg)
        Reflreg=openmc.Cell(name='cell'+str(int(Reflectordf.iloc[i][0]))+'b',fill=Reflectorlist[i], region=boringreg & -levellist[levelnum.index(40)] & -radiallist[radialnum.index(148.6)] & +levellist[levelnum.index(95)] & +radiallist[radialnum.index(108.6)])
        Reflector_cell.append(Reflreg)
    if Reflectordf.loc[i]['Left_Bound']==0 and Reflectordf.iloc[i][0]!=81 and Reflectordf.iloc[i][0]!=7 and Reflectordf.iloc[i][0]!=83 and Reflectordf.iloc[i][0]!=48 and Reflectordf.iloc[i][0]!=5:
        Reflreg=openmc.Cell(name='cell'+str(int(Reflectordf.iloc[i][0])),fill=Reflectorlist[i], region= boringreg & -levellist[levelnum.index(Reflectordf.loc[i]['Top'])] & -radiallist[radialnum.index(Reflectordf.loc[i]['Right_Bound'])] & +levellist[levelnum.index(Reflectordf.loc[i]['Bottom'])])
        Reflector_cell.append(Reflreg)
    if Reflectordf.loc[i]['Left_Bound']!=0 and Reflectordf.iloc[i][0]!=81 and Reflectordf.iloc[i][0]!=7 and Reflectordf.iloc[i][0]!=83 and Reflectordf.iloc[i][0]!=48 and Reflectordf.iloc[i][0]!=5:
        Reflreg=openmc.Cell(name='cell'+str(int(Reflectordf.iloc[i][0])),fill=Reflectorlist[i], region= boringreg & -levellist[levelnum.index(Reflectordf.loc[i]['Top'])] & -radiallist[radialnum.index(Reflectordf.loc[i]['Right_Bound'])] & +levellist[levelnum.index(Reflectordf.loc[i]['Bottom'])] & +radiallist[radialnum.index(Reflectordf.loc[i]['Left_Bound'])])
        Reflector_cell.append(Reflreg)
    if Reflectordf.iloc[i][0]==81:
        Reflreg=openmc.Cell(name='cell'+str(int(Reflectordf.iloc[i][0])),fill=Reflectorlist[i], region=(-low_surf7 & -radiallist[radialnum.index(Reflectordf.loc[i]['Right_Bound'])] & +low_surf81))
    if Reflectordf.iloc[i][0]==7:
        Reflreg=openmc.Cell(name='cell'+str(int(Reflectordf.iloc[i][0])),fill=Reflectorlist[i], region=(-low_surf & -radiallist[radialnum.index(Reflectordf.loc[i]['Right_Bound'])] & +low_surf7))
    Reflector_cell.append(Reflreg)

for i in range (0,len(Reflector_cell)):
    Reactor_univ.add_cell(Reflector_cell[i])


# ### - Finalize Geometry
# This part will plot the geometry to validate our line above. It also store the geometry of root cell (contain the whole model) to openmc geometry xml.

# In[ ]:



levellist[0].boundary_type='vacuum'
levellist[levelnum.index(0)].boundary_type='vacuum'
radiallist[radialnum.index(190)].boundary_type='vacuum'

root_cell=openmc.Cell(name='root_cell',fill=Reactor_univ,region=-radiallist[radialnum.index(190)] & -levellist[levelnum.index(0.0)] & +levellist[levelnum.index(610.0)])
root_universe=openmc.Universe(name='root_universe')
root_universe.add_cell(root_cell)

geometry = openmc.Geometry()
geometry.root_universe=root_universe
print('========Print to XML geoms========')
if args.post=='Default' or args.post=='Run_Only':
    geometry.export_to_xml()

# ## 5. Simulation Setup

# ### - Tally Setup
# Only available if **args.tally** is true. This part will perform multi group cross section evaluation for mesh tally of $\nu\Sigma_f^{delayed}$ and $\nu\Sigma_f$with 50 groups of neutron energy.This part also perform flux tally with continous cross section with mesh filter and energy filter. In addition, the calculation also evaluate the flux with legendre ($P_n$) polynomial interpolation equation for axial distribution and zernike ($Z^0_n$) polynomial equation for radial distribution.

# In[ ]:


'''
#----------------------------------------------------------------------
#==============================MGXS SET-UP=============================
#----------------------------------------------------------------------
'''
if args.tally==True:
    print('=========Start Set up MGXS========')

    mesh_res=3
    mesh = openmc.RegularMesh(mesh_id=1)
    mesh.dimension = [int(380/mesh_res), int(380/mesh_res), int(610/mesh_res)]
    mesh.lower_left = [-190,-190,351.818-610]
    mesh.upper_right = [190,190,351.818]
    mesh_filter = openmc.MeshFilter(mesh)
#=====================================================================
    tallies_file = openmc.Tallies()

# FLUX TALLY
    flux_tally = openmc.Tally(name='flux_tally')
    flux_tally.scores = ['flux']
    flux_tally.filters = [mesh_filter]
    tallies_file.append(flux_tally,merge=True)

# FUEL FLUX TALLY
    if args.burnup==True:
        fuel_tally=openmc.Tally(name='fuel_tally')
        fuel_tally.scores=['flux']
        fuel_tally.filters=[openmc.CellFilter(Fuel_tally_Filter)]
        tallies_file.append(fuel_tally,merge=True)

# FLUX Spectrum TALLY
    spectrum_tally = openmc.Tally(name='spectrum_tally')
    spectrum_tally.scores = ['flux']
    spectrum_tally.filters = [openmc.EnergyFilter(np.logspace(-3, 7.3, 51))]
    tallies_file.append(spectrum_tally,merge=True)

    cell_filter=openmc.CellFilter(root_cell)

# Zernike Dist Tally
    flux_tally_zernike=openmc.Tally(name='Zernike_tally')
    flux_tally_zernike.scores=['flux']
    zernike_filter=openmc.ZernikeRadialFilter(order=10,x=0,y=0,r=190)
    flux_tally_zernike.filters=[cell_filter,zernike_filter,openmc.EnergyFilter([0,0.025,1,1e6,2e7])]
    tallies_file.append(flux_tally_zernike,merge=True)

# Legendre Dist Tally
    flux_tally_legendre=openmc.Tally(name='Legendre_tally')
    flux_tally_legendre.scores=['flux']
    legendre_filter=openmc.SpatialLegendreFilter(10, 'z', 351.818-610, 351.818)
    flux_tally_legendre.filters=[cell_filter,legendre_filter,openmc.EnergyFilter([0,0.025,1,1e6,2e7])]
    tallies_file.append(flux_tally_legendre,merge=True)

# Zernike 2D DIst Tally
    flux_tally_zernike2d=openmc.Tally(name='2DZernike_Tally')
    flux_tally_zernike2d.scores=['flux']
    zernike2d_filter=openmc.ZernikeFilter(order=10,x=0,y=0,r=190)
    flux_tally_zernike2d.filters=[cell_filter,zernike2d_filter,openmc.EnergyFilter([0,0.025,1,1e6,2e7])]
    tallies_file.append(flux_tally_zernike2d,merge=True)
    if args.post=='Default' or args.post=='Run_Only':
        tallies_file.export_to_xml()
    
    

# ### - Criticality Calculation Setup
# this part will execute criticality calculation. The source for this calculation is the core universe.

# In[ ]:

'''
#----------------------------------------------------------------------
#==============================Keff SET-UP=============================
#----------------------------------------------------------------------
'''

print('=========Start Set up Keff========')
settings_file = openmc.Settings()
settings_file.batches = args.batch
settings_file.inactive = args.skipped
settings_file.particles = args.particle

print('Cycle ======',int(args.batch),' // ',int(args.skipped),' N= ',int(args.particle))
bounds=[-90,-90,0,90,90,180]

if args.burnup==False:
    source=openmc.Source()
    source.space=openmc.stats.Box(bounds[:3], bounds[3:],only_fissionable=False)
    settings_file.source = source

settings_file.output = {'tallies': False}
settings_file.material_cell_offsets=False
if args.post=='Default' or args.post=='Run_Only':
    settings_file.export_to_xml()

# ### - PLOT Models
# 
# In[ ]:
'''
#----------------------------------------------------------------------
#==============================PLOT MODELS=============================
#----------------------------------------------------------------------
'''
filetestresultname='Sequence'+str(stp)

pebble_sample=np.zeros((4,depleted_mat))

'''
for j in range (0,depleted_mat):    
    count=0; cont=0
    while count==0:
        if Pebbledf.iloc[cont][6]==j+1:
            for k in range (0,3):
                pebble_sample[k][j]=Pebbledf.iloc[cont][k]
            pebble_sample[3][j]=Pebbledf.index[cont]
            count+=1
        cont+=1
    plotn=openmc.Plot()
    plotn.filename='pebble_type'+str(j+1)
    plotn.basis='xz'
    plotn.origin=[pebble_sample[0][j]+Trisodf.iloc[80][0],pebble_sample[1][j]+Trisodf.iloc[80][1],pebble_sample[2][j]+Trisodf.iloc[80][2]]
    plotn.width=[0.025,0.025]
    plotn.pixels=[500,500]
    plotn.color_by='material'
    openmc.plot_inline(plotn)
'''
if args.post=='Default' or args.post=='Post_Only':
    print('========Print to PPM plots========')
    res=1000
    plot1 = openmc.Plot(plot_id=1)
    plot1.filename = 'materials-xz-'+str(int(stp))
    plot1.basis='xz'
    plot1.origin = [0, 0, 46.818]
    plot1.width = [380, 610]
    plot1.pixels = [res, int(610*res/380)]
    plot1.color_by = 'material'

    if args.control==True:
        zplot=round(352.818+Z_control[-1]-(Z_control[-1]+Z_base),3)
        print('=============='+str(zplot)+'=============')
    else:
        zplot=46.818

    plot2 = openmc.Plot(plot_id=2)
    plot2.filename = 'materials-xy-'+str(int(stp))
    plot2.basis='xy'
    plot2.origin = [0, 0, zplot]
    plot2.width = [380, 380]
    plot2.pixels = [res, res]
    plot2.color_by = 'material'
    
    plot2T = openmc.Plot(plot_id=2)
    plot2T.filename = 'materials-xyT-'+str(int(stp))
    plot2T.basis='xy'
    plot2T.origin = [Pebbledf['x'].loc[22328.0]+Trisodf['x'].iloc[2], 
                     Pebbledf['y'].loc[22328.0]+Trisodf['y'].iloc[2], 
                     Pebbledf['z'].loc[22328.0]+Trisodf['z'].iloc[2]]
    plot2T.width = [0.091, 0.091]
    plot2T.pixels = [res, res]
    plot2T.color_by = 'material'

    plot2P = openmc.Plot(plot_id=2)
    plot2P.filename = 'materials-xyP-'+str(int(stp))
    plot2P.basis='xy'
    plot2P.origin = [Pebbledf['x'].loc[22328.0], Pebbledf['y'].loc[22328.0], Pebbledf['z'].loc[22328.0]]
    plot2P.width = [6, 6]
    plot2P.pixels = [res, res]
    plot2P.color_by = 'material'

    openmc.plot_inline(plot2P)
    openmc.plot_inline(plot2T)

    plot3=openmc.Plot(plot_id=3)
    plot3.filename='Voxel_'+str(int(stp))
    plot3.type='voxel'
    plot3.origin=[0, 0, 46.818]
    plot3.width=(380,380,610)
    plot3.pixels=(500,500,int(610*500/380))
    plot3.color_by='material'

    #Show plot
    openmc.plot_inline(plot1)
    openmc.plot_inline(plot2)
    openmc.plot_inline(plot2P)
    openmc.plot_inline(plot2T)
    #openmc.plot_inline(plot3)

# ### - Run Setup
# Only available if **args.burnup** is True. The depletion calculation used CASL VERA benchmark for PWR chain downloaded in this site.
# 
# *link*: https://press3.mcs.anl.gov/openmc/depletion-chains/ and 
# https://info.ornl.gov/sites/publications/files/Pub61302.pdf
# 
# The integrator for this calculation uses EPC-RK4 algorithm. After calculation the result will be stored as **Fuel.prop** file inside the simulation directory created in the first line. 

# In[ ]:

'''
#----------------------------------------------------------------------
#========================SIMULATION SET-UP=============================
#----------------------------------------------------------------------
'''
if args.post=='Default' or args.post=='Run_Only':
    if args.burnup==False:
        print('\n')
        openmc.run()
    if args.burnup==True:
        print("======Start Set up DEPLETION======")

        operator = openmc.deplete.Operator(geometry, settings_file, header+"/chain_casl_pwr.xml")
        power=args.Burn_power*1e6
        
        print('Power =====================',int(args.Burn_power),' MW')
        days=args.timeline[stp+1]-args.timeline[stp]
        
        print('Days  ================',int(args.timeline[stp]),' to ',int(args.timeline[stp+1]),'\n')
        time_steps = [days * 24 * 60 * 60] * 1
        integrator = openmc.deplete.EPCRK4Integrator(operator, time_steps, power)
        #integrator = openmc.deplete.PredictorIntegrator(operator, time_steps, power)
        integrator.integrate()
        #openmc.run()

        results=openmc.deplete.ResultsList.from_hdf5('./depletion_results.h5')
        nuclide=list(chain.nuclide_dict.keys())
        time, k=results.get_eigenvalue()
        time/=(24*3600)
        time+=args.timeline[stp]

        matlist=list()
        for i in range(0,depleted_mat):
            if Segment_vol[i]==0:
                nuclide_list=np.zeros((len(time),len(nuclide)))
            else:
                nuclide_list=np.empty((len(time),len(nuclide)))
                for n in range(0,len(nuclide)):
                    nuclide_list[:,n]=results.get_atoms(str(i+1),nuclide[n])[1]*1e-24/Segment_vol[i]
            matlist.append(pd.DataFrame(np.column_stack([np.ones((len(time)))*(i+1),time,nuclide_list]),columns=['material','time']+nuclide))

        matlist=pd.concat(matlist,ignore_index=True)
        matlist.to_csv('Fuel.Prop'+str(stp))


# ## 6. Post Processing
# 
# only available when **args.tally** initiated.

# ### - Store the Tally Dataframe
# THis line used to extract the data from **openmc.StatePoint** output as csv dataframe.

# In[ ]:

'''
#----------------------------------------------------------------------
#==============================Post SET-UP=============================
#----------------------------------------------------------------------
'''
print('\n=========Start Set up post========')
sp=openmc.StatePoint('statepoint.'+str(settings_file.batches)+'.h5',autolink=True)

if args.tally==True:
    #if args.post=='Default':
    if args.post=='Default' or args.post=='Post_Only':
        #if args.burnup==True:
        #    fuel_tally=sp.get_tally(name='fuel_tally')
        #    fuel_result=fuel_tally.get_pandas_dataframe()
        #    fuel_result.to_csv('F4-n_result_'+str(stp)+'.csv',sep='\t')
        
        flux_tally=sp.get_tally(name='flux_tally')
        flux_result=flux_tally.get_pandas_dataframe()
        flux_result.to_csv('flux_result_'+str(stp)+'.csv',sep='\t')

        #spectrum_tally=sp.get_tally(name='spectrum_tally')
        #spectrum_result=spectrum_tally.get_pandas_dataframe()
        #spectrum_result.to_csv('spectrum_result_'+str(stp)+'.csv',sep='\t')

        #legendre_tally=sp.tallies[flux_tally_legendre.id].get_pandas_dataframe()
        #legendre_tally.to_csv('legendre_result_'+str(stp)+'.csv',sep='\t')
    
        #zernike_tally=sp.tallies[flux_tally_zernike.id].get_pandas_dataframe()
        #zernike_tally.to_csv('zernike_result_'+str(stp)+'.csv',sep='\t')
    
        #zernike2d_tally=sp.tallies[flux_tally_zernike2d.id].get_pandas_dataframe()
        #zernike2d_tally.to_csv('zernike2d_result_'+str(stp)+'.csv',sep='\t')
'''
if args.post=='Post_Only':
    spectrum_result=pd.read_csv('spectrum_result_'+str(stp)+'.csv',sep='\t',index_col=[0]	)
    flux_result=pd.read_csv('flux_result_'+str(stp)+'.csv',sep='\t',index_col=[0],header=[0,1])

    legendre_tally=pd.read_csv('legendre_result_'+str(stp)+'.csv',sep='\t',index_col=[0])
    zernike_tally=pd.read_csv('zernike_result_'+str(stp)+'.csv',sep='\t',index_col=[0])    
    zernike2d_tally=pd.read_csv('zernike2d_result_'+str(stp)+'.csv',sep='\t',index_col=[0])    
'''

# ### -Plot Flux Distribution by Polynomial Coefficient
# yyyy

# In[ ]:

def flux_distribution(header,stp,kf):
    zernike_array=np.zeros((6,4))
    legendre_array=np.zeros((11,4))
    zernike_stdev=np.zeros((6,4))
    legendre_stdev=np.zeros((11,4))
    
    style_seq=[['#1f77b4','#ff7f0e','#2ca02c','#d62728',
                '#9467bd','#8c564b','#e377c2'],
               ['D','s','^','o',',','p','*']]
    maximum_flux=list()
    function_flux=list()
    
    P=args.Burn_power*1e6
    nu=2.58
    w=202.79
    keff=kf.n
    stdevk=kf.s
    normeq=P*nu/(1.6022e-13*w*keff)

    phi_list=list(); zz_list=list()
    legendre_tally=pd.read_csv('legendre_result_'+str(stp)+'.csv',sep='\t',index_col=[0])
    zernike_tally=pd.read_csv('zernike_result_'+str(stp)+'.csv',sep='\t',index_col=[0])
            
    for j in range(0,4):
        for i in range(0,11):
            legendre_array[i,j]=legendre_tally['mean'].iloc[(j+(4*i))]
            legendre_stdev[i,j]=legendre_tally['std. dev.'].iloc[(j+(4*i))]
        for i in range(0,6):
            zernike_array[i,j]=zernike_tally['mean'].iloc[(j+(4*i))]
            zernike_stdev[i,j]=zernike_tally['std. dev.'].iloc[(j+(4*i))]
                
        
    axial=np.linspace(351.818-610,351.818,int(610)*50)
    plt.figure(dpi=300,figsize=[2.48*2,3.28*2])
    plt.title('Neutron flux axial distribution step '+str(stp)+'\n')
    phi_sums=np.zeros(len(axial))
    for j in range(0,5):
        if j!=4:
            phi = openmc.legendre_from_expcoef(legendre_array[:,j],
                                               domain=(axial[0],axial[-1]))
            phi_lo = openmc.legendre_from_expcoef(legendre_array[:,j]-legendre_stdev[:,j],
                                                  domain=(axial[0],axial[-1]))
            phi_hi = openmc.legendre_from_expcoef(legendre_array[:,j]+legendre_stdev[:,j],
                                                  domain=(axial[0],axial[-1]))
            phi_mean=phi(axial)
            phi_dvlo=phi_lo(axial)
            phi_dvhi=phi_hi(axial)
            phi_poin=list()
            for l in range(0,10):
                phi_poin.append(phi_mean[int((l/9)*(len(axial)-1))])
        
            plt.plot(phi_poin,np.linspace(axial[0],axial[-1],10)
                     ,style_seq[1][j],c=style_seq[0][j]
                     ,label='{:.1e}  eV'.format(legendre_tally['energy low [eV]'].iloc[j])+' to '+
                            '{:.1e} eV'.format(legendre_tally['energy high [eV]'].iloc[j]))
            plt.plot(phi_mean,axial,c=style_seq[0][j])
            plt.fill_betweenx(axial,phi_dvlo,phi_dvhi,color=style_seq[0][j],alpha=0.3)
            phi_sums+=phi(axial)#*normeq/(math.pi*610*190**2)
            phi_list.append(phi)
        else:
            plt.plot(phi_sums,axial,label='total')
    plt.ylabel('\nZ position [cm]')
    plt.xlabel('Flux [$neutron * cm/source$]')
    plt.legend()
    plt.xlim([0,np.max(phi_sums)])
    plt.savefig('PLOT_axialflux'+str(stp)+'.png',bbox_inches='tight')
    plt.show()
        
    radial=np.linspace(0,190,int(190)*50)
    plt.figure(dpi=300,figsize=[2.48*2,3.28*2])
    plt.title('Neutron flux radial distribution step '+str(stp)+'\n')
    zz_sums=np.zeros(len(radial))
    for j in range(0,5):
        if j!=4:
            zz = openmc.ZernikeRadial(zernike_array[:,j],radius=190)
            zz_lo=openmc.ZernikeRadial(zernike_array[:,j]-zernike_stdev[:,j],radius=190)
            zz_hi=openmc.ZernikeRadial(zernike_array[:,j]+zernike_stdev[:,j],radius=190)
            zz_mean=np.array(zz(radial))
            zz_dvlo=np.array(zz_lo(radial))
            zz_dvhi=np.array(zz_hi(radial))
            zz_poin=list()
            for l in range(0,10):
                zz_poin.append(zz_mean[int((l/9)*(len(radial)-1))])
        
            plt.plot(np.linspace(radial[0],radial[-1],10),zz_poin
                     ,style_seq[1][j],c=style_seq[0][j]
                     ,label='{:.1e}  eV'.format(legendre_tally['energy low [eV]'].iloc[j])+' to '+
                            '{:.1e} eV'.format(legendre_tally['energy high [eV]'].iloc[j]))
            plt.plot(radial,zz_mean,c=style_seq[0][j])
            plt.fill_between(radial,zz_dvlo,zz_dvhi,color=style_seq[0][j],alpha=0.3)
            zz_sums+=np.array(zz(radial))#*normeq/(math.pi*610*190**2)
            zz_list.append(zz)
        else:
            plt.plot(radial,zz_sums,label='total')
    plt.xlabel('Radial position [cm]')
    plt.ylabel('\nFlux [$neutron * cm/source$]')
    plt.ylim([0,np.max(zz_sums)])
    plt.legend()
    plt.savefig('PLOT_radialflux'+str(stp)+'.png',bbox_inches='tight')
    plt.show()
    maximum_flux.append([phi_sums,zz_sums])
    function_flux.append([phi_list,zz_list])
    return()

# ### -Plot Combined Plot
# yyyy

# In[ ]:

def Plot_Joint_Flux(stp,header,mesh_res,normeq,kf,Pebbledf):
    style_seq=[['#1f77b4','#ff7f0e','#2ca02c','#d62728',
                '#9467bd','#8c564b','#e377c2'],
               [',','s','^','o','D','p','*']]
    legendre_tally=pd.read_csv('legendre_result_'+str(stp)+'.csv',sep='\t',index_col=[0])    
    zernike_tally=pd.read_csv('zernike_result_'+str(stp)+'.csv',sep='\t',index_col=[0])

    zernike_array=np.zeros((6,4))
    legendre_array=np.zeros((11,4))
    zernike_stdev=np.zeros((6,4))
    legendre_stdev=np.zeros((11,4))
    
    for j in range(0,4):

        for i in range(0,11):
            legendre_array[i,j]=legendre_tally['mean'].iloc[(j+(4*i))]
            legendre_stdev[i,j]=legendre_tally['std. dev.'].iloc[(j+(4*i))]
        for i in range(0,6):
            zernike_array[i,j]=zernike_tally['mean'].iloc[(j+(4*i))]
            zernike_stdev[i,j]=zernike_tally['std. dev.'].iloc[(j+(4*i))]
    
    radial=np.linspace(0,190,int(190))
    axial=np.linspace(351.818-610,351.818,int(610))
    flux=list(range(0,5))
    for i in range(0,4):
        zz=openmc.ZernikeRadial(zernike_array[:,i],radius=190)
        pp=openmc.legendre_from_expcoef(legendre_array[:,i],domain=(axial[0],axial[-1]))
        flux[i]=((np.array([pp(axial)]).T @ np.array([zz(radial)])))
        flux[-1]+=flux[i]
    fig=plt.figure(dpi=300,figsize=[12*(1/6+1/10),12*(1/2)])
    plt.rcParams.update({'font.size': 9})

#    plt.tight_layout()
    gs=gridspec.GridSpec(2,2,width_ratios=[1,1],height_ratios=[1,0.02])
    gs.update(wspace=0.05,hspace=0.2)

    ax1=plt.subplot(gs[0,1])
    ax1.set_title('Flux distribution\n step '+str(stp),fontsize=9)
    plt1=ax1.pcolor(radial, axial, flux[-1], 
               cmap='viridis')
#    ax1.set_xlabel(r'Radial Position [cm]')
    ax1.set_xlabel(r'Radial Position [cm]')
    ax1.set_ylabel(r' ')
    ax1.set_xticks([50,100,150])
    ax1.set_xticklabels([50,100,150])
    ax1.set_yticks([])
    ax1.set_yticklabels([])
    ax1.set_aspect(1)            
    
#==========================,figsize=[11.7/3,8.3/2]============================    
    ax2=plt.subplot(gs[0,0])
    ax2.set_title('Pebble Distribution\n step '+str(stp),fontsize=9)
    ax2.grid(True)
    for name, Pebbledf_group in Pebbledf.groupby('label'):
        if name!=0:
            ax2.scatter(-Pebbledf_group['rho'],Pebbledf_group['z'],
                        edgecolors=None,s=0.3,label='group '+str(int(name)))
    ax2.set_aspect(1)
    ax2.set_xlabel(r' ')
    ax2.set_ylabel(r'Axial Height [cm]')
    ax2.set_xticks([-150,-100,-50])
    ax2.set_xticklabels([150,100,50])
    ax2.set_yticks(np.linspace(351.818-610,351.818,10))
    ax2.set_xlim([-190,0])
    ax2.set_ylim([351.818-610,351.818])
    
    cbax=plt.subplot(gs[1,:])
    cbar=Colorbar(mappable=plt1,ax=cbax, orientation='horizontal',ticklocation='bottom')
    cbar.set_label('$\phi$ [$neutron * cm/source$]')
    plt.savefig("PLOT_distribution_interpolated_"+str(stp)+".png",bbox_inches='tight')
    #=============================================================================   
    #=============================================================================
    return()

def vtk_file(mesh,flux_result,mesh_res,stp,normeq):
    mesh_array=np.zeros(mesh)
    #for i in range(0,len(flux_result)):
    #    mesh3darray[flux_result.iloc[i][0]-1,flux_result.iloc[i][1]-1,flux_result.iloc[i][2]-1]=flux_result.iloc[i][5]

    imdata = vtk.vtkImageData()
    depthArray = numpy_support.numpy_to_vtk(np.array(flux_result.iloc[:,5])*normeq, deep=True, array_type=vtk.VTK_DOUBLE)

    imdata.SetDimensions(mesh_array.shape)
    imdata.SetSpacing([mesh_res/100,mesh_res/100,mesh_res/100])
    imdata.SetOrigin([-1.90,-1.90,-2.58182])
    imdata.GetPointData().SetScalars(depthArray)

    writer = vtk.vtkMetaImageWriter()
    writer.SetFileName("flux_meshavg_"+str(int(stp))+".mhd")
    writer.SetInputData(imdata) 
    writer.Write()

def TallyAnimation(mesh,flux_result,stp):
    mesh3darray=np.zeros(mesh)
    for i in range(0,len(flux_result)):
        mesh3darray[flux_result.iloc[i][0]-1,flux_result.iloc[i][1]-1,flux_result.iloc[i][2]-1]=flux_result.iloc[i][5]

    mesh3darray=mesh3darray/3**3
    xgrid=np.round(np.linspace(-190,190,mesh3darray.shape[0]),3)
    ygrid=np.round(np.linspace(-190,190,mesh3darray.shape[1]),3)
    zgrid=np.round(np.linspace(351.818-610,351.818,mesh3darray.shape[2]),3)
    
    xz_extent = [xgrid[0],xgrid[-1],zgrid[0],zgrid[-1]]
    XZims=[]
    f2=plt.figure(dpi=300)
    f2.set_size_inches(6,6,True)
    for i in range(0,mesh3darray.shape[1]-1):
        if np.count_nonzero(mesh3darray[:,i,:])!=0:
            ax=f2.add_subplot(111)
            ax.set_xlabel('x [cm]')
            ax.set_ylabel('z [cm]')
            ax.set_yticks(np.round(np.arange(351.818-610,351.818,step=0.1*610),3))
            ax.set_xticks(np.round(np.arange(-190,190,step=0.25*380),3))
            ax.set_yticklabels(np.round(np.arange(351.818-610,351.818,step=0.1*610),3),fontsize=9)
            ax.set_xticklabels(np.round(np.arange(-190,190,step=0.25*380),3),fontsize=9)
            title = ax.text(0.5,1.05,'PLOT ygrid from \n'+ str(ygrid[i])+' cm \nto '+ str(ygrid[i+1])+' cm', 
                    size=plt.rcParams["axes.titlesize"],
                    ha="center", transform=ax.transAxes)
            img=ax.imshow(np.flip(np.transpose(mesh3darray[:,i,:]),0),extent=xz_extent,cmap=plt.get_cmap('magma'),interpolation='spline16',aspect=1,
                          animated=True,vmin=0,vmax=np.max(mesh3darray))
            XZims.append([img,title])
    cbar=plt.colorbar(img,format='%.3e'); cbar.set_label('flux ($n/cm^2.s$)')
    XZanim=animation.ArtistAnimation(f2, XZims,interval=500,blit=True)
    XZanim.save('XZ_'+"PLOT_distribution_meshavg_"+str(stp)+'.gif',writer='imagekick',fps=1000/100)

    xy_extent = [xgrid[0],xgrid[-1],ygrid[0],ygrid[-1]]
    XYims=[]
    f1=plt.figure(dpi=300)
    f1.set_size_inches(6.4, 4.8, True)
    for i in range(0,len(zgrid)-1):
        if np.count_nonzero(mesh3darray[:,:,i])!=0:
            ax=f1.add_subplot(111)
            ax.set_xlabel('x [cm]')
            ax.set_ylabel('y [cm]')
            ax.set_yticks(np.arange(ygrid[0], ygrid[-1], step=0.1*(ygrid[-1]-ygrid[0])))
            ax.set_xticks(np.arange(xgrid[0], xgrid[-1], step=0.1*(xgrid[-1]-xgrid[0])))
            title = ax.text(0.5,1.05,'PLOT zgrid from \n'+ str(zgrid[i])+' cm to '+ str(zgrid[i+1])+' cm', 
                    size=plt.rcParams["axes.titlesize"],
                    ha="center", transform=ax.transAxes)
            img=ax.imshow((mesh3darray[:,:,i]),extent=xy_extent,cmap=plt.get_cmap('magma'),interpolation='spline16',aspect=1,
                          animated=True,vmin=0,vmax=np.max(mesh3darray))
            XYims.append([img,title])
    cbar=plt.colorbar(img,format='%.3e'); cbar.set_label('flux ($n/cm^2.s$)')
    XYanim=animation.ArtistAnimation(f1, XYims,interval=500,blit=True)
    XYanim.save('XY_'+"PLOT_distribution_meshavg_"+str(stp)+'.gif',writer='imagekick',fps=1000/100)

# Define needed function first 
def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def vtk_interpolation(legendre_tally,zernike2d_tally,mesh_res,stp):
    legendre_array=np.zeros((11,4))
    legendre_stdev=np.zeros((11,4))
    zernike2d_array=np.zeros((66,4))
    zernike2d_stdev=np.zeros((66,4))

    for l in range(0,4):
        for m in range(0,11):
            legendre_array[m,l]=legendre_tally['mean'].iloc[(l+(4*m))]
            legendre_stdev[m,l]=legendre_tally['std. dev.'].iloc[(l+(4*m))]
        for m in range(0,66):
            zernike2d_array[m,l]=zernike2d_tally['mean'].iloc[(l+(4*m))]
            zernike2d_stdev[m,l]=zernike2d_tally['std. dev.'].iloc[(l+(4*m))]

        zz = openmc.Zernike(zernike2d_array[:,l], radius=190) 
        phi = openmc.legendre_from_expcoef(legendre_array[:,l], domain=(351.818-610, 351.818))

        x = np.linspace(-190, 190, int(380/mesh_res))
        y = x
        z = np.linspace(351.818-610, 351.818, int(610/mesh_res))

        [X,Y] = np.meshgrid(x,y)
        [r, theta] = cart2pol(X,Y)

        flux3d = np.zeros((len(x), len(y), len(z)))
        #flux3d.fill(np.nan)
        flux1d = np.zeros(len(x)*len(y)*len(z)); cnt=0
        for k in range(len(z)):
            for j in range(len(y)):
                for i in range(len(x)):
                    flux3d[i][j][k] = phi(z[k]) * zz(r[i][j], theta[i][j])
                    if flux3d[i][j][k]>=0:
                        flux1d[cnt]=(flux3d[i][j][k])
                    cnt+=1

        imdata = vtk.vtkImageData()
        depthArray = numpy_support.numpy_to_vtk(np.array(flux1d), deep=True, array_type=vtk.VTK_DOUBLE)

        imdata.SetDimensions(flux3d.shape)
        imdata.SetSpacing([mesh_res,mesh_res,mesh_res])
        imdata.SetOrigin([0,0,0])
        imdata.GetPointData().SetScalars(depthArray)

        writer = vtk.vtkMetaImageWriter()
        writer.SetFileName("flux_polynomial"+'_{:.1e}'.format(legendre_tally['energy low [eV]'].iloc[l])+'-'+
                            '{:.1e}'.format(legendre_tally['energy high [eV]'].iloc[l])+"_"+str(int(stp))+".mhd")
        writer.SetInputData(imdata) 
        writer.Write()
    return(flux3d,flux1d)

# ### -Plotting Driver
# yyyy

# In[ ]:

if args.tally==True:
    MCNPX_keff = pd.read_csv('../../MCNP_benchmark/keff_depletion.txt', sep='\t', header=[0], index_col=[0])
    if args.post=='Default' or args.post=='Post_Only':
        if args.burnup==True:
            time, k=openmc.deplete.ResultsList.from_hdf5('./depletion_results.h5').get_eigenvalue()
            k_combined=k[0,0]
        else:
            k_combined=sp.k_combined.n            
        #normeq=1e7*MCNPX_keff['ave. nu'].iloc[stp*2]/(1.6022e-13*MCNPX_keff['ave. q'].iloc[stp*2]*k_combined*mesh_res**3)
        print('keff = ',k_combined)
        normeq=1
        kf=sp.k_combined

        flux_distribution(header,stp,kf)
        Plot_Joint_Flux(stp,header,mesh_res,normeq,kf,Pebbledf)
        #flux3d,flux1d=vtk_interpolation(legendre_tally,zernike2d_tally,mesh_res,stp)
        vtk_file(mesh.dimension,flux_result,mesh_res,stp,normeq)
        #TallyAnimation(mesh.dimension,flux_result,stp)



# ## $\infty$. Finish Simulation

# In[ ]:
os.chdir(header)
'''
runlist=['0 10000 8600000 550 10000 /media/feryantama/Heeiya/openmc/TEST/data/lib80x_hdf5/cross_sections.xml --timeline 0 7 14 21 28 38 48 58 68 98 128 158 188 218 248 278 308 338 368 --control True --tally True --post Post_Only --controlstep 0 --skipped 50',
         '0 10000 8600000 550 10000 /media/feryantama/Heeiya/openmc/TEST/data/lib80x_hdf5/cross_sections.xml --timeline 0 7 14 21 28 38 48 58 68 98 128 158 188 218 248 278 308 338 368 --control True --tally True --post Post_Only --controlstep 9 --skipped 50',
         '0 10000 8600000 210 5000 /media/feryantama/Heeiya/openmc/TEST/data/lib80x_hdf5/cross_sections.xml --timeline 0 7 14 21 28 38 48 58 68 98 128 158 188 218 248 278 308 338 368 --burnup True --tally True --post Post_Only',
         '17 10000 8600000 210 5000 /media/feryantama/Heeiya/openmc/TEST/data/lib80x_hdf5/cross_sections.xml --timeline 0 7 14 21 28 38 48 58 68 98 128 158 188 218 248 278 308 338 368 --burnup True --tally True --post Post_Only']
for text in runlist:
    runfile('/media/feryantama/Data_Fereee/openmc/OpenMC_benchmark/OpenMC_RUN.py', wdir='/media/feryantama/Data_Fereee/openmc/OpenMC_benchmark',args=text)
'''
