# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 09:20:27 2020

@author: Feryantama Putra
Spyder Editor

This is a temporary script file.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import os
#import openmc.deplete
#import openmc
import argparse
from pyevtk.hl import pointsToVTK
import matplotlib.gridspec as gridspec
from matplotlib.colorbar import Colorbar
from scipy.optimize import curve_fit

class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter,
                      argparse.RawDescriptionHelpFormatter):
    pass


parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=CustomFormatter)
parser.add_argument('step', type=int,
                    help='step to be accounted')
parser.add_argument('deltas', type=int,
                    help='delta steps of DEM')
parser.add_argument('initial', type=int,
                    help='initial steps of DEM')
parser.add_argument('MCNPXoutput', type=str,
                    help="MCNPX output resume file directory")
parser.add_argument('OpenMCoutput', type=str,
                    help="MCNPX output resume file directory")
parser.add_argument('--timeline', nargs="+", type=float, required=True,
                    help='list of {t(step-1),t(step),t(step+1)} separated by space')
parser.add_argument('--Burn_power', type=float,
                    help='Power in burn code in MW',default=10)
#args = parser.parse_args()
# '''
args = parser.parse_args(args=(['17', '10000', '8600000',
                                '../MCNP_benchmark', '.',
                                '--timeline'] +
                               [str(x) for x in range(0, 35, 7)] +
                               [str(x) for x in range(38, 78, 10)] +
                               [str(x) for x in range(98, 378, 30)]))


# '''

stp = []
for i in range(0, len(args.timeline)):
    stp.append(int(args.timeline[i]*args.deltas+(100000*i)+args.initial))
interest = ['U235', 'U238', 'Pu239', 'Pu240', 'Cs137', 'Xe135', 'I135']
def polar_angle(x, y):
    theta = math.atan(y/x)
    if x >= 0 and y >= 0:
        theta = theta
    if x < 0 and y >= 0:
        theta = theta+math.pi
    if x < 0 and y < 0:
        theta = theta+math.pi
    if x >= 0 and y < 0:
        theta = theta+math.pi*2
    return theta

def make_pebbledf(filepath):
    point_position = 1e+20
    endpoint_position = 2e+20
    id_position = 1e+20
    endid_position = 2e+20
    type_position = 1e+20
    endtype_position = 2e+20

    pointList = list()
    idList = list()
    typeList = list()

    with open(filepath) as fp:
        cnt = 1
        for line in fp:
            for part in line.split():
                if "POINTS" in part:
                    point_data = line.replace(
                        'POINTS ', '').replace(' float\n', '')
                    point_data = int(point_data)
                    point_position = cnt
                    endpoint_position = math.ceil(point_position+point_data/3)
                elif "id" in part:
                    id_data = line.replace('id 1 ', '').replace(' int\n', '')
                    id_data = int(id_data)
                    id_position = cnt
                    endid_position = math.ceil(id_position+id_data/9)
                elif "type" in part:
                    type_data = line.replace(
                        'type 1 ', '').replace(' int\n', '')
                    type_data = int(type_data)
                    type_position = cnt
                    endtype_position = math.ceil(type_position+type_data/9)

            if (cnt > point_position and cnt < endpoint_position+1):
                pointList.extend(line.replace(' \n', '').split(' '))
            elif (cnt > id_position and cnt < endid_position+1):
                idList.extend(line.replace(' \n', '').split(' '))
            elif (cnt > type_position and cnt < endtype_position+1):
                typeList.extend(line.replace(' \n', '').split(' '))
            cnt += 1

    pointarr = np.zeros([point_data, 3])
    idarr = np.zeros([point_data])
    typearr = np.zeros([point_data])
    rho = np.zeros([point_data])
    theta = np.zeros([point_data])

    cnt = 0
    for i in range(0, point_data):
        pointarr[i, 0] = float(pointList[cnt*3+0])*100
        pointarr[i, 1] = float(pointList[cnt*3+1])*100
        pointarr[i, 2] = float(pointList[cnt*3+2])*100
        rho[i] = math.sqrt((pointarr[i, 0]**2)+(pointarr[i, 1]**2))
        theta[i] = (polar_angle(pointarr[i, 0], pointarr[i, 1]))
        typearr[i] = int(typeList[i])
        idarr[i] = int(idList[i])
        cnt += 1

    datasets = np.column_stack((pointarr, typearr, rho, theta))
    Pebbledf = pd.DataFrame(data=datasets, index=idarr, columns=[
                            'x', 'y', 'z', 'type', 'rho', 'theta'])
    return(Pebbledf)


def vtk_PolyDataFile(Pebbledf,stp,directory):
    pointsToVTK(directory+"/flux_pebble_"+str(int(stp)),
                np.array(Pebbledf['x'])/100,
                np.array(Pebbledf['y'])/100,
                np.array(Pebbledf['z'])/100,
                data={"flux" : np.array(Pebbledf['flux mean']), "label" : np.array(Pebbledf['label'])})

def OpenMC_FUEL_Tally(stp,MCNPX_keff=[0],OpenMC_keff=[0],normalize=False):
    Pebbledf = []
    for i in range(0, len(stp)):
        Pebbledf.append(make_pebbledf(args.OpenMCoutput +'/DEM_post/pebble_'+str(stp[i])+'.vtk'))
        labdf = pd.read_csv(args.OpenMCoutput+'/step_'+str(i)+'-DEPLETE'+'/Prop.Label',sep='\t', index_col=[0], header=[0])
        #labdf=pd.read_csv('../MCNP_benchmark/Prop_Label_'+str(int(i))+'.txt',sep='\t',index_col=[0],header=[0])
        fluxdf = pd.read_csv(args.OpenMCoutput+'/step_'+str(i)+'-DEPLETE'+'/F4-n_result_'+str(i)+'.csv',
                             sep='\t', index_col=[0], header=[0])
        Pebbledf[i] = pd.concat([Pebbledf[i], labdf], axis=1)
        fluxlist = np.zeros((len(Pebbledf[i]), 2))
        cnt = 0
        for j in range(0, len(Pebbledf[i])):
            if Pebbledf[i]['z'].iloc[j] >= (-258.182-3):
                if Pebbledf[i]['type'].iloc[j]==1:
                    fluxlist[j, :] = np.array([fluxdf['mean'].iloc[cnt],fluxdf['std. dev.'].iloc[cnt]])
                cnt += 1
        if normalize==True:
            fluxlist*=(args.Burn_power*1e6*MCNPX_keff['ave. nu'].iloc[i*2]/(1.6022e-13*MCNPX_keff['ave. q'].iloc[i*2]*OpenMC_keff[i*2,0]))/(4/3*math.pi*3**3)
        Pebbledf[i]['flux mean'], Pebbledf[i]['flux_dev'] = fluxlist[:,0], fluxlist[:, 1]
        vtk_PolyDataFile(Pebbledf[i],i,'./step_'+str(i)+'-DEPLETE')
    return(Pebbledf)


def OpenMC_BURNUP_Keff(stp, Pebblelist, MCNPX_tabledata, ax_size=5, rd_size=3,):
    header = os.getcwd()
    pebble_disrrray = np.zeros(
        (rd_size*(ax_size+(len(stp)-1)-1), (len(stp)-1)))
    keff = np.zeros(((len(stp)-1)*2, 3))

    for step in range(0, len(stp)-1):
        
        tot=0
        for i in range(0, len(Pebblelist[step])):
            if Pebblelist[step]['z'].iloc[i] >= 351.818-610+3:
                tot+=1
                if (Pebblelist[step]['label'].iloc[i] != 0): 
                    pebble_disrrray[int(Pebblelist[step]['label'].iloc[i])-1, step] += 1

        results = openmc.deplete.ResultsList.from_hdf5(
            header+'/step_'+str(step)+'-DEPLETE/depletion_results.h5')
        keff[step*2, :2], keff[step*2+1, :2] = [results.get_eigenvalue()[1][0, :],
                                                results.get_eigenvalue()[1][1, :]]
        print(str(np.sum(pebble_disrrray[:, step]))+' pebble//step ' +
              str(step)+' total = '+str(tot))
        keff[step*2, 2] = stp[step]
        keff[step*2+1, 2] = stp[step+1]
    np.savetxt('depletion_pebble.txt', pebble_disrrray, delimiter='\t')
    np.savetxt('keff_depletion.txt', keff, delimiter='\t')
    MCNPX_pebbledisrray = np.flip(np.loadtxt('../MCNP_benchmark/depletion_pebble.txt',delimiter='\t'),0)
    Burnup_fraction=[np.zeros(MCNPX_pebbledisrray.shape)]
    Power_fraction=np.zeros(MCNPX_pebbledisrray.shape)
    n=15; m=9; BUp=[]
    for i in range(0,len(args.timeline[:-1])):
        result=pd.read_csv('./step_'+str(int(i))+'-DEPLETE/Fuel.Prop'+str(int(i)), index_col=[0])
        with open('../MCNP_benchmark/MCNP-output/ENDF71steps'+str(int(args.timeline[i]*args.deltas+(100000*i)+args.initial))) as fp:
            lin=fp.readlines();cnt=1
            for line in lin:
                if line=='1burnup summary table by material                                                                       print table 210\n':
                    tab_pos=cnt
                    tab_lis=[lin[tab_pos+17+7*i] for i in range(0,m+3*i)]
                    bat_lis=lin[tab_pos+8]
                cnt+=1
        Burnup_fraction[0][-int(m+3*i):,i]=np.array([float(lis[-10:-1]) for lis in tab_lis])
        Power_fraction[-int(m+3*i):,i]=np.array([float(lis[-25:-16]) for  lis in tab_lis])
        BUp.append(float(bat_lis[-22:-12]))
    print(BUp)
    BBB=np.array([[np.sum(np.array(BUp[:i])),np.sum(np.array(BUp[:i]))] for i in range(0,len(BUp)+1)]).flatten()[1:-1]
    MCNPX_keff = pd.read_csv(MCNPX_tabledata+'/keff_depletion.txt', sep='\t', header=[0], index_col=[0])
    column_list = []
    for i in range(0, len(keff)):
        difference = (MCNPX_keff['keff'].iloc[i]-keff[i, 0])*1e5
        dif_dev = (
            math.sqrt(MCNPX_keff['sdev keff'].iloc[i]**2+keff[i, 1]**2))*1e5
        column_list.append(['{:.5f}'.format(keff[i, 0])+'$ \pm ${:.2e}'.format(keff[i, 1]),
                            '{:.5f}'.format(
                                MCNPX_keff['keff'].iloc[i])+'$ \pm ${:.2e}'.format(MCNPX_keff['sdev keff'].iloc[i]),
                            '{:.2f}'.format(difference)+'$ \pm ${:.3f}'.format(dif_dev)])
    for i in range(0,len(keff)):
        if keff[i,0]>1:
            pos1=i
        if MCNPX_keff['keff'].iloc[i]>1:
            pos2=i
    crit1=(1-keff[pos1+1,0])/(keff[pos1,0]-keff[pos1+1,0])*(BBB[pos1]-BBB[pos1+1])+BBB[pos1+1]
    #crit1=np.interp(1,[keff[pos1-1,0],keff[pos1+1,0]],[BBB[pos1],BBB[pos1+1]])
    crit2=np.interp(1,[MCNPX_keff['keff'].iloc[pos2],
                       MCNPX_keff['keff'].iloc[pos2+1]],[BBB[pos2],BBB[pos2+1]])
    
    plt.rc('ytick', labelsize=10)
    plt.rc('xtick', labelsize=9)
    fig, (ax, ax_table) = plt.subplots(ncols=2, figsize=(11, 6), dpi=300,
                                       gridspec_kw=dict(width_ratios=[2, 1]))
    fig.tight_layout(pad=0.1)
    ax_table.axis("off")
    ax.grid(True)
    ax.plot(keff[:, 2], keff[:, 0], label='OpenMC 0.12.0 [ENDF B-VIII.0 + TENDL2019]\n 99% conf.lev')
    ax.fill_between(keff[:, 2], keff[:, 0]-keff[:, 1]*3,keff[:, 0]+keff[:, 1]*3, alpha=0.3)
    ax.plot(np.array(MCNPX_keff['time'], dtype=int),
            MCNPX_keff['keff'], label='MCNPX 2.6.0 [ENDF B-VII.0 + B-VII.1]\n 99% conf.lev')
    ax.fill_between(np.array(MCNPX_keff['time'], dtype=int), MCNPX_keff['keff'] -
                    MCNPX_keff['sdev keff'], MCNPX_keff['keff']+MCNPX_keff['sdev keff'], alpha=0.3)
    ax.plot(np.array(MCNPX_keff['time'],dtype=int),np.ones(len(MCNPX_keff)),'k--')
    ax2=ax.twiny()
    ax2.annotate('{:.2f}'.format(crit1),(crit1,1),
                 textcoords='offset points',va='center',xytext=(-10,20),arrowprops=dict(arrowstyle="->"))
    ax2.annotate('{:.2f}'.format(crit2),(crit2,1),
                 textcoords='offset points',va='center',xytext=(-10,20),arrowprops=dict(arrowstyle="->"))
    ax2.spines['bottom'].set_position(('outward',36))
    ax2.set_xlabel('BurnUp [GWd/MTU]')
    ax2.xaxis.set_label_position('bottom')
    ax2.xaxis.set_ticks_position('bottom')
    ax2.plot(BBB,MCNPX_keff['keff'],alpha=0)
    #bax.set_visible(False)
    ax2.set_xticks(np.linspace(0,np.max(BBB),12))
    ax2.set_xticklabels(['{:.2f}'.format(x) for x in np.linspace(0,np.max(BBB),12)])
    ax2.set_xlim([0, np.max(BBB)])
    bbox = [0, 0, 1, 1]
    tableau = ax_table.table(cellText=column_list,
                             rowLabels=[str(int(x)) for x in keff[:, 2]],
                             colLabels=('$k_{eff}$ OpenMC', '$k_{eff}$ MCNPX', 'difference [pcm]'), bbox=bbox)
    
    with open('6.txt', 'w') as file:
        file.writelines('\t'.join(str(j) for j in i) + '\n' for i in column_list)
    
    tableau.auto_set_font_size(False)
    tableau.set_fontsize(6.7)
    ax.set_xticks([int(x) for x in np.arange(0, 365, 30)])
    ax.set_xlim([0, np.max(stp)])

    ax.set_title('combined $k_{eff}$ vs time')
    ax.legend()
    ax.set_xlabel('Time [d]')
    ax.set_ylabel('\n$k_{eff}$')
    plt.savefig('PLOT_keff_Depletion.png', bbox_inches='tight')
    return(keff, pebble_disrrray, MCNPX_keff,BBB)


# keff_reference=Burnup_keff[0,:1]
def OpenMC_CONTROL_Keff(keff_reference, MCNPX_tabledata, c_batches=550, header=os.getcwd()):
    keff = np.zeros((10, 3))
    if not os.path.isfile('./keff_control.txt'):
        for step in range(0, 10):
            sp = openmc.StatePoint(header+'/step_'+str(int(step))+'-CONTROL/' +
                                   'statepoint.'+str(int(c_batches))+'.h5', autolink=True)
            delt = (394.2-171.2)/9
            Z_base = 171.2+step*delt
            keff[step][0], keff[step][1], keff[step][2] = [
                sp.k_combined.n, sp.k_combined.s, Z_base]

        np.savetxt('keff_control.txt', keff, delimiter='\t')
    else:
        keff = np.loadtxt('./keff_control.txt', delimiter='\t')
    MCNPX_keff = pd.read_csv(
        MCNPX_tabledata+'/keff_control.txt', sep='\t', header=[0], index_col=[0])
    column_list = []
    for i in range(0, len(keff)):
        difference = (MCNPX_keff['keff'].iloc[i]-keff[i, 0])*1e5
        dif_dev = (math.sqrt(MCNPX_keff['stdev'].iloc[i]**2+keff[i, 1]**2))*1e5
        column_list.append(['{:.5f}'.format(keff[i, 0])+'$ \pm ${:.2e}'.format(keff[i, 1]),
                            '{:.5f}'.format(
                                MCNPX_keff['keff'].iloc[i])+'$ \pm ${:.2e}'.format(MCNPX_keff['stdev'].iloc[i]),
                            '{:.2f}'.format(difference)+'$ \pm ${:.3f}'.format(dif_dev)])

    OpenMC_reactivity = np.zeros((keff.shape))
    OpenMC_reactivity[:, 0] = np.abs((keff[:, 0]-keff_reference[0,0])/keff_reference[0,0])
    OpenMC_reactivity[:, 1] = np.abs(OpenMC_reactivity[:, 0]*((keff_reference[0,1]/keff_reference[0,0])**2 +
                              (((keff[:, 1]**2+keff_reference[0,1]**2)**0.5)/OpenMC_reactivity[:, 0])**2)**0.5)
    
    MCNPX_reactivity = np.zeros((keff.shape))
    MCNPX_reactivity[:, 0] = np.abs((np.array(MCNPX_keff.iloc[:, 0])-
                                     np.array(keff_reference[1,0]))/
                                     np.array(keff_reference[1,0]))
    MCNPX_reactivity[:, 1] = np.abs(MCNPX_reactivity[:, 0]*((keff_reference[1,1]/keff_reference[1,0])**2 +
                              (((np.array(MCNPX_keff.iloc[:, 1])**2+np.array(keff_reference[1,1])**2)**0.5)/
                               MCNPX_reactivity[:, 0])**2)**0.5)
    
    plt.rc('ytick', labelsize=10)
    plt.rc('xtick', labelsize=9)
    fig, (ax, ax_table) = plt.subplots(ncols=2, figsize=(10.5, 4.5), dpi=300,
                                       gridspec_kw=dict(width_ratios=[2, 1]))
    fig.tight_layout(pad=0.1)
    ax_table.axis("off")
    ax.grid(True)

    ax.plot(keff[:, 2], keff[:, 0], label='$k_{eff}$ left axis\n '+r'$\rho$'+' right axis \nOpenMC 0.12.0 [ENDF B-VIII.0]\n')
    ax.fill_between(keff[:, 2], keff[:, 0]-keff[:, 1],
                    keff[:, 0]+keff[:, 1], alpha=0.3)
    ax.set_xticks(np.linspace(171.2, 394.2, 9))

    ax.plot(keff[:, 2], MCNPX_keff['keff'],
            label='$k_{eff}$ left axis\n '+r'$\rho$'+' right axis \nMCNPX 2.6.0 [ENDF B-VII.1]\n')
    ax.fill_between(keff[:, 2], MCNPX_keff['keff']-MCNPX_keff['stdev'],
                    MCNPX_keff['keff']+MCNPX_keff['stdev'], alpha=0.3)
    ax.set_xticks(np.linspace(171.2, 394.2, 9))

    ax2=ax.twinx()
    ax2.set_ylabel('Reactivity $\dfrac{\Delta k}{k} \%$')
    ax2.plot(keff[:,2],OpenMC_reactivity[:,0]*100, label=r'$\rho$'+' OpenMC 0.12.0 [ENDF B-VIII.0]')
    ax2.fill_between(keff[:, 2], (OpenMC_reactivity[:, 0]-OpenMC_reactivity[:, 1])*100,
                    (OpenMC_reactivity[:, 0]+OpenMC_reactivity[:, 1])*100, alpha=0.3)
    ax2.set_xticks(np.linspace(171.2, 394.2, 9))

    ax2.plot(keff[:, 2], MCNPX_reactivity[:,0]*100,
            label=r'$\rho$'+' MCNPX 2.6.0 [ENDF B-VII.1]')
    ax2.fill_between(keff[:, 2], (MCNPX_reactivity[:,0]-MCNPX_reactivity[:,1])*100,
                    (MCNPX_reactivity[:,0]+MCNPX_reactivity[:,1])*100, alpha=0.3)
    ax2.set_xticks(np.linspace(171.2, 394.2, 9))
    ax2.set_ylim([0,15])
    ax2.quiver(keff[5,2],MCNPX_reactivity[5,0]*100,15,0,color='k',linewidth=1.5)
    bbox = [0.35, 0.25, 1, 0.35]
    tableau = ax_table.table(cellText=column_list,
                             rowLabels=['{:.3e}'.format(x) for x in keff[:, 2]],
                             colLabels=('$k_{eff}$ OpenMC', '$k_{eff}$ MCNPX', 'difference [pcm]'), bbox=bbox)
    
    with open('5.txt', 'w') as file:
        file.writelines('\t'.join(str(j) for j in i) + '\n' for i in column_list)
        
    tableau.auto_set_font_size(False)
    tableau.set_fontsize(6.7)
    result=keff
    ax.set_title('combined $k_{eff}$ & reactivity '+r'$\rho$'+' vs control rod bottom tip position')
    ax.set_xlabel('Z-level [cm]')
    ax.set_ylabel('\n$k_{eff}\pm \sigma$')
    ax.legend(loc='right')
    #ax2.legend()
    plt.savefig('PLOT_keff_Control.png', bbox_inches='tight')
    # plt.show()

    return(result, OpenMC_reactivity,MCNPX_reactivity, MCNPX_keff)


def OpenMC_BURNUP_Resume(interest, stp, header=os.getcwd()):
    for step in range(0, len(stp)-1):
        nuclide_correction = [['Tc99_m1', 'Tc100', 'Rh102_m1', 'Rh103_m1', 'Rh105_m1', 'Rh106', 'Rh106_m1', 'Ag109_m1',
                               'Cd115', 'Te127', 'Xe137', 'Pr144', 'Nd151', 'Sm155', 'Pa234', 'Np240', 'Np240_m1'],
                              ['Tc99', 'Ru100', 'Rh102', 'Rh103', 'Rh105', 'Pd106', 'Pd106', 'Ag109', 'Cd114', 'I127',
                               'Cs137', 'Nd144', 'Pm151', 'Eu155', 'U234', 'Pu240', 'Pu240']]
        matlistdf = pd.read_csv(
            header+'/step_'+str(step)+'-DEPLETE/Fuel.Prop'+str(step), index_col=[0])
        if step == 0:
            maxes = int(matlistdf['material'].max())
            burnup_resume = matlistdf
            burnup_resume.index = list(range(0, len(burnup_resume)))
            deplete_array = np.zeros((maxes+3*(len(stp)-2), len(matlistdf.columns)-2, len(stp)-1))
        else:
            burnup_resume = burnup_resume.append(matlistdf, ignore_index=True, sort=False)
        cnt = 0
        for i in range(0, len(matlistdf)):
            if i % 2 == 1:
                deplete_array[cnt, :, step] = matlistdf.iloc[i][2:]
                cnt += 1
    nuclide = list(matlistdf.columns[2:])
    np.save('depletion_result', deplete_array)
    burnup_resume.to_csv('depletion_result.csv', sep='\t')
    depleted_mat = int(max(burnup_resume['material']))
    Ref_Burnup=np.zeros((deplete_array.shape[0],deplete_array.shape[2]))
    for k in range(0, len(interest)):
        Burn_array = np.flip(deplete_array[:, nuclide.index(interest[k]), :], 0)
        fig = plt.figure(dpi=300, figsize=[11.7, 8.3])
        plt.rc('ytick', labelsize=7)
        ax = plt.subplot(111)
        data = Burn_array
        data[data == 0] = np.nan
        im = ax.imshow(data, cmap='coolwarm')
        ax.set_aspect(0.25)
        ax.set_xticks(np.arange(0, len(stp)-1, 1)-0.5)
        ax.set_xticks(np.arange(0, len(stp)-1, 1), minor=True)
        ax.set_yticks(np.arange(1, depleted_mat+1, 1)-0.5)
        ax.set_xticklabels('')
        ax.set_xticklabels(np.array(stp[1:], dtype=int), minor=True)
        ax.set_yticklabels(np.flip(np.linspace(1, depleted_mat, depleted_mat, dtype=int)), va='bottom')
        ax.set_xlabel('days')
        ax.set_ylabel('Fuel Pebble Group')
        cbar = ax.figure.colorbar(
            im, ax=ax, format='%.1e', fraction=0.1, pad=0.02, aspect=50)
        cbar.ax.set_ylabel(
            'atom den. [$atom/10^{-24}cm^3$]', rotation=-90, va='bottom')
        cbar.ax.yaxis.label.set_size(11)
        for edge, spine in ax.spines.items():
            spine.set_visible(False)
        for i in range(0, len(stp)-1):
            for j in range(0, depleted_mat):
                if np.isnan(data[j, i]) == False:
                    text = ax.text(i, j, "{:.3e}".format(
                        data[j, i]), ha="center", va="center", color="k", fontsize=5)

        ax.grid(which="major", color="w", linestyle='-', linewidth=1)
        ax.set_title("Nuclide '"+str(interest[k])+"' atom. den [$Atom/b.cm$]")
        fig.tight_layout()
        plt.savefig('PLOT_NuclideDepletion_' +interest[k]+'.png', bbox_inches='tight')
        # plt.show()
        if interest[k]=='U235':
            Ref_Burnup=Burn_array
            Burn_array[np.isnan(Burn_array)]=0
        if interest[k]=='Xe135':
            Ref1_Burnup=Burn_array
            Burn_array[np.isnan(Burn_array)]=0
    return(burnup_resume, deplete_array, depleted_mat, Ref_Burnup,Ref1_Burnup)


def DEM_PEBBLE_Dist(pebble_disrrray, depleted_mat, stp):
    fig = plt.figure(dpi=300, figsize=[11.7, 8.3])
    ax = plt.subplot(111)
    data = np.flip(pebble_disrrray, 0)
    data[data == 0] = np.nan
    im = ax.imshow(data, cmap='copper')
    ax.set_aspect(0.25)
    ax.set_xticks(np.arange(0, len(stp)-1, 1)-0.5)
    ax.set_xticks(np.arange(0, len(stp)-1, 1), minor=True)
    ax.set_yticks(np.arange(1, depleted_mat+1, 1)-0.5)
    ax.set_xticklabels('')
    ax.set_xticklabels(np.array(stp[1:], dtype=int), minor=True)
    ax.set_yticklabels(
        np.flip(np.linspace(1, depleted_mat, depleted_mat, dtype=int)), va='bottom')
    ax.set_xlabel('days')
    ax.set_ylabel('Fuel Pebble Group')
    cbar = ax.figure.colorbar(im, ax=ax, format='%.1e',
                              fraction=0.1, pad=0.02, aspect=50)
    cbar.ax.set_ylabel('Pebble amount', rotation=-90, va='bottom')
    cbar.ax.yaxis.label.set_size(11)
    for edge, spine in ax.spines.items():
        spine.set_visible(False)
    for i in range(0, len(stp)-1):
        for j in range(0, depleted_mat):
            if np.isnan(data[j, i]) == False:
                text = ax.text(i, j, "{:.0f}".format(
                    data[j, i]), ha="center", va="center", color="w", fontsize=5)
    ax.grid(which="major", color="w", linestyle='-', linewidth=1)
    ax.set_title("Pebble inside the reactor core")
    fig.tight_layout()
    plt.savefig('PLOT_Pebblenum.png', bbox_inches='tight')
    # plt.show()

def OpenMC_BURNUP_cmap(stp,MCNPX_keff,OpenMC_keff,Pebbledf_list):
    legendre_array = np.zeros((11, 4, len(stp)))
    legendre_stdev = np.zeros((11, 4, len(stp)))
    axial = np.linspace(351.818-610, 351.818, int(611))
    axial_dist = np.zeros((3,len(axial), 4, len(stp)))
                  
    zernike_array = np.zeros((66, 4, len(stp)))
    zernike_stdev = np.zeros((66, 4, len(stp)))
    radial = np.linspace(0, 190, int(191))
    angle  = np.radians(np.linspace(0,360,360))
    r,theta= np.meshgrid(radial,angle)
    radial_dist = np.zeros((3,len(angle),len(radial),4,len(stp)))
    
    circle_array=np.zeros((6,4,len(stp)))
    circle_stdev=np.zeros((6,4,len(stp)))
    circle_dist = np.zeros((3,len(radial),4,len(stp)))
    
    for step in range(0, len(stp)):
        print('step---',step)
        legendre_tally = pd.read_csv('step_'+str(step)+'-DEPLETE/'+'legendre_result_'+str(step)+'.csv', sep='\t', index_col=[0])
        zernike_tally  = pd.read_csv('step_'+str(step)+'-DEPLETE/'+'zernike2d_result_'+str(step)+'.csv', sep='\t', index_col=[0])
        circle_tally   = pd.read_csv('step_'+str(step)+'-DEPLETE/'+'zernike_result_'+str(step)+'.csv', sep='\t', index_col=[0])
        for j in range(0, 4):
            for i in range(0, 11):
                legendre_array[i, j, step] = legendre_tally['mean'].iloc[(j+(4*i))]
                legendre_stdev[i, j, step] = legendre_tally['std. dev.'].iloc[(j+(4*i))]
            for i in range(0, 66):
                zernike_array[i, j,step] = zernike_tally['mean'].iloc[(j+(4*i))]
                zernike_stdev[i, j,step] = zernike_tally['std. dev.'].iloc[(j+(4*i))]            
            for i in range(0, 6):
                circle_array[i, j,step] = circle_tally['mean'].iloc[(j+(4*i))]
                circle_stdev[i, j,step] = circle_tally['std. dev.'].iloc[(j+(4*i))]            
            
            axial_dist[0,:,j,step] = openmc.legendre_from_expcoef(legendre_array[:, j, step],domain=(axial[0], axial[-1]))(axial)
            axial_dist[1,:,j,step] = openmc.legendre_from_expcoef(legendre_array[:, j, step]-legendre_stdev[:, j, step],domain=(axial[0], axial[-1]))(axial)
            axial_dist[2,:,j,step] = openmc.legendre_from_expcoef(legendre_array[:, j, step]+legendre_stdev[:, j, step],domain=(axial[0], axial[-1]))(axial)
  
            radial_dist[0,:,:,j,step] = openmc.Zernike(zernike_array[:, j, step], radius=190)(radial,angle)
            radial_dist[1,:,:,j,step] = openmc.Zernike(zernike_array[:, j, step]-zernike_stdev[:, j, step], radius=190)(radial,angle)
            radial_dist[2,:,:,j,step] = openmc.Zernike(zernike_array[:, j, step]+zernike_stdev[:, j, step], radius=190)(radial,angle)
    
            circle_dist[0,:,j,step] = openmc.ZernikeRadial(circle_array[:, j, step], radius=190)(radial)
            circle_dist[1,:,j,step] = openmc.ZernikeRadial(circle_array[:, j, step]-circle_stdev[:, j, step], radius=190)(radial)
            circle_dist[2,:,j,step] = openmc.ZernikeRadial(circle_array[:, j, step]+circle_stdev[:, j, step], radius=190)(radial)
            
        radial_dist[:,:,:,:,step]*=(args.Burn_power*1e6*MCNPX_keff['ave. nu'].iloc[step*2]/(1.6022e-13*MCNPX_keff['ave. q'].iloc[step*2]*OpenMC_keff[step*2,0]))
        axial_dist[:,:,:,step]*=(args.Burn_power*1e6*MCNPX_keff['ave. nu'].iloc[step*2]/(1.6022e-13*MCNPX_keff['ave. q'].iloc[step*2]*OpenMC_keff[step*2,0]))
        circle_dist[:,:,:,step]*=(args.Burn_power*1e6*MCNPX_keff['ave. nu'].iloc[step*2]/(1.6022e-13*MCNPX_keff['ave. q'].iloc[step*2]*OpenMC_keff[step*2,0]))

    axial_dist[axial_dist<=0]=0
    radial_dist[radial_dist<=0]=0
    circle_dist[circle_dist<=0]=0
    np.save('Zernike-Tally_result', radial_dist)
    np.save('Legendre-Tally_result',axial_dist)
    
    for j in range(0,radial_dist.shape[4]):
        flux=np.zeros((radial_dist.shape[1],radial_dist.shape[2]))
        for i in range(0,radial_dist.shape[3]):
            flux+=radial_dist[0,:,:,i,j]
        fig,ax=plt.subplots(figsize=[4,4],dpi=300,subplot_kw=dict(projection='polar'))
        im=ax.contourf(theta,r,flux,vmin=0,vmax=np.max(flux))
        ax.grid(color='w',alpha=0.75,linewidth=2)
        ax.set_yticks(np.linspace(radial[0],radial[-1],5))
        cbar = ax.figure.colorbar(im, ax=ax, format='%.1e', fraction=0.1, pad=0.12, aspect=50)
        cbar.ax.set_ylabel('$\phi$ [$n.cm/s$]', rotation=-90, va='bottom')
        cbar.ax.yaxis.label.set_size(9)
        ax.set_title('Neutron flux ($\Theta$ ,$r$) distribution\n' +
                     'Day-'+str(int(args.timeline[j])),fontsize=9)

    for i in range (0,axial_dist.shape[2]):
        fig=plt.figure(dpi=300, figsize=[8.7,8.7])
        ax=fig.gca(projection='3d')
        for j in range (0,axial_dist.shape[3]):
            X=np.array([j*2,j*2+0.5,j*2+1])
            Y=axial
            X,Y=np.meshgrid(X,Y)
            ax.plot_surface(X,Y,np.transpose(axial_dist[:,:,i,j]),cmap='coolwarm')
        ax.set_title('Neutron flux axial distribution\n' +
                     '{:.1e} eV'.format(legendre_tally['energy low [eV]'].iloc[i])+' to ' +
                     '{:.1e} eV'.format(legendre_tally['energy high [eV]'].iloc[i]))
        #ax.set_xlabel('days')
        ax.set_xticks([j*2+0.5 for j in range(0,axial_dist.shape[3])])
        ax.set_xticklabels([str(int(x)) for x in args.timeline[:-1]])
        ax.set_xlabel('Day')
        ax.set_ylabel('height')
        ax.set_yticks(np.linspace(axial[0],axial[-1],10))
        ax.set_zlabel('$\phi$ [$n.cm/s$]')

    for i in range (0,circle_dist.shape[2]):
        fig=plt.figure(dpi=300, figsize=[8.7,8.7])
        ax=fig.gca(projection='3d')
        for j in range (0,circle_dist.shape[3]):
            X=np.array([j*2,j*2+0.5,j*2+1])
            Y=radial
            X,Y=np.meshgrid(X,Y)
            ax.plot_surface(X,Y,np.transpose(circle_dist[:,:,i,j]),cmap='coolwarm')
        ax.set_title('Neutron flux axial distribution\n' +
                          '{:.1e} eV'.format(legendre_tally['energy low [eV]'].iloc[i])+' to ' +
                          '{:.1e} eV'.format(legendre_tally['energy high [eV]'].iloc[i]))
        #ax.set_xlabel('days')
        ax.set_xticks([j*2+0.5 for j in range(0,circle_dist.shape[3])])
        ax.set_xticklabels([str(int(x)) for x in args.timeline[:-1]])
        ax.set_ylabel('radius')
        ax.set_xlabel('Day')
        ax.set_yticks(np.linspace(radial[0],radial[-1],10))
        ax.set_zlabel('$\phi$ [$n.cm/s$]')
        ax.view_init(azim=150)
    
    for j in range (0,radial_dist.shape[4]):
        flux=np.zeros((axial_dist.shape[1],circle_dist.shape[1]))
        for i in range(0,radial_dist.shape[3]):
            flux+=np.array([axial_dist[0,:,i,j]]).T @ np.array([circle_dist[0,:,i,j]])
        fig=plt.figure(dpi=300,figsize=[12*(1/6+1/10),12*(1/2)])
        plt.rcParams.update({'font.size': 9})
            
        gs=gridspec.GridSpec(2,2,width_ratios=[1,1],height_ratios=[1,0.02])
        gs.update(wspace=0.05,hspace=0.2)

        ax1=plt.subplot(gs[0,1])
        ax1.set_title('Flux distribution\n Day = '+str(int(args.timeline[j])),fontsize=8)
        plt1=ax1.pcolor(radial, axial, flux/(args.Burn_power*1e6*MCNPX_keff['ave. nu'].iloc[j*2]/(1.6022e-13*MCNPX_keff['ave. q'].iloc[j*2]*OpenMC_keff[j*2,0])), 
                        cmap='viridis')
        ax1.set_xlabel(r'Radial Position [cm]')
        ax1.set_ylabel(r' ')
        ax1.set_xticks([50,100,150])
        ax1.set_xticklabels([50,100,150])
        ax1.set_yticks([])
        ax1.set_yticklabels([])
        ax1.set_aspect(1)            
            
        ax2=plt.subplot(gs[0,0])
        ax2.set_title('Pebble Distribution\n Day = '+str(int(args.timeline[j])),fontsize=8)
        ax2.grid(True)
        for name, Pebbledf_group in Pebbledf_list[j].groupby('label'):
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
        cbar.set_label('$\phi$ [$n.cm/s$]')
    
    return(axial_dist,radial_dist,circle_dist)
    

def OpenMC_CONTROL_cmap(MCNPX_keff,OpenMC_keff,Pebbledf_list,stp=[int(x) for x in range(0, 10)]):
    legendre_array = np.zeros((11, 4, len(stp)))
    legendre_stdev = np.zeros((11, 4, len(stp)))
    axial = np.linspace(351.818-610, 351.818, int(611))
    axial_dist = np.zeros((3,len(axial), 4, len(stp)))
                  
    zernike_array = np.zeros((66, 4, len(stp)))
    zernike_stdev = np.zeros((66, 4, len(stp)))
    radial = np.linspace(0, 190, int(191))
    angle  = np.radians(np.linspace(0,360,360))
    r,theta= np.meshgrid(radial,angle)
    radial_dist = np.zeros((3,len(angle),len(radial),4,len(stp)))
    
    circle_array=np.zeros((6,4,len(stp)))
    circle_stdev=np.zeros((6,4,len(stp)))
    circle_dist = np.zeros((3,len(radial),4,len(stp)))
    
    for step in range(0, len(stp)):
        print('step---',step)
        legendre_tally = pd.read_csv('step_'+str(step)+'-CONTROL/'+'legendre_result_0.csv', sep='\t', index_col=[0])
        zernike_tally  = pd.read_csv('step_'+str(step)+'-CONTROL/'+'zernike2d_result_0.csv', sep='\t', index_col=[0])
        circle_tally   = pd.read_csv('step_'+str(step)+'-CONTROL/'+'zernike_result_0.csv', sep='\t', index_col=[0])
 
        for j in range(0, 4):
            for i in range(0, 11):
                legendre_array[i, j, step] = legendre_tally['mean'].iloc[(j+(4*i))]
                legendre_stdev[i, j, step] = legendre_tally['std. dev.'].iloc[(j+(4*i))]
            for i in range(0, 66):
                zernike_array[i, j,step] = zernike_tally['mean'].iloc[(j+(4*i))]
                zernike_stdev[i, j,step] = zernike_tally['std. dev.'].iloc[(j+(4*i))]            
            for i in range(0, 6):
                circle_array[i, j,step] = circle_tally['mean'].iloc[(j+(4*i))]
                circle_stdev[i, j,step] = circle_tally['std. dev.'].iloc[(j+(4*i))]            
            
            axial_dist[0,:,j,step] = openmc.legendre_from_expcoef(legendre_array[:, j, step],domain=(axial[0], axial[-1]))(axial)
            axial_dist[1,:,j,step] = openmc.legendre_from_expcoef(legendre_array[:, j, step]-legendre_stdev[:, j, step],domain=(axial[0], axial[-1]))(axial)
            axial_dist[2,:,j,step] = openmc.legendre_from_expcoef(legendre_array[:, j, step]+legendre_stdev[:, j, step],domain=(axial[0], axial[-1]))(axial)
  
            radial_dist[0,:,:,j,step] = openmc.Zernike(zernike_array[:, j, step], radius=190)(radial,angle)
            radial_dist[1,:,:,j,step] = openmc.Zernike(zernike_array[:, j, step]-zernike_stdev[:, j, step], radius=190)(radial,angle)
            radial_dist[2,:,:,j,step] = openmc.Zernike(zernike_array[:, j, step]+zernike_stdev[:, j, step], radius=190)(radial,angle)
    
            circle_dist[0,:,j,step] = openmc.ZernikeRadial(circle_array[:, j, step], radius=190)(radial)
            circle_dist[1,:,j,step] = openmc.ZernikeRadial(circle_array[:, j, step]-circle_stdev[:, j, step], radius=190)(radial)
            circle_dist[2,:,j,step] = openmc.ZernikeRadial(circle_array[:, j, step]+circle_stdev[:, j, step], radius=190)(radial)

        radial_dist[:,:,:,:,step]*=(args.Burn_power*1e6*MCNPX_keff['ave. nu'].iloc[0]/(1.6022e-13*MCNPX_keff['ave. q'].iloc[0]*OpenMC_keff[step,0]))
        axial_dist[:,:,:,step]*=(args.Burn_power*1e6*MCNPX_keff['ave. nu'].iloc[0]/(1.6022e-13*MCNPX_keff['ave. q'].iloc[0]*OpenMC_keff[step,0]))
        circle_dist[:,:,:,step]*=(args.Burn_power*1e6*MCNPX_keff['ave. nu'].iloc[0]/(1.6022e-13*MCNPX_keff['ave. q'].iloc[0]*OpenMC_keff[step,0]))

    axial_dist[axial_dist<=0]=0
    radial_dist[radial_dist<=0]=0
    circle_dist[circle_dist<=0]=0

    np.save('Zernike-CTally_result', radial_dist)
    np.save('Legendre-CTally_result',axial_dist)

    for j in range(0,radial_dist.shape[4]):
        flux=np.zeros((radial_dist.shape[1],radial_dist.shape[2]))
        for i in range(0,radial_dist.shape[3]):
            flux+=radial_dist[0,:,:,i,j]
        fig,ax=plt.subplots(figsize=[4,4],dpi=300,subplot_kw=dict(projection='polar'))
        im=ax.contourf(theta,r,flux,vmin=0,vmax=np.max(flux))
        ax.grid(color='w',alpha=0.75,linewidth=2)
        ax.set_yticks(np.linspace(radial[0],radial[-1],5))
        cbar = ax.figure.colorbar(im, ax=ax, format='%.1e', fraction=0.1, pad=0.12, aspect=50)
        cbar.ax.set_ylabel('flux [$n.cm/s$]', rotation=-90, va='bottom')
        cbar.ax.yaxis.label.set_size(9)
        ax.set_title('Neutron flux ($\Theta$ ,$r$) distribution\n' +
                     'Control rod tip = {:.3f}'.format(171.2+(394.2-171.2)/9*j),fontsize=8)
                
    for i in range (0,axial_dist.shape[2]):
        fig=plt.figure(dpi=300, figsize=[8.7,8.7])
        ax=fig.gca(projection='3d')
        for j in range (0,axial_dist.shape[3]):
            X=np.array([j*2,j*2+0.5,j*2+1])
            Y=axial
            X,Y=np.meshgrid(X,Y)
            ax.plot_surface(X,Y,np.transpose(axial_dist[:,:,i,j]),cmap='coolwarm')
        ax.set_title('Neutron flux axial distribution\n' +
                     '{:.1e} eV'.format(legendre_tally['energy low [eV]'].iloc[i])+' to ' +
                     '{:.1e} eV'.format(legendre_tally['energy high [eV]'].iloc[i]))
        #ax.set_xlabel('days')
        ax.set_xticks([j*2+0.5 for j in range(0,axial_dist.shape[3])])
        ax.set_xticklabels(['{:.3f}'.format(171.2+(394.2-171.2)/9*j) for j in range(0,10)])
        ax.set_ylabel('height')
        ax.set_xlabel('CR tip [cm]')
        ax.set_yticks(np.linspace(axial[0],axial[-1],10))
        ax.set_zlabel('$\phi$ [$n.cm/s$]')

    for i in range (0,circle_dist.shape[2]):
        fig=plt.figure(dpi=300, figsize=[8.7,8.7])
        ax=fig.gca(projection='3d')
        for j in range (0,circle_dist.shape[3]):
            X=np.array([j*2,j*2+0.5,j*2+1])
            Y=radial
            X,Y=np.meshgrid(X,Y)
            ax.plot_surface(X,Y,np.transpose(circle_dist[:,:,i,j]),cmap='coolwarm')
        ax.set_title('Neutron flux axial distribution\n' +
                          '{:.1e} eV'.format(legendre_tally['energy low [eV]'].iloc[i])+' to ' +
                          '{:.1e} eV'.format(legendre_tally['energy high [eV]'].iloc[i]))
        #ax.set_xlabel('days')
        ax.set_xticks([j*2+0.5 for j in range(0,circle_dist.shape[3])])
        ax.set_xticklabels(['{:.3f}'.format(171.2+(394.2-171.2)/9*j) for j in range(0,10)])
        ax.set_ylabel('radius')
        ax.set_xlabel('CR tip [cm]')
        ax.set_yticks(np.linspace(radial[0],radial[-1],10))
        ax.set_zlabel('$\phi$ [$n.cm/s$]')
        ax.view_init(azim=150)

    for j in range (0,radial_dist.shape[4]):
        flux=np.zeros((axial_dist.shape[1],circle_dist.shape[1]))
        for i in range(0,radial_dist.shape[3]):
            flux+=np.array([axial_dist[0,:,i,j]]).T @ np.array([circle_dist[0,:,i,j]])
        fig=plt.figure(dpi=300,figsize=[12*(1/6+1/10),12*(1/2)])
        plt.rcParams.update({'font.size': 9})
            
        gs=gridspec.GridSpec(2,2,width_ratios=[1,1],height_ratios=[1,0.02])
        gs.update(wspace=0.05,hspace=0.2)

        ax1=plt.subplot(gs[0,1])
        ax1.set_title('Flux distribution\n CR tip = {:.3f}'.format(171.2+(394.2-171.2)/9*j),fontsize=9)
        plt1=ax1.pcolor(radial, axial, flux/(args.Burn_power*1e6*MCNPX_keff['ave. nu'].iloc[0]/(1.6022e-13*MCNPX_keff['ave. q'].iloc[0]*OpenMC_keff[i,0])), 
                        cmap='viridis')
        ax1.set_xlabel(r'Radial Position [cm]')
        ax1.set_ylabel(r' ')
        ax1.set_xticks([50,100,150])
        ax1.set_xticklabels([50,100,150])
        ax1.set_yticks([])
        ax1.set_yticklabels([])
        ax1.set_aspect(1)            
            
        ax2=plt.subplot(gs[0,0])
        ax2.set_title('Pebble Distribution\n CR tip = {:.3f}'.format(171.2+(394.2-171.2)/9*j),fontsize=8)
        ax2.grid(True)
        for name, Pebbledf_group in Pebbledf_list[0].groupby('label'):
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
        cbar.set_label('$\phi$ [$n.cm/s$]')

    return (axial_dist,radial_dist,circle_dist)

def func(x, a, b):
    return a * np.exp(-b *x) 

Pebbledf_list = OpenMC_FUEL_Tally(stp[:-1])
keff_list, pebble_disrrray, MCNPX_keff ,Bup= OpenMC_BURNUP_Keff(args.timeline, Pebbledf_list, args.MCNPXoutput)
burnup_resume, deplete_array, depleted_mat, Burn_array, Xe135arr = OpenMC_BURNUP_Resume(interest, args.timeline, header=os.getcwd())

def Residence_Relation(pebble_disrrray):
    MCNPX_pebbledisrray = np.flip(np.loadtxt('../MCNP_benchmark/depletion_pebble.txt',delimiter='\t'),0)
    MCNPX_burnups       = np.flip(np.load('../MCNP_benchmark/depletion_result.npy')[:,:,0].T,0)
    pebble_disrrray     = np.flip(pebble_disrrray, 0)
    n=15; m=9
    Pebble_time=[np.zeros(MCNPX_pebbledisrray.shape),np.zeros(pebble_disrrray.shape)]
    for i in range(0,len(args.timeline[1:])):
        Pebble_time[0][-(m+3*i):,i]=Pebble_time[0][-(m+3*i):,i-1]+(args.timeline[i+1]-args.timeline[i])
        Pebble_time[1][-(n+3*i):,i]=Pebble_time[1][-(n+3*i):,i-1]+(args.timeline[i+1]-args.timeline[i])
    fig,(ax1, ax2) = plt.subplots(nrows=2,figsize=[8,8],dpi=300,sharex=True,gridspec_kw={"height_ratios": (1, 1)})
    fig.tight_layout(pad=0.5)
    ax1.set_title('$U_{235}$ in atom density in Pebble\n vs Residence time', fontsize=14)

    for i in range(1,pebble_disrrray.shape[0]):
       if i>=n+1:
           col='#1f77b4';mark='o'
           if i==n+1:
               lab='Recirculation'
           else:
               lab=None
       elif i<n+1 and i%3==1:
           col='#2ca02c';mark='s'
           if i==1:
               lab='@ Radius 0~25 cm'
           else:
               lab=None
       elif i<n+1 and i%3==2:
           col='#d62728';mark='^'
           if i==2:
               lab='@ Radius 25~57.5 cm'
           else:
               lab=None
       elif i<n+1 and i%3==0:
           col='#9467bd';mark='*'
           if i==3:
               lab='@ Radius 57.5~90 cm'
           else:
               lab=None
       ax1.scatter(Pebble_time[1][-i,:][Pebble_time[1][-i,:]!=0],
                Burn_array[-i,:][Burn_array[-i,:]!=0],c=col,label=lab,marker=mark,s=25) 
    for i in range(1,MCNPX_pebbledisrray.shape[0]):
        if i>=m+1:
            col='#1f77b4' ;mark='o'
            if i==m+1:
                lab='Recirculation'
            else:
                lab=None
        elif i<m+1 and i%3==1:
            col='#2ca02c';mark='s'
            if i==1:
                lab='@ Radius 0~25 cm'
            else:
                lab=None
        elif i<m+1 and i%3==2:
            col='#d62728';mark='^'
            if i==2:
                lab='@ Radius 25~57.5 cm'
            else:
                lab=None
        elif i<m+1 and i%3==0:
            col='#9467bd';mark='*'
            if i==3:
                lab='@ Radius 57.5~90 cm'
            else:
                lab=None
            ax2.scatter(Pebble_time[0][-i,:][Pebble_time[0][-i,:]!=0],
                        MCNPX_burnups[-i,:][MCNPX_burnups[-i,:]!=0],c=col,label=lab,marker=mark,s=25)

    for i in range(0,5):
        ax1.annotate('axial-'+str(int(i+1)),(args.timeline[-1],Burn_array[-(i*3+1),-1]),
                     textcoords='offset points',va='center',xytext=(-10,0))
    for i in range(0,3):
        ax2.annotate('axial-'+str(int(i+1)),(args.timeline[-1],MCNPX_burnups[-(i*3+1),-1]),
                     textcoords='offset points',va='center',xytext=(-10,0))

    Reg1=Burn_array[:-n,:][Burn_array[:-n,:]!=0].flatten()
    Reg2=MCNPX_burnups[:-m,:][MCNPX_burnups[:-m,:]!=0].flatten()
    Dom1=Pebble_time[1][:-n,:][Pebble_time[1][:-n,:]!=0].flatten()
    Dom2=Pebble_time[0][:-m,:][Pebble_time[0][:-m,:]!=0].flatten()
    fit1=curve_fit(func,  Dom1,  Reg1,p0=[0,0.00399217])
    fit2=curve_fit(func,  Dom2,  Reg2,p0=[0,0.00399217])

    ax1.plot(np.unique(Dom1),func(np.unique(Dom1),*fit1[0]),'k--',
             label='\n{:.3e}'.format(fit1[0][0])+' x exp(-{:.3e}'.format(fit1[0][1])+'t)')
    ax2.plot(np.unique(Dom2),func(np.unique(Dom2),*fit2[0]),'k--',
             label='\n{:.3e}'.format(fit2[0][0])+' x exp(-{:.3e}'.format(fit2[0][1])+'t)')

    ax1.legend()
    ax2.legend()
    plt.xlabel('Residence time [days]')
    ax1.set_ylabel('OpenMC\n$U_{235}$ atom. den [$Atom/b.cm$]')
    ax2.set_ylabel('MCNPX\n$U_{235}$ atom. den [$Atom/b.cm$]')
    return(Dom1,Dom2,Reg1,Reg2,Pebble_time)

Dom1,Dom2,Reg1,Reg2,Pebble_time=Residence_Relation(pebble_disrrray)
'''
OpenMC_control_keff, OpenMC_reactivity,MCNPX_reactivity, MCNPX_control_keff=OpenMC_CONTROL_Keff(np.array([keff_list[0,:2],[MCNPX_keff['keff'].iloc[0],MCNPX_keff['sdev keff'].iloc[0]]]),
                                                                                                args.MCNPXoutput,c_batches=550, header=os.getcwd())

OpenMC_control_keff, OpenMC_reactivity,MCNPX_reactivity, MCNPX_control_keff=OpenMC_CONTROL_Keff(np.array([[1.13403,44e-5],[1.14482,38e-5]]),
                                                                                                args.MCNPXoutput,c_batches=550, header=os.getcwd())

DEM_PEBBLE_Dist(pebble_disrrray, depleted_mat, args.timeline)
OpenMC_FUEL_Tally(stp[:-1],MCNPX_keff,keff_list,normalize=True)

test=[MCNPX_reactivity,OpenMC_reactivity,MCNPX_control_keff,OpenMC_control_keff]

for i in range(0,len(test)):
    print(i)
    np.savetxt(str(i)+'.txt',test[i],delimiter='\t')

axial_dist,radial_dist,circle_dist=OpenMC_BURNUP_cmap(args.timeline[:-1],MCNPX_keff,keff_list,Pebbledf_list)
axial_cont,radial_cont,circle_cont=OpenMC_CONTROL_cmap(MCNPX_keff,OpenMC_control_keff,Pebbledf_list)

Pebbledf_list.append(make_pebbledf('./DEM_post/pebble_4400000.vtk'))
Pebbledf_list.append(make_pebbledf('./DEM_post/pebble_6200000.vtk'))
for i in range(0,len(Pebbledf_list)):
    print('max height = ',max(Pebbledf_list[i]['z']+3))
'''