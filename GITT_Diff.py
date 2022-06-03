from tkinter import *
from tkinter import filedialog
from sklearn.linear_model import LinearRegression
import re
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator, MaxNLocator)
import numpy as np

########################################################################
#Constants
Li = 6.94
Ni = 58.693
Co = 58.933
Al = 26.982
O = 15.999

t = 15 * 60  # s
m = 0.00615  # g
M = 96.08145  # g/mol for NCA, density 2.5 g/cm3
Vm = M / 2.5  # # 38.43258 cm3/mol
A = 0.95  # cm2
n1=3
############################################################################



###########################################################################GUI
root = Tk()
root.title('GITT to diffusion data plotting')
# root.iconbitmap('logo_vAS_icon.ico') #E:\OneDrive - University of Missouri\Python\GUI\logo_vAS_icon.ico
root.geometry('630x250')

labelb = Label(root, text="Made by Xuan on 12/26/2020, if there is any problem please contact xuanshi@missouri.edu")
labelb.grid(row=0, column=0, columnspan=2)

##############################open files
labelb = Label(root, text="Input files")
labelb.grid(row=1, column=0)


def browse():
    global filenames
    root.filename=filedialog.askopenfilename(
        title='select files', filetypes=[("xlsx", "*.*")],
        multiple=True) #        initialdir='E:\OneDrive - University of Missouri\Battery data\Python\Ch-Dis-User_define_cycle\test\all kinds of data',
    filenames=root.filename
    my_label=Label(root, text=str(len(filenames))+" file(s) selected")
    my_label.grid(row=2,column=1, columnspan=1)


b1 = Button(root, text="Browse...", command=browse)
b1.grid(row=1, column=1)

########################### save data or plot
var1 = IntVar(value=1)
c1 = Checkbutton(root, text='Save GITT Plot', variable=var1)
c1.grid(row=3, column=0)

var4 = IntVar(value=1)
c4 = Checkbutton(root, text='Save data', variable=var4)
c4.grid(row=4, column=0)

var3 = IntVar(value=1)
c3 = Checkbutton(root, text='Save Diffusion coefficient plot', variable=var3)
c3.grid(row=3, column=1)

# e = Entry(root, width=50, borderwidth=5, state='disabled')
# e.grid(row=5, column=1, columnspan=1, padx=5, pady=10)


# def uncheck():
#     # input cycles
#     if var2.get() == 1:
#         e.config(state='normal')
#         e.delete(0, END)
#         e.insert(0, '1')
#     if var2.get() == 0:
#         e.config(state='disabled')
#         # e = Entry(root, width=50, borderwidth=5, state='disabled')
#         # e.grid(row=5,column=1,columnspan=1, padx=5, pady=10)


var2 = IntVar(value=1)
c2 = Checkbutton(root, text='Save_GITT_section', variable=var2)
c2.grid(row=4, column=1)

# input cycles

# Le = Label(root, text="Cycle number e.g. \"1,50,100\"")
# Le.grid(row=5, column=0)


def p():
    # if var2.get():
    #     global Cycles
    #     Cycles = e.get().split(',')
    #     for c in range(len(Cycles)):
    #         Cycles[c] = int(Cycles[c])
    root.quit()


button_ok = Button(root, text='     OK     ', command=p)
button_ok.grid(row=7, column=1, columnspan=1, padx=5, pady=10)

root.mainloop()
############################################################################### GUI end
###############################################################################

Save_GITT_plot=  var1.get() #False
Save_GITT_section=  var2.get() # False
Save_Linearity_plot=  var3.get() # True
Save_data= var4.get()

# filenames=[('F:/OneDrive - University of Missouri/Python/Electrochemistry/GITT/Xuan-1218-shell-air-A-GITT-0_00615.xls')]

def lin_fit(x,y):
    linreg=LinearRegression()
    x=x.reshape(-1,1)
    linreg.fit(x,y)
    y_pred= linreg.predict(x)
    k=linreg.coef_
    b=linreg.intercept_
    
    # plt.scatter(x,y)
    # plt.plot(x,y_pred, color='red')
    
    return k[0], b, x, y_pred

file_names=[]
for f in filenames:
    slash=f.rfind('/')
    file_names.append(f[slash+1::])
    cwd=f[0:slash]

for f in file_names:
    xls=pd.ExcelFile(cwd+'\\'+f) # separate sheet by index
    sheetnames=xls.sheet_names
    
    data=pd.DataFrame()
    for s in sheetnames:
        if 'Channel' in s:
            if data.empty:
                data=xls.parse(s) 
            else:
                data3=xls.parse(s)    # sometime it has multiple channel tab
                data=data.append(data3,ignore_index=True)
        if 'Statistics' in s:
            data2=xls.parse(s) # Cyclic performance sheet

    charge_v=[]
    charge_t=[]
    charge_i=[]
    discharge_v=[]
    discharge_t=[]
    discharge_i=[]
    turn=0
    
    for i in range(max(data['Data_Point'])):
        if data['Current(A)'][i] <0:
            turn=i
            break

    for i in range(turn):
            charge_v.append(data['Voltage(V)'][i])
            charge_i.append(data['Current(A)'][i])
            charge_t.append(data['Test_Time(s)'][i])
        
    for i in range(turn,max(data['Data_Point'])):
        discharge_v.append(data['Voltage(V)'][i])
        discharge_i.append(data['Current(A)'][i])
        discharge_t.append(data['Test_Time(s)'][i])


    ####################### For charging 
    v_peak_charge=[] 
    v_bottom_charge=[] 
    Et=[]
    Es=[]
    ESET=[]
    segment=[]
    segment1=[]
    v_peak_charge_t=[]
    v_peak_discharge_t=[]

    for i in range(len(charge_v)-1):
        if charge_i[i]==0 and charge_i[i+1]!=0:
            v_bottom_charge.append(charge_v[i])
            segment.append(i)
            
        if charge_i[i]!=0 and charge_i[i+1]==0:
            v_peak_charge.append(charge_v[i])
            v_peak_charge_t.append(charge_t[i])
            segment1.append(i)
        
    for i in range(len(v_bottom_charge)-1):
        Et.append(v_peak_charge[i]-v_bottom_charge[i])
        Es.append(v_bottom_charge[i+1]-v_bottom_charge[i])
        ESET.append(Es[i]/Et[i])
    
    ESET=np.array(ESET)
    
    Diff_charge=4/(3.1415*t)*(m*Vm/(M*A))**2*ESET**2  #cm^2/s
    
    # plt.yscale("log")
    # plt.scatter(v_peak_charge[0:-1],Diff_charge)
        
    ####################### For discharging  

    ####################### For discharging  
    v_peak_discharge=[] 
    v_bottom_discharge=[] 
    Et=[]
    Es=[]
    ESET=[]
    segment2=[]
    segment3=[]
    for i in range(len(discharge_v)-1):
        if discharge_i[i]==0 and discharge_i[i+1]!=0:
            v_peak_discharge.append(discharge_v[i])
            v_peak_discharge_t.append(discharge_t[i])
            segment2.append(i)

            
        if discharge_i[i]!=0 and discharge_i[i+1]==0:
            v_bottom_discharge.append(discharge_v[i])
            segment3.append(i)
        
    v_bottom_discharge.pop(0)
    for i in range(len(v_peak_discharge)-1):
        Et.append(v_peak_discharge[i]-v_bottom_discharge[i]) 
        Es.append(v_peak_discharge[i]-v_peak_discharge[i+1])
        ESET.append(Es[i]/Et[i])
    
    ESET=np.array(ESET)
    
    Diff_discharge=4/(3.1415*t)*(m*Vm/(M*A))**2*ESET**2  #cm^2/s   
    # plt.yscale("log")
    # plt.scatter(v_peak_discharge[0:-1],Diff_discharge)   

    v_peak_charge_t=np.array(v_peak_charge_t)/3600
    v_peak_discharge_t=np.array(v_peak_discharge_t)/3600
    
    
    # Save_GITT_plot=  1
    # Save_GITT_section=  1
    # Save_Linearity_plot= 1
    # Save_data= 1


    ftsz = 12
    lcs = 0  # Legend location for ch-dis
    colors = ['k', 'tab:red', 'tab:blue', 'tab:green', 'm', 'y', 'c']  # color
    fileend = re.findall('\.xls*', f)[0]
    
    df1=pd.DataFrame()
    df2=pd.DataFrame()
    df3=pd.DataFrame()
    df4=pd.DataFrame()
    df5=pd.DataFrame()
    df6=pd.DataFrame()
    df7=pd.DataFrame()
    df1['Time (h)']= data['Test_Time(s)']/3600
    df1['Potential vs Li/Li+ (V)']=data['Voltage(V)']
    df6['Time (h)']=v_peak_charge_t[0:-1]
    df6['Potential vs Li/Li+ (V)']=v_peak_charge[0:-1]
    df6['D_ch (cm2/s)']=Diff_charge
    df7['Time (h)']=v_peak_discharge_t[0:-1]
    df7['Potential vs Li/Li+ (V)']=v_peak_discharge[0:-1]
    df7['D_ch (cm2/s)']=Diff_discharge

    if Save_GITT_plot:
        fig, ax1 = plt.subplots()
        ax1.xaxis.set_minor_locator(AutoMinorLocator())
        ax1.yaxis.set_minor_locator(AutoMinorLocator())
        ax1.tick_params(which='both')
        ax1.tick_params(which='major', length=7, width=1, direction='in', top='True', right ='True')
        ax1.tick_params(which='minor', length=4, width=1, direction='in', top='True', right ='True')
        
        ax1.set_xlabel('Time (h)',fontsize=ftsz) 
        ax1.set_ylabel('Potential vs Li/Li$^+$ (V)',fontsize=ftsz)    # we already handled the x-label with ax1
        ax12 = ax1.twinx()
        ax12.set_ylabel('Diffusion coefficient (cm$^2$.s$^{-1}$)',fontsize=ftsz)
        legend=[]
        legend2=[]
        
        ax1.plot(data['Test_Time(s)']/3600,data['Voltage(V)'],color=colors[1%7])
        legend.append('GITT')
        ax1.legend(legend,loc=2,framealpha =0.5)

        ax12.scatter(v_peak_charge_t[0:-1], Diff_charge,color=colors[2%7])
        #legend2.append('D$_{ch}$')
        ax12.scatter(v_peak_discharge_t[0:-1], Diff_discharge,color=colors[2%7])
        #legend2.append('D$_{dis}$')
        ax12.set_yscale("log")
        ax12.legend(legend2,loc=1,framealpha =0.5)
    
        figname=('GITT '+f).replace(fileend,'')
        fig.savefig(cwd+'\\'+figname+'.png', dpi=300, bbox_inches='tight') 
        

        ################################################################################### D vs V
        fig6, ax6 = plt.subplots()
        ax6.xaxis.set_minor_locator(AutoMinorLocator())
        ax6.yaxis.set_minor_locator(AutoMinorLocator())
        ax6.tick_params(which='both')
        ax6.tick_params(which='major', length=7, width=1, direction='in', top='True', right ='True')
        ax6.tick_params(which='minor', length=4, width=1, direction='in', top='True', right ='True')
        
        ax6.set_xlabel('Potential vs Li/Li$^+$ (V)',fontsize=ftsz)  
        ax6.set_ylabel('Diffusion coefficient (cm$^2$.s$^{-1}$)',fontsize=ftsz)
        legend=[]

        ax6.scatter(v_peak_charge[0:-1],Diff_charge,color=colors[2%7])
        legend.append('D$_{ch}$')
        ax6.scatter(v_peak_discharge[0:-1],Diff_discharge,color=colors[3%7])
        legend.append('D$_{dis}$')
        ax6.set_yscale("log")
        ax6.legend(legend,loc=lcs,framealpha =0.5)

        figname=('D vs V '+f).replace(fileend,'')
        fig6.savefig(cwd+'\\'+figname+'.png', dpi=300, bbox_inches='tight') 
        

        ###################################################################################

    df2["Time Ch (h)"]=data.iloc[segment[n1]:segment[n1+1],1]/3600
    df2["'Potential Ch vs Li/Li+ (V)"]=data.iloc[segment[n1]:segment[n1+1],7]
    df3["Time Dis Ch (h)"]=np.array(discharge_t[segment2[n1]:segment2[n1+1]])/3600
    df3["'Potential Dis Ch vs Li/Li+ (V)"]=discharge_v[segment2[n1]:segment2[n1+1]]
    
    if Save_GITT_section:    
        fig2, ax2 = plt.subplots()
        ax2.xaxis.set_minor_locator(AutoMinorLocator())
        ax2.yaxis.set_minor_locator(AutoMinorLocator())
        ax2.tick_params(which='both')
        ax2.tick_params(which='major', length=7, width=1, direction='in', top='True', right ='True')
        ax2.tick_params(which='minor', length=4, width=1, direction='in', top='True', right ='True')
        
        ax2.set_xlabel('Time (h)',fontsize=ftsz) 
        ax2.set_ylabel('Potential vs Li/Li$^+$ (V)',fontsize=ftsz)    # we already handled the x-label with ax1
        legend=[]
        

        ax2.plot(data.iloc[segment[n1]:segment[n1+1],1]/3600,data.iloc[segment[n1]:segment[n1+1],7],color=colors[1%7])
        legend.append('Charging')
        ax2.legend(legend,loc=lcs,framealpha =0.5)

        figname=('GITT Seg-Ch '+f).replace(fileend,'')
        fig2.savefig(cwd+'\\'+figname+'.png', dpi=300, bbox_inches='tight') 

        ###############################Discharging
        if segment2:
            fig3, ax3 = plt.subplots()
            ax3.xaxis.set_minor_locator(AutoMinorLocator())
            ax3.yaxis.set_minor_locator(AutoMinorLocator())
            ax3.tick_params(which='both')
            ax3.tick_params(which='major', length=7, width=1, direction='in', top='True', right ='True')
            ax3.tick_params(which='minor', length=4, width=1, direction='in', top='True', right ='True')
            
            ax3.set_xlabel('Time (h)',fontsize=ftsz) 
            ax3.set_ylabel('Potential vs Li/Li$^+$ (V)',fontsize=ftsz)    # we already handled the x-label with ax1
            legend=[]
            
    
            ax3.plot(np.array(discharge_t[segment2[n1]:segment2[n1+1]])/3600,discharge_v[segment2[n1]:segment2[n1+1]],color=colors[2%7])
            legend.append('Disharging')
            ax3.legend(legend,loc=lcs,framealpha =0.5)
    
            figname=('GITT Seg-Dis-Ch '+f).replace(fileend,'')
            fig3.savefig(cwd+'\\'+figname+'.png', dpi=300, bbox_inches='tight') 

    df4['Time^0.5 (s^0.5)']=(data.iloc[segment[n1]:segment1[n1],1]-data.iloc[segment[n1],1])**0.5
    df4['Potential vs Li/Li+ (V)']=data.iloc[segment[n1]:segment1[n1],7]
    df5['Time^0.5 (s^0.5)']= (np.array(discharge_t[segment3[n1]:segment2[n1]])-discharge_t[segment3[n1]])**0.5
    df5['Potential vs Li/Li+ (V)']=discharge_v[segment3[n1]:segment2[n1]]
    
    if Save_Linearity_plot:
        ######################################Charging
        fig4, ax4 = plt.subplots()
        ax4.xaxis.set_minor_locator(AutoMinorLocator())
        ax4.yaxis.set_minor_locator(AutoMinorLocator())
        ax4.tick_params(which='both')
        ax4.tick_params(which='major', length=7, width=1, direction='in', top='True', right ='True')
        ax4.tick_params(which='minor', length=4, width=1, direction='in', top='True', right ='True')
        
        ax4.set_xlabel('Time$^{0.5}$ (s$^{0.5}$)',fontsize=ftsz) 
        ax4.set_ylabel('Potential vs Li/Li$^+$ (V)',fontsize=ftsz)    # we already handled the x-label with ax1
        legend=[]
        
        k1,b1,x1,y1=lin_fit(np.array(df4.iloc[:,0]),df4.iloc[:,1])
        df4['V fit (V)']=y1

        ax4.scatter((data.iloc[segment[n1]:segment1[n1],1]-data.iloc[segment[3],1])**0.5,data.iloc[segment[n1]:segment1[n1],7],color=colors[1%7])
        ax4.plot(x1,y1,color=colors[1%7])
        legend.append('Charging')
        ax4.legend(legend,loc=lcs,framealpha =0.5)

        figname=('GITT CH Linearity'+f).replace(fileend,'')
        fig4.savefig(cwd+'\\'+figname+'.png', dpi=300, bbox_inches='tight') 

    #     # ######################################Discharging
    #     # fig5, ax5 = plt.subplots()
    #     # ax5.xaxis.set_minor_locator(AutoMinorLocator())
    #     # ax5.yaxis.set_minor_locator(AutoMinorLocator())
    #     # ax5.tick_params(which='both')
    #     # ax5.tick_params(which='major', length=7, width=1, direction='in', top='True', right ='True')
    #     # ax5.tick_params(which='minor', length=4, width=1, direction='in', top='True', right ='True')
        
    #     # ax5.set_xlabel('Time$^{0.5}$ (s$^{0.5}$)',fontsize=ftsz) 
    #     # ax5.set_ylabel('Potential vs Li/Li$^+$ (V)',fontsize=ftsz)    # we already handled the x-label with ax1
    #     # legend=[]
        
    #     # k2,b2,x2,y2=lin_fit(np.array(df5.iloc[:,0]),df5.iloc[:,1])
    #     # df5['V fit (V)']=y2

    #     # ax5.scatter(df5.iloc[:,0],df5.iloc[:,1],color=colors[2%7])
    #     # ax5.plot(x2,y2,color=colors[2%7])
    #     # legend.append('Discharging')
    #     # ax5.legend(legend,loc=lcs,framealpha =0.5)

    #     # figname=('GITT Disch Linearity'+f).replace(fileend,'')
    #     # fig5.savefig(cwd+'\\'+figname+'.png', dpi=300, bbox_inches='tight') 


    if Save_data:
        dfs={'GITT':df1,'Charge_seg':df2,'Dis_Charge_seg':df3,'Ch_linearity':df4,'Dis_linearity':df5 ,'Ch_D': df6, 'Dis_D':df7}
        writer=pd.ExcelWriter(cwd+'\\'+'Data '+f.replace(fileend,'')+'.xlsx', engine='xlsxwriter')
        for name in dfs.keys():
            dfs[name].to_excel(writer, sheet_name=name, index=False)
        writer.save()





