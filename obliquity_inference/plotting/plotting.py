import matplotlib.pyplot as plt



def plot_velocity_comparison(ax,df,labels=False):
    """
    Construct a scatter plot of 'Vsini' (x-axis) and 'Vequatorial' (y-axis) using asymmetric errorbars
    whenever available.

    """
        
    count_anomalous = 0
    count_nonanomalous = 0
    
    for index,row in df.iterrows():
        kic = int(row['KIC'])
        vs = float(row['Vsini'])
        dvs = float(row['dVsini'])
        veq = float(row['Veq'])
        dveq = [float(row['dVeq_minus']),float(row['dVeq_plus'])]
        I_ul = float(row['I_ul95'])
        if (I_ul < 86.0):
            count_nonanomalous+=1
            _,caps,_ = ax.errorbar([vs],[veq],xerr=[dvs],yerr=[[dveq[0]],[dveq[1]]],color='b',fmt='s',markersize=5.3,zorder=100000,capsize=4, elinewidth=1.0)
        else:
            count_nonanomalous+=1
            if ((vs < (veq + 3 * dvs)) | (veq > (vs - 3 * max(dveq)))):
                _,caps,_ = ax.errorbar([vs],[veq],xerr=[dvs],yerr=[[dveq[0]],[dveq[1]]],color='royalblue',fmt='s',markersize=3,zorder=10000,capsize=2, elinewidth=0.6,markeredgewidth=0.4,markeredgecolor='gray')
            else:
                veq *=2
                if ((vs < (veq + 3 * dvs)) | (veq > (vs - 3 * max(dveq)))): color = 'royalblue'
                else:
                    count_anomalous+=1
                    color = 'lightcoral'
                _,caps,_ = ax.errorbar([vs],[veq],xerr=[dvs],yerr=[[dveq[0]],[dveq[1]]],color=color,fmt='^',markersize=3.5,zorder=20000,capsize=2, elinewidth=0.6,markeredgewidth=0.4,markeredgecolor=color)
                if labels:
                     if ((vs > (veq + 5 * dvs)) & (veq < (vs - 5 * max(dveq)))):
                         if ((vs > (veq + 8 * dvs)) | (veq < (vs - 8 * max(dveq)))):
                             ax.text(vs * 1.01,veq*1.05,"%i" % kic,transform=ax.transData,size=5,zorder=1000000,
                                     bbox=dict(facecolor='none', edgecolor='gray',pad=0.5))
                
                
        for cap in caps: cap.set_markeredgewidth(1)

    print "Anomalous targets:",count_anomalous
    print "Non-anomalous targets:",count_nonanomalous
    
    ax.plot([0,1000],[0,1000],ls=':',lw=2.0,color='gray')
    ax.set_xlabel(r'$V\,\sin I_*$ [km s$^{-1}$]',fontname = "Times New Roman",size=18)
    ax.set_ylabel(r'$V_{\rm eq}$ [km s$^{-1}$]',fontname = "Times New Roman",size=18)
    ax.set_xlim(0,25)
    ax.set_ylim(0,25)
    ax.set_aspect(1.0)
    
    return 


def scatter_velocities():

    return


def scatter_velocities_from_dataframe(ax, df,columns):
    
    for index,row in df.iterrows():
        kic = int(row['KIC'])
        vs = float(row['Vsini'])
        dvs = float(row['dVsini'])
        veq = float(row['Veq'])
        dveq = [float(row['dVeq_minus']),float(row['dVeq_plus'])]
        I_ul = float(row['I_ul95'])
    
    return scatter_velocities
