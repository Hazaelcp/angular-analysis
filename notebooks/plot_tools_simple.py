import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
plt.style.use(hep.style.CMS)
import sys
sys.path.insert(0,'/home/ghcp/Documentos/CINVESTAV/SEMESTRE_1/scripts')
from customStats import chi2_histogram
from plot_tools import hist_weighted
# from histos_weighted import *

def create_ratio(histo_num, histo_den):
    ratio = histo_num[0]/histo_den[0]
    err = np.hypot(histo_num[-1]/histo_num[0], histo_den[-1]/histo_den[0])*ratio
    return ratio, err
    

def get_chi2(histo_num, histo_den):
    diff = histo_num[0]-histo_den[0]
    err = np.hypot(histo_num[-1], histo_den[-1])
    pull = diff/err
    isnan = np.isnan(pull)
    return np.sum(np.power(pull[np.logical_not(isnan)],2))
    

def create_axes_for_pulls2(fig, split1 = 60, split2=80, space_between = 2):
    
    ax  = plt.subplot2grid(shape = (100,1), loc = (0,0),
                           rowspan = split1, fig = fig)
    
    axp = plt.subplot2grid(shape = (100,1), loc = (split1+space_between,0),
                           rowspan = 100-(split2+space_between), fig = fig)
    
    axp2 = plt.subplot2grid(shape = (100,1), loc = (split2+space_between,0),
                           rowspan = 100-(split2+space_between), fig = fig)
    
    axp.get_shared_x_axes().join(axp, ax)
    axp2.get_shared_x_axes().join(axp2, ax)
    
    ax.set_xticklabels([])
    axp.set_xticklabels([])
    
    return ax, axp, axp2


def create_axes_for_pulls(fig, split = 70, space_between = 2):
    ax  = plt.subplot2grid(shape = (100,1), loc = (0,0),
                           rowspan = split, fig = fig)
    axp = plt.subplot2grid(shape = (100,1), loc = (split+space_between,0),
                           rowspan = 100-(split+space_between), fig = fig)
    
    axp.get_shared_x_axes().join(axp, ax)
    
    ax.set_xticklabels([])
    return ax, axp

def double_compare_plot(denominator, 
		        numerator1, 
			numerator2, 
			weights_Den=None, 
			weights_Num1=None,
			weights_Num2=None,
			label_Den='',
			label_Num1='',
			label_Num2='',
			operation='ratio',
			hist_opts=dict(bins=50),
      pval = 'chi2', 
			density=True,
      figsize=[10,8]):

    fig = plt.figure(figsize=figsize)
    main, ax1, ax2 = create_axes_for_pulls2(fig)

    if pval == 'chi2':

      chi2_1, dof_1, pval_1 = chi2_histogram(denominator, numerator1, 
                                    weights1=weights_Den, 
                                    weights2=weights_Num1, 
                                    **hist_opts)
      label_Num1 += r'   $p_{val}$ = '+ f'{round(pval_1,3)}'

      chi2_2, dof_2, pval_2 = chi2_histogram(denominator, numerator2, 
                                      weights1=weights_Den, 
                                      weights2=weights_Num2, 
                                      **hist_opts)
      label_Num2 += r'   $p_{val}$ = '+ f'{round(pval_2,3)}'



    Histo_Den = hist_weighted(denominator,
                              **hist_opts,
                              weights=weights_Den,
                                axis=main,
                               density=density,
                              label = label_Den)

    Histo_Num1 = hist_weighted(numerator1, 
                              bins=Histo_Den[1],
                              weights=weights_Num1, 
                              axis=main, 
                              density=density,
                              label = label_Num1 ,)
    
    Histo_Num2 = hist_weighted(numerator2, 
                              bins=Histo_Den[1],
                              weights=weights_Num2, 
                              axis=main, 
                              density=density,
                              label = label_Num2)

    bin_size = Histo_Den[1][1]-Histo_Den[1][0]
    if density:
      main.set_ylabel(f'Density / {str(round(bin_size, 4))}')
    else:
      main.set_ylabel(f'Counts / {str(round(bin_size, 4))}')

    bin_mean = (Histo_Den[1][1:]+Histo_Den[1][:-1])/2
    bin_size_h = (Histo_Den[1][1:]-Histo_Den[1][:-1])/2

    ratio1 = Histo_Num1[0]/Histo_Den[0]
    ratio2 = Histo_Num2[0]/Histo_Den[0]

    error1 = ratio1*np.hypot(Histo_Num1[-1]/Histo_Num1[0], Histo_Den[-1]/Histo_Den[0])
    error2 = ratio2*np.hypot(Histo_Num2[-1]/Histo_Num2[0], Histo_Den[-1]/Histo_Den[0])

    finite_mask = np.isfinite(ratio1)
    ax1.errorbar(bin_mean[finite_mask], 
                    ratio1[finite_mask], 
                    xerr=bin_size_h[finite_mask], 
                    #label=r'$p_{val}$ = '+ f'{round(pval_1,3)}',
                    yerr=error1[finite_mask])

    finite_mask = np.isfinite(ratio2)
    ax2.errorbar(bin_mean[finite_mask], 
                    ratio2[finite_mask], 
                    xerr=bin_size_h[finite_mask], 
                    #label=r'$p_{val}$ = '+ f'{round(pval_2,3)}',
                    yerr=error2[finite_mask])

    return fig, main, ax1, ax2


def compare_plot(Data_Num,
                 Data_Den,
                 weights_Num=None,
                 weights_Den=None,
                 label_Num = '',
                 label_Den = '',
                 title='',
                 operation='ratio',
                 hist_opts=dict(bins=50), 
                 density=True,
                 opts_commons  = dict(),
                 opts_Num_plot = dict(),
                 opts_Den_plot = dict(),
                 opts_lower_plot = dict(),
                 low_ylabel='',
                 low_xlabel='',
                 low_ylim = [0,2],
                 ylim = None,
                 ks_t  = False,
                 chi2_test= True, 
                 out_dir=None,
                 out_name='',
                 axes = [None, None],
                 return_axis=False,
                 show=False,
                 return_k_val=False,
                 lower_lines=True,
                 xlim_tight=False,\
                 params_axes_for_pulls =dict( split = 70, space_between = 2),
                 ):
    """Plot two samples as histograms with same binning and evaluate the ratio of their hieghts,
    if both samples came from the distribution the ratio should be distributied uniformly
    
    Params:
    ks_t = bool, str
        If True, evaluate the weighted KS test with the complete samples
        If 'cut', evaluate the weighted KS test with a sub-sample as seen in the plot
    operation = str,
        Valid opts: ratio, difference
    """
    
    
    if all(axes):
        _main, _lower = axes
        #_main.set_title(title)
    else:
        fig = plt.figure()
        fig.suptitle(title, y=0.93)
        _main, _lower = create_axes_for_pulls(fig, **params_axes_for_pulls)
    

    
    Histo_Num = hist_weighted(Data_Num, 
                              **hist_opts,
                              weights=weights_Num, 
                              axis=_main, 
                              density=density,
                              label = label_Num,
                             **opts_Num_plot,
                             **opts_commons)
    
    Histo_Den = hist_weighted(Data_Den, 
                              bins=Histo_Num[1],
                              weights=weights_Den, 
                              axis=_main, 
                              density=density,
                              label = label_Den,
                             **opts_Den_plot,
                             **opts_commons)
    
    ## To be removed! Not really useful
    if ks_t=='cut':
        low_cut = np.max([
                    Histo_Num[1][0],
                    np.percentile(Data_Num, 0.1),
                    np.percentile(Data_Den, 0.1) ])
        upp_cut = np.min([
                    Histo_Num[1][-1],
                    np.percentile(Data_Num, 99.9),
                    np.percentile(Data_Den, 99.9) ])
        print(low_cut, upp_cut)
        """
        low_cut = Histo_Num[1][0] \
                 if Histo_Num[1][0]>np.percentile(Data_Num, 0.1) \
                 else np.percentile(Data_Num, 0.1)
        upp_cut = Histo_Num[1][-1] \
                  if Histo_Num[1][-1]<np.percentile(Data_Den, 99.9) else np.percentile(Data_Den, 99.9)"""
        ks_ = ks_test.ks_2samp_weighted(
            Data_Num[(Data_Num>=low_cut) & (Data_Num<=upp_cut)],
            Data_Den[(Data_Den>=low_cut) & (Data_Den<=upp_cut)],
            weights_Num[(Data_Num>=low_cut) & (Data_Num<=upp_cut)] \
                                                     if np.all(weights_Num) else None,
            weights_Den[(Data_Den>=low_cut) & (Data_Den<=upp_cut)] \
                                                    if np.all(weights_Den) else None,)
    
    elif ks_t:
        ks_ = ks_test.ks_2samp_weighted(
            Data_Num,    Data_Den,
            weights_Num, weights_Den )
    else:
        ks_ = None

    if chi2_test:
        chi2_res = chi2_histogram(Data_Num,    Data_Den,
                                  weights_Num, weights_Den, 
                                  ensure_positive_counts=False, 
                                  bins=Histo_Num[1])
        chi2_v, dof_v, chi2_p = chi2_res

    else:
        chi2_v, dof_v, chi2_p = None, None, None
    
    
    label_title_ks   = 'KS $p_{val}$ = '+ str(round(ks_[1], 4)) if ks_ else None
    label_title_chi2 = r'$\chi^2$ $p_{val}$ = '+ str(round(chi2_p, 4)) if chi2_p else None
    labels = [label_title_ks, label_title_chi2]
    label_title =  '\n'.join([lbl for lbl in labels if lbl])
    if label_Den or label_Num:
        if axes[0] and axes[0].get_legend():
            previous_title = axes[0].get_legend().get_title().get_text()
            if previous_title: 
                label_title = previous_title+'\n'+label_title
            
        _main.legend(frameon=True, title=label_title, fontsize=16, title_fontsize=20)
    if ylim=='zero':
        _main.set_ylim(ymin=0)
    elif ylim:
        _main.set_ylim(*ylim)
        
    bin_mean = (Histo_Num[1][1:]+Histo_Num[1][:-1])/2
    bin_size = (bin_mean[1]-bin_mean[0])/2
    r = int(abs(np.log10(bin_size)))+2
    if density:
        _main.set_ylabel(f'Density / {str(round(bin_size*2,r))[:r+1]}')
    else:
        _main.set_ylabel(f'Counts / {str(round(bin_size*2,r))[:r+1]}')
    
    if operation=='ratio':
        ratio = Histo_Num[0]/Histo_Den[0]
        ratio = np.where(np.isnan(ratio), np.inf, ratio)
        scale_sum_ratio = np.sum(Histo_Num[0])/np.sum(Histo_Den[0])
        error = ratio*np.hypot(Histo_Num[-1]/Histo_Num[0], Histo_Den[-1]/Histo_Den[0])
    
    elif operation=='difference':
        ratio = Histo_Num[0]-Histo_Den[0]
        ratio = np.where(np.isnan(ratio), np.inf, ratio)
        scale_sum_ratio = np.mean(Histo_Num[0]-Histo_Den[0])
        error = np.hypot(Histo_Num[-1], Histo_Den[-1])

    # Error bar used to handle nans and infs quite well (ignoring them), 
    # now we have to make sure they are finite numbers, at least.
    finite_mask = np.isfinite(ratio)
    _lower.errorbar(bin_mean[finite_mask], 
                    ratio[finite_mask], 
                    xerr=bin_size, 
                    yerr=error[finite_mask], 
                    **opts_lower_plot)
    
    if type(lower_lines)==str and (lower_lines.lower()=='mean' or lower_lines.lower()=='average'):
        _lower.axhline(scale_sum_ratio,     ls=':',  color='grey')
    elif lower_lines:
        _lower.axhline(0.5*scale_sum_ratio, ls='--', color='grey', alpha=0.75)
        _lower.axhline(1.5*scale_sum_ratio, ls='--', color='grey', alpha=0.75)
        _lower.axhline(scale_sum_ratio,     ls=':',  color='grey')
    _lower.set_ylabel(low_ylabel, fontsize=15, loc='center')
    _lower.set_xlabel(low_xlabel)
    
    if low_ylim and density==False:
        print(scale_sum_ratio)
        _lower.set_ylim(low_ylim[0]*scale_sum_ratio, low_ylim[1]*scale_sum_ratio)
    elif low_ylim:
        _lower.set_ylim(*low_ylim)
    
    if xlim_tight:
        _lower.set_xlim([Histo_Num[1][0], Histo_Num[1][-1]])
        _main.set_xlim([Histo_Num[1][0], Histo_Num[1][-1]])

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        plt.savefig(os.path.join(out_dir, f'{k}{out_name}.png'),
                    bbox_inches='tight',
                    dpi=100
                   )
        
    to_return = list()
    
    if return_axis:
        to_return+=[fig, (_main, _lower)]
    
    if return_k_val:
        to_return.append(ks_[1])
        
    if show:
        plt.show()
    
    if to_return:
        if len(to_return)==1: return to_return[0]
        return to_return
        

