import numpy as np 
import plot_tools
import zfit
import matplotlib.pyplot as plt
import math
import SLSQPv2 as SLSQP_zfit

def get_ints(xs, ys, Level):

        '''
        xs: list of mus
        ys: List of  1-CL
        Level: Confidence Level
        
        '''
        
        diff = ys - Level
        cross = np.where(np.diff(np.sign(diff)))[0]

        x_ints = []
        y_ints = []
        status = True

        for i in cross:
            x0, x1 = xs[i], xs[i + 1]
            y0, y1 = ys[i], ys[i + 1]

            # Linear interpolation to find more accurate x where y = Level
            x_i = x0 + (Level - y0) * (x1 - x0) / (y1 - y0)
            x_ints.append(x_i)
            y_ints.append(Level)

        # In case there is only one intersection, set the first x=0
        if len(x_ints) == 1:
             x_ints.append(x_i)
             y_ints.append(Level)
             x_ints[0] = 0
             y_ints[0] = ys[0]
        elif len(x_ints) == 0:
             print('There are no intersections.')
             print('Continue executed.')
             status=False
        else:
             pass

        return x_ints, y_ints, status
         
def fit1(data, mu_r, label):
    
    # Free training with restrictions

    zfit.core.parameter.Parameter._existing_params.pop('mu'+str(label), None)
    zfit.core.parameter.Parameter._existing_params.pop('sigma'+str(label), None)
    
    obs = zfit.Space('x', limits=(0, 5))

    mu = zfit.Parameter('mu'+str(label), mu_r, mu_r-1.0, mu_r+1.0) 
    sigma = zfit.Parameter('sigma'+str(label), 1.0, 0.0, 2.0)

    gauss = zfit.pdf.Gauss(obs=obs, mu=mu, sigma=sigma)
    
    const = {'type':'ineq', 'fun': lambda x: x[1]}
    
    nll = zfit.loss.UnbinnedNLL(model=gauss, data=data) 
    SLSQP_FULL = SLSQP_zfit.SLSQP(constraints=const)
    result =  SLSQP_FULL.minimize(nll)

    hesse_dict = result.hesse()

    mubest = result.params[mu]['value']
    mu_u = hesse_dict[mu]['error']
    b_likelihood = nll.value().numpy() 

    return mubest, mu_u, b_likelihood

def fit2(data, mu_i, label):

    # Training without constraints and fixed mu
    zfit.core.parameter.Parameter._existing_params.pop('mu'+str(label), None)
    zfit.core.parameter.Parameter._existing_params.pop('sigma'+str(label), None)

    obs = zfit.Space('x', limits=(0, 5))

    mu = zfit.Parameter('mu'+str(label), mu_i, mu_i-0.00001, mu_i+0.00001, floating=False) 
    sigma = zfit.Parameter('sigma'+str(label), 1.0, 0, 2.0)

    gauss = zfit.pdf.Gauss(obs=obs, mu=mu, sigma=sigma)

    const = {'type':'ineq', 'fun': lambda x: x[1]}
    
    nll = zfit.loss.UnbinnedNLL(model=gauss, data=data) 
    SLSQP_prof = SLSQP_zfit.SLSQP()
    result =  SLSQP_prof.minimize(nll)
    
    p_likelihood = nll.value().numpy()

    #zfit.run.clear_graph_cache()  # clear cached TF graphs

    return p_likelihood

def gen_data(mu, sigma, n_events, label):

    zfit.core.parameter.Parameter._existing_params.pop('mu'+str(label), None)
    zfit.core.parameter.Parameter._existing_params.pop('sigma'+str(label), None)

    obs = zfit.Space('x', limits=(0, 5))
    mu = zfit.Parameter('mu'+str(label), mu, mu-0.001, mu+0.001) 
    sigma = zfit.Parameter('sigma'+str(label), sigma, 0, 1.0)

    gauss_pdf = zfit.pdf.Gauss(obs=obs, mu=mu, sigma=sigma)
    sampler = gauss_pdf.create_sampler(n=n_events)
    sampler.resample()

    return sampler

def Ext_KFC(data, mu, sigma, N_mus, CL, N_MC, iter_label):

    '''Function that computes the CIs for a gaussian with the extended 1-CL curves.

    Parameters
    ----------------------------
        data: list or np.array
        mu: float
        sigma: float
        N_mus: int
        CL: float
        N_MC: int
            Number of intervals
    Output
    ----------------------------
    '''

    print('Starting extended Karbach-Feldman-Cousins method for a gaussian: \n')
    # First fit
    
    hmu, hmu_u, data_best_L = fit1(data, mu, label='RD')
    
    l_inf = hmu - 3*hmu_u
    l_sup = hmu + 3*hmu_u
    
    # Verify that the generated mu list will span allowed regions only.
    if l_inf <= 0:
        l_inf = 0
        l_sup = l_sup + (hmu_u/2)
    else:
        pass
    
    m=N_mus
    step = (l_sup-l_inf)/m
    mus =  np.arange(l_inf, l_sup, step) 

    dic_OCL = []

    # ===== Process for different values of mu =====
    for z in range(len(mus)): #[0.2, 0.3, 0.4,...]

        # ===== Setting mu value for this iteration =====
        mu_i = mus[z]

        # ===== Second fit for data =====
        data_prof_L = fit2(data, mu_i, label='RD2')

        print(f'Real data values for mu = {mu_i}')
        print(f'Profile likelihood: {data_prof_L}')
        print(f'Best likelihood: {data_best_L}')

        dchi_data = data_prof_L - data_best_L    

        # ===== Generation of N_MC toy MC samples for a single value of mu =====
        # ===== with size equal to that of data                            =====
        
        N_toys = 0
        dchis = []
        for mc in range(N_MC):

            labmc = 'MC'
            labmc2 = 'MC2'

            xtoys = gen_data(mu_i, sigma, n_events=5000, label='xtoys')

            #First fit for MC
            hmut, hmut_u, mc_best_L = fit1(xtoys, mu_i, label=labmc)
            mc_prof_L = fit2(xtoys, mu_i, label=labmc2)

            dchi_mc = mc_prof_L - mc_best_L
            dchis.append(dchi_mc)

            if dchi_mc >= dchi_data:
                N_toys+=1
            else:
                pass

            del xtoys
        
            #zfit.util.cache.clear_graph_cache()

        OneminusCL = N_toys/N_MC
        #print(N_toys)
        dic_OCL.append(OneminusCL)

        #zfit.util.cache.clear_graph_cache()
    
    print('Generating the 1-CL curves') 
    Level = 1-CL
    percent = CL*100
    CL2 = 0.6827
    L2 = 1-CL2
    percent2 = CL2*100

    OmCL = np.array(dic_OCL)

    # ===== Intersections with the 1-CL curve at 1-CL confidence level

    print(f'Calculating intersections and confidence intervals for the given data.')

    x1s, y1s, st1 = get_ints(mus, OmCL, Level)
    x2s, y2s, st2 = get_ints(mus, OmCL, L2)

    if st1 == True:
        pass
    else:
        print('No intersections.')
    
    
    lim_1 = np.round(hmu - 6*hmu_u, 2)
    lim_2 = np.round(hmu + 6*hmu_u, 2)
    
    plt.figure()
    plt.plot(mus, OmCL, color='blue')
    plt.scatter(mus, OmCL, color='black')
    plt.axhline(y=Level, color='red', linestyle='--', label=f'{percent}%')
    #plt.axhline(y=L2, color='green', linestyle='--', label=f'{percent2}%')
    plt.scatter(x1s, y1s, color='red')
    #if st2 != False: plt.scatter(x2s, y2s, color='green')
    plt.vlines(x1s, 0, y1s, color='red', linestyle='--')
    #if st2!= False: plt.vlines(x2s, 0, y2s, color='green', linestyle='--')
    plt.vlines(0, 0, OmCL[0], color='black', linestyle='--')
    plt.vlines(mu, 0, 1, color='orange', linestyle='--', label='$\\mu_v$')
    plt.vlines(hmu, 0, 1, color='purple', label='$\\mu_{{obs}}$')
    plt.xlim(lim_1, lim_2)
    plt.xlabel('$\\mu$')
    plt.ylabel('1-CL')
    plt.title('$\\mu_{{obs}}=$'+str(np.round(hmu, 4)))
    plt.legend()
    plt.savefig('/eos/user/n/ntepecti/PhD/Feldman_Cousins/Test_results/Test_SLSQP/MoreMCs/Extended_1-CL-Curve_'+str(iter_label)+'_.png')
    plt.close()

    CI = [x1s[0], x1s[-1]]
    CI60 = [x2s[0], x2s[-1]]
    #CI.extend([x1s[0], x1s[-1]])
    #CI60.extend([x2s[0], x2s[-1]])
    #dic_CIs['CIs'+str(j+1)].append(CIs)
    print(f'Confidence interval of [{x1s[0]}, {x1s[-1]}] with a {percent}% confidence level.')
    if st2 != False: print(f'Confidence interval of [{x2s[0]}, {x2s[-1]}] with a {percent2}% confidence level.') 

    #CIs.append(CI)
    #CI60.append(CI60)

    print('Process terminated. Thank you for your patience.')
    
    #CI60 = [] # REMOVEEEE
    return CI, CI60


if __name__ == '__main__':

    import argparse

    my_parser = argparse.ArgumentParser(prog='Feldman Cousins demo for a Gaussian distribution with one nuisance parameter.',
                                        description='This script automatically does everything to obtain confidence intervals \
                                            and 1-CL curves for a dataset.')
    
    
    my_parser.add_argument('--mu',
                           type = float,
                           default=1.5,
                           help='Value of mu')
    my_parser.add_argument('--sigma',
                           type=float,
                           default=0.5,
                           help='Sigma value')
    my_parser.add_argument('--CL',
                           type=float, 
                           default=0.90,
                           help='Confidence level')
    my_parser.add_argument('--N_MC',
                           type=int,
                           default=50, 
                           help='Number of toy MC samples')
    my_parser.add_argument('--it_label',
                           type=int,
                           default=100,
                           help='Number of label iteration')  
    


    args  = my_parser.parse_args()


    k_alpha = args.CL
    mu_real = args.mu
    sigma_true = args.sigma
    ilabel = args.it_label
    data = gen_data(mu_real, sigma_true, 5000, 'RD')

    #plt.figure()
    #plt.hist(data, bins=30)
    #plt.title('Gen-data distribution')
    #plt.xlabel('x')
    #plt.ylabel('N events')
    #plt.savefig('/eos/user/n/ntepecti/PhD/Feldman_Cousins/Test_results/Test_SLSQP/Data_distribution_SLSQP.png')
    #plt.close()

    N_MC = args.N_MC

    CIs, CIs68 = Ext_KFC(data, mu_real, sigma_true, 40, k_alpha, N_MC, ilabel)

    with open("/eos/user/n/ntepecti/PhD/Feldman_Cousins/Test_results/Test_SLSQP/MoreMCs/Constraint_CIs.txt", "a") as f:
        f.write(f"{CIs}\n")
        
    with open("/eos/user/n/ntepecti/PhD/Feldman_Cousins/Test_results/Test_SLSQP/MoreMCs/Constraint_CIs_68.txt", "a") as f:
        f.write(f"{CIs68}\n")