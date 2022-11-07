import pandas as pd
import numpy as np
import numba
#S = Current stock Price
#X = Strike Price
#years = Time to maturity (months or years)
#rf = risk free interest rate
#steps = time steps
#sig = volatility 
 
# Defining the stock price 
@numba.njit()  
def path(S,rf,sig,years,steps):
    sPath =[S]
    days=[0]
    del_t = years/steps
    
    for i in range (years,steps):
        Z= np.random.normal()
        S= S* np.exp((rf-0.5*sig)*(del_t)+np.sqrt(sig)*np.sqrt(del_t)*Z)
        sPath.append(S)
        days.append(i)
    return(S)

# Monte Carlo simulation using 1000 (or any) loops to estimate option value
simulations=1000    

@numba.njit()
def monte_carlo(S,X,rf,sig,years,steps):
    value = []
    del_t = years/steps
    for i in range(simulations):
        intrinsic_value = np.maximum(0,(path(S,rf,sig,years,steps)-X))
        pv=intrinsic_value/((1+rf)**del_t)   #Discounted back to Present Value
        value.append(pv)
        optionValue=np.array(value).mean()    #average of simulated values
    print(optionValue)
