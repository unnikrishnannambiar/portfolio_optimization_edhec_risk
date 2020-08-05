import pandas as pd
from scipy.optimize import minimize
def ann_vals(r):
    '''
    Takes a time series of asset returns.
    Returns a pandas DataFrame with columns for
    Annualized Returns,
    Annualized Volatility
    '''
    import numpy as np
    n_months = r.shape[0]
    ann_rets = ((1 + r).prod()) ** (12 / n_months) - 1 
    rets_std = r.std()
    ann_vol = rets_std * np.sqrt(12)
    return pd.DataFrame({
        'Annualized Returns': ann_rets,
        'Annualized Volatility': ann_vol
    })

def annualized_rets(r, periods_per_year):
    '''
    Takes a time series of asset returns.
    Returns the Annualized Returns.
    '''
    import numpy as np
    n_periods = r.shape[0]
    ann_rets = ((1 + r).prod()) ** (periods_per_year / n_periods) - 1 
    return ann_rets   
    

def annualized_vol(r, periods_per_year):
    '''
    Takes a time series of asset returns.
    Returns the Annualized Volatility.
    '''
    import numpy as np
    rets_std = r.std()
    ann_vol = rets_std * np.sqrt(periods_per_year)
    return ann_vol
    

def sharpe_ratio(r, riskfree_rate, periods_per_year):
    '''
    Computes the annualized Sharpe Ratio of a set of returns.
    '''
    # convert the annual risk free rate to per period
    rf_per_period = (1 + riskfree_rate)**(1 / periods_per_year) - 1
    excess_ret = r - rf_per_period
    ann_ex_ret = annualized_rets(excess_ret, periods_per_year)
    ann_vol = annualized_vol(r, periods_per_year)
    return ann_ex_ret / ann_vol

def drawdown(return_series: pd.Series):
    '''
    Takes a time series of asset returns.
    Returns a pandas DataFrame with columns for 
    Wealth Index,
    Previous Peaks, and
    Percentage Drawdown.
    '''
    wealth_index = 1000 * (1 + return_series).cumprod()
    prev_peaks = wealth_index.cummax()
    drawdowns = (wealth_index - prev_peaks)/prev_peaks
    return pd.DataFrame({
        "Wealth": wealth_index,
        "Peaks": prev_peaks,
        "Drawdown": drawdowns
    })
    
def get_ffme_returns():
    '''
    Loads the Fama-French Dataset for the returns of the Top and Bottom Quintiles by Market Cap
    '''
    me_m = pd.read_csv('data\Portfolios_Formed_on_ME_monthly_EW.csv', header = 0, index_col = 0, parse_dates = True, na_values = -99.99)
    columns = ['Lo 20', 'Hi 20']
    rets = me_m[columns]
    rets.columns = ['SmallCap', 'LargeCap']
    rets = rets / 100
    rets.index = pd.to_datetime(rets.index, format = '%Y%m').to_period('M')
    return rets

def get_hfi_returns():
    '''
    Loads and formats the EDHEC Hedge Fund Index Returns
    '''
    hfi = pd.read_csv('data\edhec-hedgefundindices.csv', header = 0, index_col = 0, parse_dates = True)
    hfi = hfi / 100
    hfi.index = hfi.index.to_period('M')
    return hfi

def get_ind_returns():
    '''
    Loads and formats the Ken French 30 Industry Portfolio Values Weighted Monthly Returns
    '''
    import pandas as pd
    import numpy as np 
    ind = pd.read_csv('data/ind30_m_vw_rets.csv', header = 0, index_col = 0, parse_dates = True) / 100
    ind.index = pd.to_datetime(ind.index, format = '%Y%m').to_period('M')
    ind.columns = ind.columns.str.rstrip()
    return ind

def get_ind_size():
    '''
    
    '''
    import pandas as pd
    import numpy as np 
    ind = pd.read_csv('data/ind30_m_size.csv', header = 0, index_col = 0, parse_dates = True) 
    ind.index = pd.to_datetime(ind.index, format = '%Y%m').to_period('M')
    ind.columns = ind.columns.str.rstrip()
    return ind

def get_ind_nfirms():
    '''
    
    '''
    import pandas as pd
    import numpy as np 
    ind = pd.read_csv('data/ind30_m_nfirms.csv', header = 0, index_col = 0, parse_dates = True) 
    ind.index = pd.to_datetime(ind.index, format = '%Y%m').to_period('M')
    ind.columns = ind.columns.str.rstrip()
    return ind

def semideviation(r):
    '''
    Returns the semideviation aka negative semideviation of r
    r must be a Series or a Dataframe
    '''
    is_negative = r < 0 # Boolean Mask
    return r[is_negative].std(ddof = 0)

def skewness(r):
    '''
    Alternative to scipy.stats.skew()
    Computes the skewness of the supplied Series or DataFrame
    Returns a float or a Series
    '''
    demeaned_r = r - r.mean()
    # use the population standard deviation, so set dof = 0
    exp = (demeaned_r ** 3).mean()
    sigma_r = r.std(ddof = 0)
    return exp / (sigma_r ** 3)

def kurtosis(r):
    '''
    Alternative to scipy.stats.kurtosis()
    Computes the kurtosis of the supplied Series or DataFrame
    Returns a float or a Series
    '''
    demeaned_r = r - r.mean()
    # use the population standard deviation, so set dof = 0
    exp = (demeaned_r ** 4).mean()
    sigma_r = r.std(ddof = 0)
    return exp / (sigma_r ** 4)

def is_normal(r, level = 0.01):
    '''
    Applies the Jaruqe - Bera test to determine if a Series is normal or not
    Test is applied at the 1% level of confidence by default
    Returns True if the hypothesis of normality is accepted, False otherwise
    '''
    import scipy.stats
    statistic, p_value = scipy.stats.jarque_bera(r)
    return p_value > level

def var_historic(r, level = 5):
    '''
    Returns the historic Value at Risk at a specified level
    i.e returns the number such that 'level' percent of the returns
    fall below that number, and the (100 - level) percent are above
    '''
    import numpy as np
    if isinstance(r, pd.DataFrame):
        return r.aggregate(var_historic, level = level)
    elif isinstance(r, pd.Series):
        return -np.percentile(r, level)
    else:
        raise TypeError("Expected r to be Series or DataFrame")

def var_gaussian(r,level = 5, modified = False):
    '''
    Returns the Parametric Gaussian VaR of a Series or a DataFrame
    If 'modified' is True, then the modified VaR is returned,
    based on Cornish - Fisher modification
    '''
    from scipy.stats import norm
    #computes the Z score with Gaussian Assumption
    z = norm.ppf(level / 100)
    if modified == True:
        #modify the Z score based on the observed skewness and kurtosis
        s = skewness(r)
        k = kurtosis(r)
        z = z + ((z ** 2) - 1) * (s / 6) + ((z ** 3) - (3 * z)) * (k - 3) / 24 - (2 * (z ** 3) - 5 * z) * (s ** 2) / 36
        
    return -(r.mean() + r.std(ddof = 0) * z)
 
def cvar_historic(r, level = 5):
    '''
    Computes the Conditional VaR of Series or DataFrame
    '''
    import numpy as np
    if isinstance(r, pd.DataFrame):
        return r.aggregate(cvar_historic, level = level)
    elif isinstance(r, pd.Series):
        is_beyond = r <= -var_historic(r, level = level)
        return -r[is_beyond].mean()
    else:
        raise TypeError("Expected r to be Series or DataFrame")
    
def portfolio_return(weights, returns):
    '''
    Weights -> Returns
    '''
    return weights.T @ returns #matrix mult

def portfolio_vol(weights, covmat):
    '''
    Weights -> Volatility
    '''
    return (weights.T @ covmat @ weights) ** 0.5 # matrix mult

def plot_ef2(n_points, er, cov, style = '.-'):
    '''
    Plots the 2 - Asset Efficient Frontier
    '''
    import numpy as np
    if er.shape[0] != 2 or er.shape[0] != 2:
        raise ValueError['plot_ef2 can only plot 2 - Asset frontiers']
    weights = [np.array([w, 1 -w]) for w in np.linspace(0, 1, n_points)]
    rets = [portfolio_return(w, er) for w in weights]
    vols = [portfolio_vol(w, cov) for w in weights]
    ef = pd.DataFrame({
        'Returns': rets,
        'Volatility': vols
    })
    return ef.plot.line(x = 'Volatility', y = 'Returns', style = style)

import numpy as np

def minimize_vol(target_return, er, cov):
    '''
    Target_return -> W
    '''
    n = er.shape[0]
    init_guess = np.repeat(1/n, n)
    bounds = ((0.0, 1.0),) * n
    return_is_target = {
        'type': 'eq',
        'args': (er,),
        'fun': lambda weights, er: target_return - portfolio_return(weights, er)
    }
    weights_sum_to_1 = {
        'type': 'eq',
        'fun': lambda weights: np.sum(weights) - 1
    }
    results = minimize(portfolio_vol, init_guess, args = (cov,), method = 'SLSQP', constraints = (return_is_target, weights_sum_to_1),bounds = bounds
    )
    return results.x

def optimal_weights(n_points, er, cov):
    '''
    -> list of weights to run the optimizer on to minimize the volatility
    '''
    target_rs = np.linspace(er.min(),er.max(), n_points)
    weights = [minimize_vol(target_return, er, cov) for target_return in target_rs]
    return weights

def gmv(cov):
    '''
    Returns weights of the Global Minimum Vol portfolio
    given covariance matrix.
    '''
    n = cov.shape[0]
    return msr(0, np.repeat(1,n), cov)

def plot_ef( n_points, er, cov, show_cml = False, riskfree_rate = 0, style = '.-', show_ew = False, show_gmv = False):
    '''
    Plots the N - Asset Efficient Frontier
    '''
    import numpy as np
    weights = optimal_weights(n_points, er, cov)
    rets = [portfolio_return(w, er) for w in weights]
    vols = [portfolio_vol(w, cov) for w in weights]
    ef = pd.DataFrame({
        'Returns': rets,
        'Volatility': vols
    })
    ax = ef.plot.line(x = 'Volatility', y = 'Returns', style = style)
    if show_ew is True:
        n = er.shape[0]
        w_ew = np.repeat(1/n, n)
        r_ew = portfolio_return(w_ew, er)
        vol_ew = portfolio_vol(w_ew, cov)
        # Display EW
        ax.plot([vol_ew], [r_ew], color = 'goldenrod', markersize = 12, marker = 'o' )
    if show_gmv is True:
        w_gmv = gmv(cov)
        r_gmv = portfolio_return(w_gmv, er)
        vol_gmv = portfolio_vol(w_gmv, cov)
        # Display EW
        ax.plot([vol_gmv], [r_gmv], color = 'midnightblue', markersize = 10, marker = 'o' )
    if show_cml is True:
        ax.set_xlim(left = 0)
        rf = riskfree_rate
        # weights
        w_msr = msr(riskfree_rate, er, cov)
        # returns and volatilities
        r_msr = portfolio_return(w_msr, er)
        vol_msr = portfolio_vol(w_msr, cov)

        # Add Capital Market Line
        cml_x = [0, vol_msr]
        cml_y = [riskfree_rate, r_msr]
        ax.plot(cml_x, cml_y, color = 'green', marker = 'o', linestyle = 'dashed', markersize = 12, linewidth = 2)
    return ax

def msr(riskfree_rate, er, cov):
    '''
    Returns the Portfolio with the Maximum Sharpe Ratio
    Using the Risk Free Rate, covariance matrix and the returns.
    '''
    n = er.shape[0]
    init_guess = np.repeat(1/n, n)
    bounds = ((0.0, 1.0),) * n

    weights_sum_to_1 = {
        'type': 'eq',
        'fun': lambda weights: np.sum(weights) - 1
    }
    def neg_sharpe_ratio(weights, riskfree_rate, er, cov):
        '''
        Returns the negative of the Sharpe Ratio of a portfolio, given weights
        '''
        r = portfolio_return(weights,er)
        vol = portfolio_vol(weights, cov)
        return -(r - riskfree_rate) / vol

    results = minimize(neg_sharpe_ratio, init_guess, args = (riskfree_rate, er, cov,), method = 'SLSQP', constraints = (weights_sum_to_1),bounds = bounds
    )
    return results.x