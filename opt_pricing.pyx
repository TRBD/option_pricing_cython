import numpy as np
cimport numpy as np
cimport cython

cdef extern from "math.h":
    double exp(double x)
    double pow(double base, double exponent)
    double sqrt(double x)
    double max(double x, double y)


@cython.boundscheck(False)   
def option_price_american_binomial(float flag,
                                        float S,
                                        float X,
                                        float r,
                                        float sigma,
                                        float t,
                                        int steps):
    cdef int cstep 
    cdef int x
    cdef int step
    cdef int i
    cdef float R = exp(r*(t/steps))
    cdef float Rinv = 1.0/R
    cdef float u = exp(sigma * sqrt(t / steps))
    cdef float uu = u * u
    cdef float d = 1.0/u
    cdef float p_up = (R - d) / (u - d)
    cdef float p_down = 1-p_up
    cdef np.ndarray[np.double_t, ndim=1] prices
    cdef np.ndarray[np.double_t, ndim=1] option_values
    prices = np.zeros(steps + 1, dtype=np.double)
    option_values = np.zeros(steps + 1, dtype=np.double)
    prices[0] = S * pow(d, steps)
    for i in range(1, steps + 1):
        prices[i] = uu * prices[i-1]
    for i in range(steps+1):
        option_values[i] = max(0., flag * (prices[i]-X))
    for step in range(steps-1, -1, -1):
        cstep = step
        for i in range(cstep+1):
            option_values[i] = (p_up * option_values[i+1] + p_down * option_values[i])*Rinv
            prices[i] = d * prices[i+1]
            option_values[i] = max(option_values[i], flag*(prices[i]-X))
    return option_values[0]
    
@cython.boundscheck(False)   
def option_price_american_discrete_divs(float flag,
                                        float S,
                                        float X,
                                        float r,
                                        float sigma,
                                        float t,
                                        int steps,
                                        np.ndarray[np.double_t, ndim=1] div_times,
                                        np.ndarray[np.double_t, ndim=1] div_amts):
    cdef int n_dividends = div_times.shape[0]
    if n_dividends == 0:
        return option_price_american_binomial(flag, S, X, r, sigma, t, steps)
    cdef int steps_before = <int> (steps*(div_times[0]/t))
    if steps_before < 0:    
        steps_before = 0
    if steps_before > steps:
        steps_before = steps-1
    cdef double value_alive
    cdef int cstep
    cdef int x
    cdef int step
    cdef int i
    cdef float R = exp(r*(t/steps))
    cdef float Rinv = 1.0/R
    cdef float u = exp(sigma * sqrt(t/steps))
    cdef float uu = u * u
    cdef float d = 1.0/u
    cdef float p_up = (R-d)/(u-d)
    cdef float p_down = 1-p_up
    cdef double dividend_amount = div_amts[0]
    cdef np.ndarray[np.double_t, ndim=1] tmp_dividend_times
    cdef np.ndarray[np.double_t, ndim=1] tmp_dividend_amts
    cdef np.ndarray[np.double_t, ndim=1] prices
    cdef np.ndarray[np.double_t, ndim=1] option_values
    if n_dividends > 1:
        tmp_dividend_times = np.zeros(n_dividends-1, dtype=np.double)
        tmp_dividend_amts = np.zeros(n_dividends-1, dtype=np.double)
        for i in range(0, n_dividends-1, 1):
            tmp_dividend_times[i] = div_times[i-1]
            tmp_dividend_amts[i] = div_amts[i-1]
        prices = np.zeros(steps_before+1, dtype=np.double)
        option_values = np.zeros(steps_before+1, dtype=np.double)
        prices[0]=S*pow(d, steps_before)
        for i in range(1, steps_before+1):
            prices[i] = uu * prices[i-1]
        for i in range(steps_before+1):
            value_alive = option_price_american_discrete_divs(flag, prices[i]-dividend_amount, X, r, sigma, t-div_times[0], steps-steps_before, tmp_dividend_times,tmp_dividend_amts)
            option_values[i] = max(value_alive, flag * (prices[i]-X))
        for step in range(steps_before-1, -1, -1):
            cstep = step
            for i in range(cstep+1):
                option_values[i] = (p_up * option_values[i+1] + p_down * option_values[i])*Rinv
                prices[i] = d * prices[i+1]
                option_values[i] = max(option_values[i], flag*(prices[i]-X))
    else:       
        prices = np.zeros(steps_before+1, dtype=np.double)
        option_values = np.zeros(steps_before+1, dtype=np.double)
        prices[0]=S*pow(d, steps_before)
        for i in range(1, steps_before+1):
            prices[i] = uu * prices[i-1]
        for i in range(steps_before+1):
            value_alive = option_price_american_binomial(flag, prices[i]-dividend_amount, X, r, sigma, t-div_times[0], steps-steps_before)
            option_values[i] = max(value_alive, flag * (prices[i]-X))
        for step in range(steps_before-1, -1, -1):
            cstep = step
            for i in range(cstep+1):
                option_values[i] = (p_up * option_values[i+1] + p_down * option_values[i])*Rinv
                prices[i] = d * prices[i+1]
                option_values[i] = max(option_values[i], flag*(prices[i]-X))
    return option_values[0]    
