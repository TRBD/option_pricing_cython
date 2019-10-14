import numpy as np

def option_price_american_binomial(flag,
        s,
        k,
        r,
        sigma,
        t,
        steps):
    R = np.exp(r * (t / steps))
    rInv = 1. / R
    u = np.exp(sigma * np.sqrt(t / steps))
    uu = u * u
    d = 1. / u
    p_up = (R - d) / (u - d)
    p_down = 1.0 - p_up
    prices = np.empty(steps + 1)
    option_values = np.empty(steps + 1)
    prices[0] = s * (d**steps)
    for i in range(1, steps + 1):
        prices[i] = uu * prices[i-1]
    for i in range(steps+1):
        option_values[i] = np.max([0.0, flag*(prices[i] - k)])
    for step in range(steps-1, -1, -1):
        for i in range(step+1):
            option_values[i] = (p_up * option_values[i+1] + p_down * option_values[i]) * rInv
            prices[i] = d * prices[i+1]
            option_values[i] = np.max([option_values[i], flag*(prices[i] - k)])
    return option_values[i]

def option_price_american_discrete_divs(flag,
        s,
        k,
        r,
        sigma,
        t,
        steps,
        div_times,
        div_amounts):
    
    no_dividends = len(div_times)
    if no_dividends == 0:
        return option_price_american_binomial(flag, s, k, r, sigma, t, steps)
    steps_before_dividend = int((div_times[0] / t) * steps)
    R = np.exp(r * (t / steps))
    Rinv = 1. / R
    u = np.exp(sigma * np.sqrt(t/ steps))
    d = 1. / u    
    pUp = (R-d)/(u-d)
    pDown = 1.0 - pUp
    dividend_amount = div_amounts[0]    
    tmp_dividend_times = np.empty(no_dividends - 1)
    tmp_dividend_amounts = np.empty(no_dividends - 1)    
    for i in range(no_dividends - 1):
        tmp_dividend_times[i] = div_times[i+1]
        tmp_dividend_amounts[i] = div_amounts[i+1]
    prices = np.empty(steps_before_dividend + 1)
    option_values = np.empty(steps_before_dividend + 1)
    prices[0] = s * d ** (steps_before_dividend)
    for i in range(1, steps_before_dividend + 1):
        prices[i] = u * u * prices[i-1]
    for i in range(steps_before_dividend+1):
        value_alive = option_price_american_discrete_divs(
                flag, 
                prices[i] - dividend_amount,
                k,
                r,
                sigma,
                t - div_times[0],
                steps - steps_before_dividend,
                tmp_dividend_times,
                tmp_dividend_amounts)
        option_values[i] = np.max([value_alive, flag*(prices[i] - k)])
    for step in range(steps_before_dividend - 1, -1, -1):
        for i in range(step+1):
            prices[i] = d * prices[i+1]
            option_values[i] = (pDown * option_values[i] + pUp * option_values[i+1]) * Rinv
            option_values[i] = np.max([option_values[i], flag*(prices[i] - k)])
    
    return option_values[0]
