import opt_pricing
import py_opt_pricing

def test_cython():
    opt_pricing.option_price_american_discrete_divs(1, 208.38, 210., .0255, .17, 47.5/252, 50, np.array([45/365.]), np.array([1.25]))
    
def test_py():
    py_opt_pricing.option_price_american_discrete_divs(1, 208.38, 210., .0255, .17, 47.5/252, 50, np.array([45/365.]), np.array([1.25]))
    

if __name__ == '__main__':    
    import timeit
    print(timeit.timeit('test_cython()', setup="from __main__ import test_cython", number=1000))
    print(timeit.timeit('test_py()', setup="from __main__ import test_py", number=1000))
    # 0.2127398
    # 105.758376
