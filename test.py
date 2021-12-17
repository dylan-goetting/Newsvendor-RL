from scipy.stats import expon

a = expon(loc=30, scale=5)
print(a.cdf(35.5))
print(a.std())