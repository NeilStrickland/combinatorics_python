from sympy import *
x = symbols('x')
k = symbols('k', integer=True)

p = sum(map(lambda k : binomial(4,k) ** 2 * factorial(k) * x ** k,range(5)))

p

r = expand(p ** 13)

r

n = sum(map(lambda k : (-1) ** k * factorial(52 - k) * r.coeff(x,k),range(53)))

n

n.evalf()

