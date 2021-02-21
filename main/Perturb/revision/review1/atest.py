from utils import *
dx = 0.0001
print((exact_u(0.25, 0)-exact_u(0.25, 0-dx))/dx)
# dux = (5*x*sin((5*atan2(y, x))/3))/(3*(x^2 + y^2)^(1/6)) - (5*cos((5*atan2(y, x))/3)*(x^2 + y^2)^(5/6)*(imag(x) + real(y)))/(3*((imag(x) + real(y))^2 + (imag(y) - real(x))^2))
# duy = 