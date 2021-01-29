def ddexact(x, y):
    theta = get_theta(x, y)
    duxx = -(10*(y**2*sin((5*theta)/3) - x**2*sin((5*theta)/3) + 2*x*y*cos((5*theta)/3)))/(9*(x**2 + y**2)**(7/6))
    duxy = (10*(x**2*cos((5*theta)/3) - y**2*cos((5*theta)/3) + 2*x*y*sin((5*theta)/3)))/(9*(x**2 + y**2)**(7/6))
    duyx = duxy
    duyy = (10*(y**2*sin((5*theta)/3) - x**2*sin((5*theta)/3) + 2*x*y*cos((5*theta)/3)))/(9*(x**2 + y**2)**(7/6))
    return duxx, duxy, duyx, duyy