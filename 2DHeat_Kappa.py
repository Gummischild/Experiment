"""Heizen und kÃ¼hlen"""
# Credits: Jan Blechta
# https://github.com/blechta/fenics-handson/blob/master/heatconv
from dolfin import *
import matplotlib.pyplot as plt
#set_log_level(30)

Start_T_Sockel = 15
Heizungsrohr_T = 25
Fenster_T =5




# Create mesh and build function space
mesh = UnitSquareMesh(60, 60, "crossed")
V = FunctionSpace(mesh, "Lagrange", 1)

# Create boundary markers
tdim = mesh.topology().dim()
boundary_parts = MeshFunction("size_t", mesh, tdim - 1)
right = AutoSubDomain(lambda x:   near(x[0]> 0.2, 0) and near(x[1]> 0.2, 0))
left = AutoSubDomain(lambda x:   near(pow(pow(x[0]-.25,2)+pow(x[1]-.75,2),0.5)> 0.1, 0) )
#right = AutoSubDomain(lambda x:  near(x[0], 0.0))
bottom = AutoSubDomain(lambda x: near(x[1], 0.0))
top = AutoSubDomain(lambda x: near(x[1], 1.0))
left.mark(boundary_parts, 1)
right.mark(boundary_parts, 2)
bottom.mark(boundary_parts, 3)
top.mark(boundary_parts, 4)
# Initial condition and right-hand side
ic = Expression("""1*pow(x[0] - 0.25, 0) + 1*pow(x[1] - 0.25, 0) < 0.2*0.2 
                   ? -25.0 * ((pow(x[0] - 0.25, 2) + pow(x[1] - 0.25, 2)) - 0.2*0.2)
                   : Start_T_Sockel""",Start_T_Sockel=Start_T_Sockel, degree=1)

# Lasten pro Zeitschritt im Beton
f1 = Expression("""pow(x[0] - 1, 2) + pow(x[1] - 1, 2) < 0.02*0.02
                  ? -20000.0 : 0.0""", degree=1)
# Lasten pro Zeitschritt in der Luft (20 mal so groß wie im Beton)
f2 = Expression("""pow(x[0] - 1, 2) + pow(x[1] - 0, 2) < 0.02*0.02
                  ?  20*20000.0 : 0.0""", degree=1)
# Equation coefficients
K = Constant(.01)  # thermal conductivity
g = Constant(10.01)  # Neumann heat flux
b = Expression(("-(x[1] - 0.5)*1", "(x[0] - 0.5)*1"), degree=1)  # convecting velocity
plot(project(b,VectorFunctionSpace(mesh, "CG", 1)))
# Define boundary measure on Neumann part of boundary
dsN = Measure("ds", subdomain_id=4, subdomain_data=boundary_parts)
## Dem dsN (Neumann-RB) kann man nicht den Kreis (subdomain_id=1) oder das Viereck (subdomain_id=2) zuweisen.


k_0 =20 #(Kappa für Luft)
k_1 =1  #(Kappa für Beton)
K= Expression('x[1] <= 0.5  ? k_0 : k_1', degree=0, k_0=k_0, k_1=k_1)



# Define steady part of the equation
def operator(u, v):
    return (K * inner(grad(u), grad(v)) 
            - f1 * v
            - f2 * v
            + dot(0*b, grad(u)) * v) * dx - K * g * v * dsN

# Define trial and test function and solution at previous time-step
u = TrialFunction(V)
v = TestFunction(V)
u0 = Function(V)

# Time-stepping parameters
dt = 0.03
theta = Constant(1)  # Crank-Nicolson scheme

# Define time discretized equation
F = ((1.0 / dt) * inner(u - u0, v) * dx
    + theta * operator(u, v)
    + (1.0 - theta) * operator(u0, v) )

# Define boundary condition
bc = (DirichletBC(V, Expression('Heizungsrohr_T',Heizungsrohr_T=Heizungsrohr_T,degree = 1), boundary_parts, 1),
      DirichletBC(V, Expression('Fenster_T',Fenster_T=Fenster_T,degree = 1), boundary_parts, 2))
# Prepare solution function and solver
u = Function(V)
problem = LinearVariationalProblem(lhs(F), rhs(F), u, bc)
solver = LinearVariationalSolver(problem)

# Prepare initial condition
u0.interpolate(ic)
u.interpolate(ic)


######################################################Time-stepping

# some script controlflow constants:
PLOT_INITIAL = True
WRITE_ALL_TIMESTEPS = True
# output file
fname = "./output/simulation.xdmf"

from vedo.dolfin import *

t = 0.0

with XDMFFile(fname) as results_file:
    results_file.parameters["flush_output"] = True
    results_file.parameters["functions_share_mesh"] = True
    results_file.write(u, -1) # store initial condition
    while t < 1:
        
        solver.solve()
     
    #    plot(u,wireframe = True,interactive=False)
        plot(u,
            text=__doc__+"\nTemperature at t = %g" % t,
           style=2,
           axes=2,
            lw=0, # no mesh edge lines
            warpZfactor=0.05,
            isolines={"n": 12, "lw":1, "c":'black', "alpha":0.1},
            wireframe = False,
            interactive=False,)
        # Move to next time step
        u0.assign(u)
        t += dt
        if WRITE_ALL_TIMESTEPS:
            results_file.write(u, t)
    if not WRITE_ALL_TIMESTEPS:   
        results_file.write(u, t)
#exportWindow('Export_Bild.x3d')        
interactive()
