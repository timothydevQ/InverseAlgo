import pyvista as pv
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def run_inverse_reconstruction(file_path):
    print(f"Loading Observational Data: {file_path}...")
    mesh = pv.read(file_path)
    num_points = mesh.n_points
    
    
    d_exp_magnitude = mesh["d_exp_magnitude (mm)"]
    
    
    E_true = mesh["E_true (MPa)"]
    

    delta_P = 0.0053
    alpha = 1e-4 
    C = delta_P * 40.0 
    
    def forward_model(E_guess):
        return C / E_guess
    
    def cost_functional(E_guess):
        d_sim = forward_model(E_guess)
        data_misfit = 0.5 * np.sum((d_sim - d_exp_magnitude)**2)
        regularization = 0.5 * alpha * np.sum((E_guess - 1.5)**2)
        return data_misfit + regularization

    def exact_gradient(E_guess):
        
        d_sim = forward_model(E_guess)
       
        grad_misfit = (d_sim - d_exp_magnitude) * (-C / (E_guess**2))
       
        grad_reg = alpha * (E_guess - 1.5)
        
        return grad_misfit + grad_reg

    
    E_0 = np.ones(num_points) * 1.5
    iteration_history = [0]
    cost_history = [cost_functional(E_0)]

    def callback(E_k):
        
        current_cost = cost_functional(E_k)
        iteration_history.append(len(iteration_history))
        cost_history.append(current_cost)
        print(f"Iteration {len(iteration_history)-1} | Cost J(E): {current_cost:.6f}")

  
    print("\nInitializing L-BFGS Adjoint Optimization Loop...")
    
    bounds = [(0.2, 2.0) for _ in range(num_points)]
    
   
    result = minimize(
        fun=cost_functional,
        jac=exact_gradient, 
        x0=E_0,
        method='L-BFGS-B',
        bounds=bounds,
        callback=callback,
        options={'maxiter': 100, 'ftol': 1e-9, 'disp': True}
    )
    
    E_star = result.x
    mesh["E_reconstructed (MPa)"] = E_star
    
  
    l2_error = np.linalg.norm(E_star - E_true) / np.linalg.norm(E_true)
    print(f"\nOptimization Converged. Exit Status: {result.message}")
    print(f"Relative L2 Reconstruction Error: {l2_error * 100:.2f}%")
    
    
    plt.figure(figsize=(6, 4))
    plt.plot(iteration_history, cost_history, color='black', linewidth=2)
    plt.yscale('log')
    plt.xlabel('L-BFGS Iteration $(k)$')
    plt.ylabel('Cost Functional $\mathcal{J}(E_k)$')
    plt.title('Optimization Convergence (Stochastic Noise Field)')
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    plt.show()

    
    print("\nRendering comparative parameter fields...")
    plotter = pv.Plotter(shape=(1, 2))
    
    plotter.subplot(0, 0)
    plotter.add_mesh(mesh, scalars="E_true (MPa)", cmap="jet", clim=[0.5, 1.5],
                     show_edges=False, scalar_bar_args={"title": "Target E_true (MPa)"})
    plotter.add_text("Ground Truth (Hidden from Solver)", font_size=10)
    
    plotter.subplot(0, 1)
    plotter.add_mesh(mesh, scalars="E_reconstructed (MPa)", cmap="jet", clim=[0.5, 1.5],
                     show_edges=False, scalar_bar_args={"title": "Recovered E* (MPa)"})
    plotter.add_text(f"Reconstructed Field (L2 Error: {l2_error * 100:.2f}%)", font_size=10)
    
    plotter.link_views()
    plotter.show(cpos="xy")


run_inverse_reconstruction('synthetic_mri_phantom.vtk')
