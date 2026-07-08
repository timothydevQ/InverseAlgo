import pyvista as pv
import numpy as np

def generate_synthetic_mri_data(file_path):
    print(f"Loading Patient Geometry: {file_path}...")
    mesh = pv.read(file_path)
    num_points = mesh.n_points
    coords = mesh.points
    
    
    E_baseline = 1.5 
    E_weak = 0.5      
    E_true = np.ones(num_points) * E_baseline
    
    center_x = np.mean(coords[:, 0])
    center_y = np.mean(coords[:, 1])
    
    distances_to_dome = np.linalg.norm(coords - np.array([center_x, center_y, coords[:,2].max() - 10]), axis=1)
    
    radius_of_effect = 6.0 
    for i in range(num_points):
        decay = np.exp(-(distances_to_dome[i]**2) / (2 * radius_of_effect**2))
        E_true[i] = E_baseline - (E_baseline - E_weak) * decay
        
    mesh["E_true (MPa)"] = E_true

    
    print("Computing surface normals and generating forward displacement field...")
    
    
    mesh_with_normals = mesh.compute_normals(point_normals=True, cell_normals=False, flip_normals=False)
    normals = mesh_with_normals.point_data["Normals"]
    
    
    delta_P = 0.0053 
    
    
    displacement_magnitude = (delta_P * 40.0) / E_true
    
    d_sim = np.zeros((num_points, 3))
    for i in range(num_points):
        d_sim[i, :] = displacement_magnitude[i] * normals[i, :]
        
    mesh["d_sim_magnitude (mm)"] = displacement_magnitude
    mesh["d_sim_vectors"] = d_sim

   
    print("Applying stochastic noise operator (Lemma 9)...")
    
    
    noise_percentage = 0.10
    mean_displacement = np.mean(displacement_magnitude)
    sigma = mean_displacement * noise_percentage
    
    
    gaussian_noise = np.random.normal(0, sigma, size=(num_points, 3))
    
    
    d_exp = d_sim + gaussian_noise
    d_exp_magnitude = np.linalg.norm(d_exp, axis=1)
    
    mesh["d_exp_magnitude (mm)"] = d_exp_magnitude
    mesh["d_exp_vectors"] = d_exp
    
    
    signal_power = np.mean(displacement_magnitude**2)
    noise_power = sigma**2
    snr_db = 10 * np.log10(signal_power / noise_power)
    
    print("\n--- Signal-to-Noise Data Report ---")
    print(f"Assigned Noise Standard Deviation (Sigma): {sigma:.4f} mm")
    print(f"Calculated Observational SNR: {snr_db:.2f} dB")

   
    print("\nRendering comparative domain plots...")
    
    
    min_disp = np.min(displacement_magnitude)
    max_disp = np.max(displacement_magnitude)
    
    plotter = pv.Plotter(shape=(1, 2)) 
    
   
    plotter.subplot(0, 0)
    plotter.add_mesh(mesh, scalars="d_sim_magnitude (mm)", cmap="turbo", 
                     clim=[min_disp, max_disp], 
                     show_edges=False, scalar_bar_args={"title": "True Displacement d_sim (mm)"})
    plotter.add_text("A: Forward FSI Continuum Field", font_size=10)
    
    
    plotter.subplot(0, 1)
    plotter.add_mesh(mesh, scalars="d_exp_magnitude (mm)", cmap="turbo", 
                     clim=[min_disp, max_disp], 
                     show_edges=False, scalar_bar_args={"title": "Noisy Observation d_exp (mm)"})
    plotter.add_text("B: Simulated Noisy 4D-Flow MRI Data", font_size=10)
    
    
    plotter.link_views()
    

    mesh.save('synthetic_mri_phantom.vtk')
    print("\nDataset successfully frozen and saved as 'synthetic_mri_phantom.vtk'")
    
    plotter.show(cpos="xy")




generate_synthetic_mri_data('model.vtp')
