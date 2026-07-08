import pyvista as pv
import numpy as np

def generate_synthetic_mri_data(file_path):
    print(f"Loading Patient Geometry: {file_path}...")
    mesh = pv.read(file_path)
    num_points = mesh.n_points
    coords = mesh.points
    
    # -------------------------------------------------------------------------
    # 1. Regenerate Ground Truth Stiffness (From Section 4.4 Point Selection)
    # -------------------------------------------------------------------------
    E_baseline = 1.5  # Healthy wall (MPa)
    E_weak = 0.5      # Aneurysm dome weak spot (MPa)
    E_true = np.ones(num_points) * E_baseline
    
    # Automatically locate the aneurysm dome using the geometry bounds 
    # (Approximating the central bulbous region along the vessel axis)
    center_x = np.mean(coords[:, 0])
    center_y = np.mean(coords[:, 1])
    # Target nodes near the localized neck/dome assembly
    distances_to_dome = np.linalg.norm(coords - np.array([center_x, center_y, coords[:,2].max() - 10]), axis=1)
    
    radius_of_effect = 6.0 # mm
    for i in range(num_points):
        decay = np.exp(-(distances_to_dome[i]**2) / (2 * radius_of_effect**2))
        E_true[i] = E_baseline - (E_baseline - E_weak) * decay
        
    mesh["E_true (MPa)"] = E_true

    # -------------------------------------------------------------------------
    # 2. Simulate Forward Displacement Field (d_sim)
    # -------------------------------------------------------------------------
    print("Computing surface normals and generating forward displacement field...")
    
    # Compute surface normal vectors at each node
    mesh_with_normals = mesh.compute_normals(point_normals=True, cell_normals=False, flip_normals=False)
    normals = mesh_with_normals.point_data["Normals"]
    
    # Simulate a systolic pressure pulse acting normally to the wall (e.g., delta_P = 40 mmHg = 0.0053 MPa)
    delta_P = 0.0053 
    
    # Displacement magnitude is inversely proportional to stiffness: d = (delta_P * R) / E
    # Let's assume an effective radius multiplier to achieve physiological displacements (~0.3 mm)
    displacement_magnitude = (delta_P * 40.0) / E_true
    
    # Construct 3D displacement vector field (d_sim = magnitude * normal_vector)
    d_sim = np.zeros((num_points, 3))
    for i in range(num_points):
        d_sim[i, :] = displacement_magnitude[i] * normals[i, :]
        
    mesh["d_sim_magnitude (mm)"] = displacement_magnitude
    mesh["d_sim_vectors"] = d_sim

    # -------------------------------------------------------------------------
    # 3. Add Gaussian Measurement Noise to Simulate 4D-Flow MRI (d_exp)
    # -------------------------------------------------------------------------
    print("Applying stochastic noise operator (Lemma 9)...")
    
    # Target a 10% noise level relative to the mean peak displacement
    noise_percentage = 0.10
    mean_displacement = np.mean(displacement_magnitude)
    sigma = mean_displacement * noise_percentage
    
    # Generate zero-mean white Gaussian noise for each coordinate (X, Y, Z)
    gaussian_noise = np.random.normal(0, sigma, size=(num_points, 3))
    
    # Noisy observational data: d_exp = d_sim + noise
    d_exp = d_sim + gaussian_noise
    d_exp_magnitude = np.linalg.norm(d_exp, axis=1)
    
    mesh["d_exp_magnitude (mm)"] = d_exp_magnitude
    mesh["d_exp_vectors"] = d_exp
    
    # Calculate achieved Signal-to-Noise Ratio (SNR) for the text block
    signal_power = np.mean(displacement_magnitude**2)
    noise_power = sigma**2
    snr_db = 10 * np.log10(signal_power / noise_power)
    
    print("\n--- Signal-to-Noise Data Report ---")
    print(f"Assigned Noise Standard Deviation (Sigma): {sigma:.4f} mm")
    print(f"Calculated Observational SNR: {snr_db:.2f} dB")

   # -------------------------------------------------------------------------
    # 4. Comparative Visualizations for Manuscript Figures
    # -------------------------------------------------------------------------
    print("\nRendering comparative domain plots...")
    
    # Dynamically calculate the minimum and maximum displacement 
    # to stretch the colormap across the entire available spectrum
    min_disp = np.min(displacement_magnitude)
    max_disp = np.max(displacement_magnitude)
    
    plotter = pv.Plotter(shape=(1, 2)) # Side-by-side subplot layout
    
    # Subplot 1: Clean Forward Simulation (d_sim)
    plotter.subplot(0, 0)
    plotter.add_mesh(mesh, scalars="d_sim_magnitude (mm)", cmap="turbo", 
                     clim=[min_disp, max_disp], # <-- DYNAMIC BOUNDS APPLIED HERE
                     show_edges=False, scalar_bar_args={"title": "True Displacement d_sim (mm)"})
    plotter.add_text("A: Forward FSI Continuum Field", font_size=10)
    
    # Subplot 2: Noisy Clinical 4D-Flow MRI (d_exp)
    plotter.subplot(0, 1)
    plotter.add_mesh(mesh, scalars="d_exp_magnitude (mm)", cmap="turbo", 
                     clim=[min_disp, max_disp], # <-- DYNAMIC BOUNDS APPLIED HERE
                     show_edges=False, scalar_bar_args={"title": "Noisy Observation d_exp (mm)"})
    plotter.add_text("B: Simulated Noisy 4D-Flow MRI Data", font_size=10)
    
    # Link the cameras so both panels rotate together
    plotter.link_views()
    
    # Save the prepared synthetic data for the inverse solver before rendering
    mesh.save('synthetic_mri_phantom.vtk')
    print("\nDataset successfully frozen and saved as 'synthetic_mri_phantom.vtk'")
    
    plotter.show(cpos="xy")



# Execute verification pipeline
# Link the cameras so both panels rotate together
generate_synthetic_mri_data('model.vtp')