import pyvista as pv
import numpy as np

def analyze_aneurisk_mesh(file_path):
    print(f"Loading Patient Geometry: {file_path}...")

    mesh = pv.read(file_path)
    num_points = mesh.n_points
    coords = mesh.points
    
    print("\n--- Geometric & Mesh Diagnostics ---")
    print(f"Total Nodes: {num_points}")
    print(f"Total Elements (Triangles): {mesh.n_cells}")
    print(f"Total Domain Volume: {mesh.volume:.2f} mm^3")
    print(f"Total Surface Area: {mesh.area:.2f} mm^2")

    
    E_baseline = 1.5 
    E_weak = 0.5 
    
    
    mesh["E_true (MPa)"] = np.ones(num_points) * E_baseline 
    
   
    plotter = pv.Plotter()
    
    def apply_gaussian_stiffness(point):
        
        print(f"\nAneurysm tip identified at: {point}")
        dome_center = np.array(point)
        
        
        distances = np.linalg.norm(coords - dome_center, axis=1)
        
        
        radius_of_effect = 4.0 
        E_true = np.ones(num_points) * E_baseline 
        
        for i in range(num_points):
            decay_factor = np.exp(-(distances[i]**2) / (2 * radius_of_effect**2))
            E_true[i] = E_baseline - (E_baseline - E_weak) * decay_factor
        
       
        mesh["E_true (MPa)"] = E_true
        plotter.add_mesh(mesh, scalars="E_true (MPa)", cmap="jet", 
                         clim=[0.5, 1.5], # <-- ADD THIS LINE
                         show_edges=True, edge_color="black", line_width=0.5,
                         scalar_bar_args={"title": "Ground Truth Stiffness (MPa)"},
                         name="vessel")
    

    plotter.add_mesh(mesh, scalars="E_true (MPa)", cmap="jet", 
                     clim=[0.5, 1.5], # <-- ADD THIS LINE
                     show_edges=True, edge_color="black", line_width=0.5,
                     scalar_bar_args={"title": "Ground Truth Stiffness (MPa)"},
                     name="vessel")
    
    plotter.add_text("AneuriskWeb Patient Geometry\nFluid-Solid Interface $\\Gamma_0$\n\nACTION: Click exactly on the tip of the aneurysm!", font_size=12)
    
    
    plotter.enable_surface_point_picking(callback=apply_gaussian_stiffness, show_message=False)
    
    print("\n--- INTERACTIVE MODE ---")
    print("The 3D window will open.")
    print("1. Rotate the vessel until you have a clear view of the aneurysm dome.")
    print("2. Left-click exactly on the tip of the aneurysm.")
    print("3. The Gaussian decay field will instantly generate around your click.")
    
    plotter.show(cpos="xy")


analyze_aneurisk_mesh('model.vtp')
