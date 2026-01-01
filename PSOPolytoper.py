import numpy as np
import random
import polytope_face_extractor
import graphs
import GeneticUnfolder
import UnfoldingFlattener
import polytope_point_generator

class Particle:
    def __init__(self, num_points, bounds=(-100, 100)):
        """Initialize a particle with random 3D points."""
        self.num_points = num_points
        self.dim = num_points * 3  # 3 coordinates per point
        self.bounds = bounds
        
        # Position: flattened array of 3D points
        self.position = np.random.uniform(bounds[0], bounds[1], self.dim)
        
        # Velocity
        self.velocity = np.random.uniform(-1, 1, self.dim) * (bounds[1] - bounds[0]) * 0.1
        
        # Personal best (now maximizing, so start with -inf)
        self.pbest_position = self.position.copy()
        self.pbest_fitness = float('inf')
        
        # Current fitness
        self.fitness = float('inf')
    
    def get_points(self):
        """Convert flattened position to Nx3 array of points."""
        return self.position.reshape(-1, 3)
    
    def update_velocity(self, gbest_position, w=0.7, c1=1.5, c2=1.5):
        """Update velocity using PSO formula."""
        r1 = np.random.random(self.dim)
        r2 = np.random.random(self.dim)
        
        cognitive = c1 * r1 * (self.pbest_position - self.position)
        social = c2 * r2 * (gbest_position - self.position)
        
        self.velocity = w * self.velocity + cognitive + social
        
        # Clamp velocity
        max_vel = (self.bounds[1] - self.bounds[0]) * 0.2
        self.velocity = np.clip(self.velocity, -max_vel, max_vel)
    
    def update_position(self):
        """Update position and clamp to bounds."""
        self.position = self.position + self.velocity
        self.position = np.clip(self.position, self.bounds[0], self.bounds[1])


class PSOPolytoper:
    def __init__(self, num_particles=20, num_points=50, max_iterations=50, 
                bounds=(-100, 100), verbose=False):
        """
        Initialize PSO for polytope optimization (MAXIMIZING fitness).
        
        Args:
            num_particles: Number of particles in swarm
            num_points: Number of 3D points per particle
            max_iterations: Maximum PSO iterations
            bounds: Coordinate bounds for points
            verbose: Print progress
        """
        self.num_particles = num_particles
        self.num_points = num_points
        self.max_iterations = max_iterations
        self.bounds = bounds
        self.verbose = verbose
        
        # Initialize swarm
        self.particles = [Particle(num_points, bounds) for _ in range(num_particles)]
        
        # Global best (maximizing, so start with -inf)
        self.gbest_position = None
        self.gbest_fitness = float('inf')
        
        # History
        self.fitness_history = []
    
    def evaluate_fitness(self, particle):
        """
        Evaluate fitness of a particle (MAXIMIZING).
        Fitness = generations to 0 collisions, or 100 + remaining collisions if not reached.
        Higher fitness = harder to unfold = better.
        """
        points = particle.get_points()
        
        try:
            # Get convex hull faces
            faces, changed = polytope_face_extractor.get_conv_hull_faces(points)
            
            # Need at least 4 faces for a valid polytope
            if len(faces) < 4:
                return float('inf')  # Invalid polytope penalty (worst fitness for maximization)
            
            # Build graphs
            G_f = graphs.make_face_graph(faces)
            faces = graphs.fix_face_orientation(G_f, faces)
            
            # Run genetic unfolder with limited generations
            _, final_fit = GeneticUnfolder.GeneticUnfolder(G_f, faces, points, verbose=False)
            
            return final_fit
            
        except Exception as e:
            if self.verbose:
                print(f"Error evaluating particle: {e}")
            return 0  # Error penalty (worst fitness for maximization)
    
    def optimize(self):
        """Run PSO optimization (MAXIMIZING fitness)."""
        if self.verbose:
            print("Starting PSO optimization (maximizing fitness)...")
        
        # Initial evaluation
        for i, particle in enumerate(self.particles):
            if self.verbose:
                print(f"Evaluating initial particle {i+1}/{self.num_particles}...")
            
            particle.fitness = self.evaluate_fitness(particle)
            particle.pbest_fitness = particle.fitness
            particle.pbest_position = particle.position.copy()
            
            if particle.fitness < self.gbest_fitness:
                self.gbest_fitness = particle.fitness
                self.gbest_position = particle.position.copy()
        
        self.fitness_history.append(self.gbest_fitness)
        
        if self.verbose:
            print(f"Initial best fitness: {self.gbest_fitness}")
        
        # Main loop
        for iteration in range(self.max_iterations):
            if self.verbose:
                print(f"\n--- Iteration {iteration + 1}/{self.max_iterations} ---")
            
            for i, particle in enumerate(self.particles):
                # Update velocity and position
                particle.update_velocity(self.gbest_position)
                particle.update_position()
                
                # Evaluate
                particle.fitness = self.evaluate_fitness(particle)
                
                # Update personal best (maximizing)
                if particle.fitness < particle.pbest_fitness:
                    particle.pbest_fitness = particle.fitness
                    particle.pbest_position = particle.position.copy()
                
                # Update global best (maximizing)
                if particle.fitness < self.gbest_fitness:
                    self.gbest_fitness = particle.fitness
                    self.gbest_position = particle.position.copy()
                    if self.verbose:
                        print(f"New best fitness: {self.gbest_fitness}")
            
            self.fitness_history.append(self.gbest_fitness)
            
            if self.verbose:
                print(f"Iteration {iteration + 1} - Best fitness: {self.gbest_fitness}")
        
        return self.get_best_points(), self.gbest_fitness
    
    def get_best_points(self):
        """Get best points as Nx3 array."""
        return self.gbest_position.reshape(-1, 3)


if __name__ == "__main__":
    # Run PSO optimization
    pso = PSOPolytoper(
        num_particles=20,
        num_points=3000,
        max_iterations=20,
        bounds=(-100, 100),
        verbose=True
    )
    
    best_points, best_fitness = pso.optimize()
    
    print(f"\nOptimization complete!")
    print(f"Best fitness (higher = harder to unfold): {best_fitness}")
    
    # Save results
    np.savetxt("pso_best_points.txt", best_points, fmt='%.10f')
    
    # Visualize best result
    faces, changed = polytope_face_extractor.get_conv_hull_faces(best_points)
    G_f = graphs.make_face_graph(faces)
    faces = graphs.fix_face_orientation(G_f, faces)
    
    print(f"Number of faces: {len(faces)}")
    
    # Run full genetic unfolder on best result
    T_f, _ = GeneticUnfolder.GeneticUnfolder(G_f, faces, best_points, verbose=True)
    polygons = UnfoldingFlattener.flatten_poly(T_f, best_points)
    collisions = UnfoldingFlattener.SAT(polygons)
    
    print(f"Final collisions: {len(collisions)}")
    
    polytope_face_extractor.draw_polytope(best_points, faces, changed)
    UnfoldingFlattener.visualize_flat_faces(polygons, collisions)
