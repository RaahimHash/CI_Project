# CI_Project

- Point generator has schemes for generating different types of polygons
- Face Extractor uses convex hull to get the faces and their points
- Graphs contains functions for creating a face graph/dual or a vertex graph. Also fixes up any issues in vertex ordering for ease during unfolding
- Unfolder creates a tree using the face graph, points etc. Has BFS unfolder, Steepest edge unfolder and chromosome unfolder
- Unfolding Flattener -> Contains functions for taking in an unfolding tree and outputs the result of unfolding
- EvolvingPopulation has high level logic for EA
- GeneticUnfolder has EA logic for unfolding
- UnfolderComparison runs all 3 unfolding algorithms
