# CI_Project

- Point generator has schemes for generating different types of polygons
- Face Extractor uses convex hull to get the faces and their points
- Graphs contains functions for creating a face graph/dual or a vertex graph. Also fixes up any issues in vertex ordering for ease during unfolding
- Unfolder creates a tree using the face graph, points etc. Currently has BFS unfolder. Should also be able to support Steepest edge unfolder
- Unfolding Flattener -> takes in an unfolding tree and outputs the result of unfolding
