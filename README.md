<p align="center"><a href="https://www.globalaihub.com/" target="_blank"><img src="https://cdn.prod.website-files.com/672899d4a3a2c262d4b15fb6/6728ce7f9cf8e02920fa7522_logo-p-500.png" width="300" alt="Global AI Hub Logo"></a></p>

<p align="center"><a href="https://10million.ai/akbank-python-yapay-zekaya-giris-bootcamp-yeni-nesil-proje-kampi-basliyor/" target="_blank"><img src="https://10million.ai/wp-content/uploads/2025/01/1-1.png" width="800" alt="Akbank Python ve Yapay Zekaya Giriş Bootcamp"></a></p>

# Driverless-Metro-Simulation-Route-Optimization-
# Global AI Hub Akbank Python Artificial Intelligence Introduction Boot Camp Project

## About the Project
This project is the "Driverless Metro Simulation" developed as part of the Akbank Python and Introduction to Artificial Intelligence Bootcamp organized by Global AI Hub Turkey. The work aims to determine the most optimal routes between stations in a metro network using graph data structures and various search algorithms.

## Project Objective
This project aims to achieve the following goals:

Modeling the metro network using graph data structures
Finding the route with minimum transfers using BFS (Breadth-First Search) algorithm
Finding the fastest route using A* algorithm
Solving real-world problems with algorithmic thinking

## Features
Finding the route requiring the minimum number of transfers between two stations
Calculating the fastest route (minimum travel time) between two stations
Determining stations where transfers are required
Supporting bidirectional travel between stations
Detailed route information (transfer points, total duration, etc.)

## Technologies and Libraries Used
The following Python libraries were used in the project:

collections: For data structures (defaultdict, deque)
heapq: For priority queue used in A* algorithm
typing: Type hints to improve code readability
itertools: For the count function used as a tiebreaker in priority queue
pandas: To display and format output data (optional)


## Theoretical Background
### Graph Representation
The metro network is modeled as a weighted and undirected graph:

Stations are represented as nodes
Connections between stations are represented as edges
Travel times between stations are represented as edge weights
Line changes are handled by tracking which line each station belongs to

### Breadth-First Search (BFS) Algorithm
BFS is used to find routes requiring minimum line changes. The algorithm:

Starts from the source station and explores all neighboring stations
For each neighbor, checks if a line change is required
Maintains a queue of paths to be explored and prioritizes paths requiring fewer transfers
Continues until the destination station is reached or all possibilities are exhausted

Time Complexity: O(V + E) (V: number of vertices/stations, E: number of edges/connections)
### A* Search Algorithm
A* is used to find the fastest route (minimum travel time) between stations. The algorithm:

Uses a priority queue to explore paths, prioritizing those likely to lead to the best solution
Uses a heuristic function (pre-calculated minimum travel times) to guide the search
Balances the actual travel time so far with the estimated remaining time
Guarantees an optimal solution when an admissible heuristic is used

Time Complexity : O(E log V) (E: number of edges, V: number of vertices)
### Floyd-Warshall Algorithm
Floyd-Warshall is used to pre-calculate the minimum travel times between all pairs of stations and serves as a heuristic function for A* search. The algorithm:

Calculates the shortest paths between all pairs of nodes in a weighted graph
Works by progressively improving the estimate of the shortest path between nodes
Produces a matrix of minimum distances between all pairs of stations

Time Complexity: O(V³) (V: number of vertices/stations)
## Code Structure
### Station Class
Represents individual metro stations with the following properties:

Unique identifier (id)
Station name
Line information
List of neighboring stations and travel times

### MetroNetwork Class
Manages the entire metro network with the following methods:

Adding stations and connections
Finding routes with minimum transfers
Finding fastest routes
Calculating minimum distances between all stations

### Helper Functions

find_transfer_stations: Determines stations where passengers need to change lines


### Example Metro Network
The included example creates a metro network with the following features:

3 metro lines (Red Line, Blue Line, Orange Line)
12 stations with strategic transfer points
Realistic travel times between stations

The example demonstrates 10 different route queries showing both the minimum transfer and fastest route algorithms.
#### Test Scenarios
The code includes 10 different test scenarios:

From AŞTİ to OSB
From Batıkent to Keçiören
From Keçiören to AŞTİ
From OSB to AŞTİ
From AŞTİ to Keçiören
From AŞTİ to Batıkent
From Keçiören to OSB
From Keçiören to Ulus
From Batıkent to Sıhhiye
From OSB to Sıhhiye

For each scenario, both minimum transfer and fastest route results are displayed.
### Project Development Opportunities
The project can be extended in the following ways:

Time-dependent routing (rush hours, weekends)
Service disruption and alternative routing support
Visualization of the network and routes
Real-time data integration
Waiting times at stations
Multi-modal transportation support (bus, tram, walking)

### Performance Evaluations
Points that need optimization for larger networks with hundreds of stations:

Using more efficient algorithms for sparse networks instead of Floyd-Warshall
Implementing memory-efficient path tracking in BFS
Caching frequently requested routes
Optimizing data structures for faster search and transition

## Thank You
Global AI Hub Turkey - Akbank Python and Introduction to Artificial Intelligence Bootcamp


This project was developed as part of the Akbank Python and Introduction to Artificial Intelligence Bootcamp organized by Global AI Hub Turkey. The bootcamp aims to help participants gain basic skills in Python programming and artificial intelligence.

### Contact
Mazlum Davut CELIK
   [linkedln](https://www.linkedin.com/in/mazlumdavutcelik/)
   [Medium](https://medium.com/@mazlumdavutcelik)
