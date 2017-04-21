
# Pacman Medium Search (Euler Path)

## Nice Experience
This is a project proposed by the class of Stanford Artifical Intelligence. Our team aims at sloving the problem of eating all dots in mediumSearch whose path is filled with dots.

The method is proposed by Jitong Qi, and thanks for his persistance, we finished this extra problem before deadline. In the process, Siyuan Wang came up with some interesing ideas and her implement is very nice!

This is an absolutely pleasure to cooperate with them!


## Code Running
You can input the following command in terminal to run our programme:

python pacman.py -l mediumSearch -p SearchAgent -a prob=MediumGraphicProblem

Runtime of the programme is about 7 seconds.

## Algorithm Introduction
### Abstract
Recall that the Euler path problem is to find a trail in a graph which visits every edge exactly once and an undirected graph has an Eulerian path if and only if exactly zero or two vertices have odd degree. Thus we abstract the maze into a graph by choosing all dots which is adjacent to 1, 3, 4 dots as graph's nodes so that the MediumSearch problem is transformed to find a path which visit each edge at least once with the lowest cost(Really?).

Obviously, the graph abstracted from Pac-man map is almost impossible to be an Eulerian graph, let alone the starting position is determined. So we need to revisit some edges, in other words, add some edges between odd vertexes with a minimal cost to change the graph into an Eulerian graph since the amount of singular nodes are even.

Our core code is all included in *class MediumGraphicProblem*. The rough steps are as follows and we will discuss their details later.

   1. Extract a graph from initial Pac-man map.
   2. Determine odd vertexes in the graph.
   3. Compute minimun distance between nodes we concern.
   4. Choose the best connect method so that the new graph is an Eulerian graph.
   5. Calculate the Eulerian path or circle of the new graph and make some adjustments to reduce the final cost.

### Abstract Graph and Determine Odd Vertexes
In this part, the jobs are to abstract a graph for Step 3(*self.Graphic*), choose odd vertexes in Step 2 and calculate some basic values for Step 4 (*self.ReGraphic*).

It's not diffcult to extract a graph from the given map by data structure *class Node* defined by ourself. While if we determine odd vertexes without any disposition, a number of odd vertices are unnecessary given that there're lots of deadends. Therefore, we replicate each deadend, that's to say, all of them will be visited twice and the exception will be discussed in below.

Then corresponding singluar nodes can be decided without difficulty. It's worthwhile to mention that we only save theirs' position in *self.SingularPositions* and save deadend nodes in *self.DeadNodes* as a list of *class Node*.

### Compute Minimun Distance
In this part, we compute the minimal path between given starting positions and ending positions which is implemented in *self.ComputeMinDistance*.

Since we have already known the graph of the map, we can apply BFS (Dijsktra algorithm in graph theory) to compute. Normally we have to compute the minimal path between each node but for velocity, we add a variable in our function $K$. The meaning of $K$ is that we only compute first $K$ paths from one starting position. As a result, every singluar node is connected with no more than $K$ nodes. For example, node A is in connection with node B while for node B, node A is so far away that it's not in the  K-nearest neighbor.
### Choose Minimal Cost Connection
In this part, the aim is choosing the best connecting pairs. *self.ComputeBestConnect* is the crucial function in which we input a set of nodes' positions with the cost between them and it will return the best connecting method.

In general, we regard this problem as a Constraint Satisfication Problem and our code is adjusted from Homework2. The variables are all of these positions whose domians are K-nearest neighbor positions. Thus the only constraint is that if position A is assigned to position B, position B must be assigned to position A. Besides, we employ *weight* in original code as the presentation of path cost.

As we all know, *Backtracking* will return a result only if there's one solution found or no solution. However, we just want to find all optimal solutions so most valid solutions are meaningless. Two methods are applyed to avoid useless computation.

The first method is pass the minimal cost of previous solution to the *Backtracking* function so that if the weight of a possible solution has already exceeded the optimal cost, the solver will return immediately.

The other one is that when we define our csp problem,  compute the minimal path cost of each node and minus it to abtain a new path cost. As a result, the method above can be applied more efficiently and speed up our computation.

Nevertheless, there's still a distance of finding out the minimal cost connection because the starting node can be an odd vertex after connection. Hence totally there're three cases shown below assuming that the starting node is an even vertex originally and there're $2n$ odd vertexes in sum.

  1. Link $n$ pairs of singular nodes, i.e., end up in the starting point.
  2. Link the starting node with one singular node and $n-1$ pairs of singular nodes, i.e., end up in a singular point.
  3. Link the starting node with one singular node, delete the replication of one deadend path and link the rest of singular nodes, i.e., end up in a deadend.

The whole idea is implemented in *self.ComputeBestEuler* returning an adjusted Eulerian graph.
### Eulerian Path
In this part, we compute our final path given the adjusted Eulerian graph and make some amendments to the final path in *class EulerianCircuit*.

We apply Hierholzer's algorithm to derive the path given the starting position:

  1. Follow a trail of edges from the starting point until returning to it or the ending point. It is not possible to get stuck at any vertex other than them, because the even degree of all vertices ensures that, when the trail enters another vertex $w$ there must be an unused edge leaving $w$. The tour formed in this way is a closed tour, but may not cover all the vertices and edges of the initial graph.
  2. As long as there exists a vertex $u$ that belongs to the current tour but that has adjacent edges not part of the tour, start another trail from $u$, following unused edges until returning to $u$, and join the tour formed in this way to the previous tour.

The step 2 can be performed in constant time each, so the overall algorithm takes linear time, $O(E)$.

Sometimes, the Pac-man needn't head to the node position because it's already been eaten. To be specific, if the Pac-man heads to one node then returns immediately, and moreover, the degree of this node is at least 3, e.g., it can be visited again, the Pac-man doesn't have to reach the position.
