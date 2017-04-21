
# Pacman Medium Search (Euler Path)

## Nice Experience
This is a project proposed by the class of Stanford Artifical Intelligence. Our team aims at sloving the problem of eating all dots in mediumSearch whose path is filled with dots.

The method is proposed by Jitong Qi, and thanks for his persistance, we finished this extra problem before deadline. In the process, Siyuan Wang came up with some interesing ideas and her implement is very nice!

This is an absolutely pleasure to cooperate with them!


## Code Running
You can input the following command in terminal to run our programme:

python pacman.py -l mediumSearch -p SearchAgent -a prob=MediumGraphicProblem

Runtime of the programme is about 7 seconds.

## Abstract Graph and Determine Odd Vertexes
In this part, the jobs are to abstract a graph for Step 3(*self.Graphic*), choose odd vertexes in Step 2 and calculate some basic values for Step 4 (*self.ReGraphic*).

It's not diffcult to extract a graph from the given map by data structure *class Node* defined by ourself. While if we determine odd vertexes without any disposition, a number of odd vertices are unnecessary given that there're lots of deadends. Therefore, we replicate each deadend, that's to say, all of them will be visited twice and the exception will be discussed in below.

Then corresponding singluar nodes can be decided without difficulty. It's worthwhile to mention that we only save theirs' position in *self.SingularPositions* and save deadend nodes in *self.DeadNodes* as a list of *class Node*.
