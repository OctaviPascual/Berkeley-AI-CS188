# Artificial Intelligence - Berkeley - CS188 (Summer 2016)

These are my solutions to [edX Edge Artificial Intelligence - Berkeley CS188-SU16 (Summer 2016) course](https://edge.edx.org/courses/course-v1:BerkeleyX+CS188-SU16+SU16/about) instructed by Davis Foote and Jacob Andreas. I just want to thank them for this amazing course and for those challenging projects :bowtie:.

* **[Project 0: Tutorial][project0]**
* **[Project 1: Search][project1]**
* **[Project 2: Multiagent][project2]**
* **[Project 3: Reinforcement Learning][project3]**
* **Project 4: Bayes Nets (Not available on CS188 website)**
* **[Project 5: Tracking][project5]**
* **[Project 6: Classification][project6]**

[project0]: http://ai.berkeley.edu/tutorial.html
[project1]: http://ai.berkeley.edu/search.html
[project2]: http://ai.berkeley.edu/multiagent.html
[project3]: http://ai.berkeley.edu/reinforcement.html
[project5]: http://ai.berkeley.edu/tracking.html
[project6]: http://ai.berkeley.edu/classification.html

## Usage :nut_and_bolt:

For projects 0 to 5, it is straightforward to run them locally since they only use [Python Standard Library](https://docs.python.org/2.7/library/index.html). Just make sure you are using Python 2. However, for the last project, tensorflow library is needed and it might be tricky to install it. For that reason, a [Docker](https://www.docker.com/get-docker) image which has all the required dependencies already installed is provided. Check how to use it in [Docker](#docker) section.

If you want to run a single question from a project, use the following commands. Note that `QUESTION` is q1, q2, up to the number of questions of the project.

```bash
git clone https://github.com/EikaNN/Berkeley-AI-CS188.git
cd Berkeley-AI-CS188
cd project1-search
python autograder.py [-q QUESTION]
```

If you want to run multiple projects, or all the questions from one project, you can use the [main.py](main.py) script that I have implemented. Note that `PROJECT` is either 0, 1, 2, 3, 4 or 5.

```bash
git clone https://github.com/EikaNN/Berkeley-AI-CS188.git
cd Berkeley-AI-CS188
python main.py [-p PROJECT]
```

## Docker :whale:

To run the projects with Docker (again, `PROJECT` is either 0, 1, 2, 3, 4 or 5):

```bash
git clone https://github.com/EikaNN/Berkeley-AI-CS188.git
cd Berkeley-AI-CS188
./docker.sh [-p PROJECT]
```

## Contents :books:

In this section the algorithms that were implemented in each project are listed. For each project, there is a commit with the blank project followed by a commit with all the changes that were made. This way, it is easy to identify what has been implemented. All projects are completed with the maximum score.

### [Project 0: Tutorial](https://github.com/EikaNN/Berkeley-AI-CS188/commit/0e40e4f57c558f9160aa487312c554d535331344)

Introduction to the grading environment.

```
Provisional grades
==================
Question q1: 1/1
Question q2: 1/1
Question q3: 1/1
------------------
Total: 3/3
```

### [Project 1: Search](https://github.com/EikaNN/Berkeley-AI-CS188/commit/1e101226201fdfcd6aef6c30c7653181521723a9)

Implementation of DFS (Depth First Search), BFS (Breadth First Search), UCS (Uniform Cost Search) and A* search with heuristics.

```
Provisional grades
==================
Question q1: 3/3
Question q2: 3/3
Question q3: 3/3
Question q4: 3/3
Question q5: 3/3
Question q6: 3/3
Question q7: 5/4
Question q8: 3/3
------------------
Total: 26/25
```

### [Project 2: Multiagent](https://github.com/EikaNN/Berkeley-AI-CS188/commit/d2d4274a02771bc4a87c2b824ac9d41a917f4d07)

Implementation of Minimax, Alpha-Beta Pruning and Expectimax.

```
Provisional grades
==================
Question q1: 4/4
Question q2: 5/5
Question q3: 5/5
Question q4: 5/5
Question q5: 6/6
------------------
Total: 25/25
```

### [Project 3: Reinforcement Learning](https://github.com/EikaNN/Berkeley-AI-CS188/commit/5df0940aca637be9f7ae271ecb81d171ddd7a916)

Implementation of Q-learning.

```
Provisional grades
==================
Question q1: 4/4
Question q2: 1/1
Question q3: 5/5
Question q4: 1/1
Question q5: 3/3
Question q6: 4/4
Question q7: 2/2
Question q8: 1/1
Question q9: 1/1
Question q10: 3/3
------------------
Total: 25/25
```

### [Project 4: Bayes Nets](https://github.com/EikaNN/Berkeley-AI-CS188/commit/6d7ad323c3251f3446641de19921ce14cd37acf2)

Implementation of variable elimination and value-of-perfect-information computations for Bayes Nets.

```
Provisional grades
==================
Question q1: 3/3
Question q2: 3/3
Question q3: 5/5
Question q4: 4/4
Question q5: 4/4
Question q6: 4/4
Question q7: 1/1
Question q8: 4/4
------------------
Total: 28/28
```

### [Project 5: Tracking](https://github.com/EikaNN/Berkeley-AI-CS188/commit/08950e9269ed5f0cf9735a6f2c4e4f6190179fff)

Implementation of exact and approximante inference using Bayes Nets.

```
Provisional grades
==================
Question q1: 2/2
Question q2: 3/3
Question q3: 3/3
Question q4: 2/2
Question q5: 2/2
Question q6: 3/3
Question q7: 3/3
Question q8: 1/1
Question q9: 3/3
Question q10: 3/3
------------------
Total: 25/25
```

### [Project 6: Classification](https://github.com/EikaNN/Berkeley-AI-CS188/commit/8eb4861cb0a9807c90f597dc7002584d17fb8f2b)

Implementation of the perceptron algorithm and gradient descent to train neural network classifiers.

```
Provisional grades
==================
Question q1: 4/4
Question q2: 1/1
Question q3: 1/1
Question q4: 2/2
Question q5: 1/1
Question q6a: 2/2
Question q6b: 1/1
Question q6c: 1/1
Question q7: 1/1
Question q8: 2/2
Question q9: 5/5
Question q10: 4/4
------------------
Total: 25/25
```
