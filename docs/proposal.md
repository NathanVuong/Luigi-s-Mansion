---
layout: default
title: Proposal
---

## **Summary**
We will be training an RL agent to complete Super Mario Bro's first level. We will judge success by how far right Mario ('s position) can make it into the level. The agent will take information about the setup of the level and what it has "learned" from past attempts to make it as far into the level as possible. Our stretch goal would be to apply the same agent to complete other levels or make different agents for different goal optimziations (Fastest Completion, Best Score).

Some applications for this project could be movement of human agents in a obstacle course (3D ommitted)

First meeting with the professor is scheduled for week 5, 2/5.  

## **Algorithms**

We anticipate using Q-learning (off policy) to reward the RL agent for different actions and building a neural network for the RL agent to make decisions with.

**More Notes**:
Since progression in a stage is independent besides the score(no carry over buffs), there's no need to account for progression factors.    
We can potentially use a on policy algorithm to account for enemies in stretch goals.

Basic Heuristic Guidelines:
- \+ Moving forward
- \+ Avoiding enemies
- \- Time Penalties
- \- Moving backwards
- \- Dying

## **Evaluation**

**Quantitative Evaluation**:
The primary metric for evaluating the agent will be distance traveled, measured as Mario's horizontal position at the end of the stage or when he dies. A secondary metric could include completion time and score. For a baseline, the goal is to have the agent complete the first stage in the generated world. Overall, the agent will initially be evaluated in Mario's first stage, to refine the trained model to potentially complete later levels.


**Qualitative Analysis**:
To qualitatively verify if our project works, we could record gameplays to demonstrate how the agent improves over time. For example, we can show side-by-side gameplay comparisons of early training attempts versus later attempts, highlighting the agent's progression in strategy. Our sanity checks will focus on Mario's isolated challenges, such as jumping over a single gap or avoiding a single enemy. Our moonshot case would be successfully transferring the trained agent from the first level to other levels, showcasing the agent's ability to generalize its learned strategies. It would be impressive if the agent could consistently complete more than two stages.


## **Tools**
We will be using the gym-super-mario-bros library from Open AI Gymnasium.   
This library will allow us to set up an environment to run Super Mario Bros while allowing us to focus on the actual reinforcement learning. 
