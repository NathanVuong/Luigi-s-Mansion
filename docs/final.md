---
layout: default
title: Final
---

## **Video**
<iframe width="560" height="315" src="https://www.youtube.com/embed/EzysVRQjWPM" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

## **Project Summary**
Using OpenAI’s Gym environment, we aimed to train a reinforcement learning agent to complete the first level of the original Super Mario Bros. To accomplish this feat, we tried multiple algorithms such as Rainbow DQN and PPO in order to compare different methods of learning. We conducted extensive testing, allowing Mario to experiment for millions of lives with different inputs and rewards, and collected the results of his performance through different phases of our project. Through these results and recordings, we observed Mario’s behavior with recurring obstacles and looked for ways to improve his movement and bring him closer to the final flag.

Super Mario Bros is a complex game that allows for many different outcomes and game states which we believe makes it a perfect problem for an RL agent to solve. There are enemies, tall obstacles, and traps for the RL agent to learn all within the first level. Additionally, there are various things which we could choose to shape the reward system for the agent. These attributes of Super Mario Bros make it a good example of a small-scale obstacle course. A similar task on a larger scale could be navigating terrain in robotics, or self driving vehicles. Obstacle detection, in games and real life, are optimal when controlled by human intelligence due to their ability to adapt quickly to dynamic enviornments. However, human intelligence is fickle and dependent on wellbeing and many other factors. Optimizing computer controlled movement according to scenario rules can be a breakthrough for automating tasks for consistency and safety. Learning to first optimize computer controlled movement for a simpler environment, such as Super Mario Bros, will build up knowledge of how to guide computers to complete larger tasks.

## **Approaches**

### <ins>Setup and Preprocessing</ins>
Each episode represents a single life in which the agent is rewarded for progressing (winning a level) and penalized for dying. Training on one life incentivizes the agent to be more cautious, leading to more deliberate learning. In addition to the baseline reward, we reward Mario for completing the level or penalize him for dying.
To optimize the environment setup for training, we limited Mario’s action space to a predefined set of simplified movement actions. The default action space in Super Mario Bros Gym is extensive, including all possible button combinations. By using SIMPLE_MOVEMENT, we reduce the number of actions to essential ones, such as moving left, right, and jumping. 

We applied the GrayScaleObservation wrapper to convert the environment’s observation from color to grayscale. This reduces the dimensionality of the input data (from three color channels to one), making training more efficient. Furthermore, we decided that color is not crucial for gameplay since edges and object shapes are more important for decision-making. We also used the ResizeObservation wrapper to optimize efficiency and scale the environment to 84x84 pixels. This resolution is sufficient for learning while significantly reducing computational load. Lastly, we implemented frame skipping, allowing the agent to select an action only once every four frames instead of every frame. This approach reduces computation by avoiding unnecessary updates since the environment does not change significantly within a few frames.

### <ins>Improvements and Development Timeline</ins>

### <ins>Naive</ins>
One of our baseline approaches was to evaluate Mario’s performance when selecting movements randomly. In this approach, Mario makes decisions without any awareness of the environment, and each movement is chosen at random with equal probability. This method is a simple reference point to measure how well other approaches perform in comparison. 

Some advantages of this naive approach are minimal computation cost, no training required, and suitability for establishing a lower bound. Some disadvantages are uninformed/inefficient movement, lack of learning or adaptation, and high failure rate.

### <ins>DQN</ins>

### <ins>Rainbow DQN</ins>

### <ins>PPO</ins>

## **Evaluation**
We ran three different models, DQN, Rainbow DQN, and PPO. We evaluated the success of our agent through tracking reward, x position, and time spent. 

## **References**
Some of the resources we used include official documentation from OpenAI on their Gym environment as well as various random StackOverflow pages from other users who have run into similar issues as us.
* https://pypi.org/project/gym-super-mario-bros/
* https://www.gymlibrary.dev/index.html
* https://stackoverflow.com/search?q=gym+Super+Mario+bros&s=40640cc8-8588-47e6-8b29-9e3c8d397d5b
* https://www.romhacking.net/utilities/178/ 

Luu, Q. T. (2023, April 10). Q-Learning vs. Deep Q-Learning vs. Deep Q-Network \| Baeldung on Computer Science. www.baeldung.com. https://www.baeldung.com/cs/q-learning-vs-deep-q-learning-vs-deep-q-network

## **AI Tool Usage**

