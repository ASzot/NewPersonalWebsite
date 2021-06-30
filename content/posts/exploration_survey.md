---
title: Curiosity Driven AI
date: 2019-04-01
---

<h3>Abstract</h3>

  What has driven human accomplishments in sciences? Why do humans want to
  understand the world around them? The answer could possibly have to do with
  curiosity, the desire to learn or know something. Artificial intelligence
  (AI) seeks to create machines that can learn and think for themselves, just
  as humans do. Intuitively, curiosity is a crucial component of our learning.
  How can we give machines a sense of curiosity and desire to learn about the
  world around them? This post will examine how AI researchers have succeeded
  in giving machines a sense of curiosity to better learn on their own. The
  focus is on the intuitions behind concepts and will not be
  technical. This post is meant for a wide audience and no experience in
  reinforcement learning is assumed.

## Introduction

Modern AI is not as smart as one would think. AI is able to solve tasks in
which there is a clear correct answer and developers provide many examples
from which the AI algorithm can learn. This has enabled AI to do incredible
things understand what objects are in a photo or recognize speech. However,
AI fails to solve some remarkably easy tasks. AI struggles in areas where the
correct answer is unclear and it must learn the correct answer on its own,
without examples being provided. For instance, how can we teach an AI
algorithm how to play the game of Mario? It's difficult to show the AI how to
solve a level and then expect it to copy us because we want the AI to figure
out the knowledge it needs on its own. In Mario, whatever sequence of actions
result in us reaching the end of the level is the correct behavior. There is
no one right answer. 

For these types of environments, like Mario, where there is no correct answer
but we have some goal we want to achieve (say completing a level in a video
game or picking up an object with a robotic hand) we use a type of AI
algorithm called reinforcement learning (RL for short). The fundamental
theory of RL is clear, however, there exist substantial engineering problems
to apply these fundamental theories to real applications. One of the major
challenges has been how AI can understand the environment around it.
To address this, scientists have found inspiration in how human curiosity
drives us to understand our world. 
These scientists found that they could engineer curiosity into AI, making the
AI algorithm want to discover and understand the environment around it. This
methodology has culminated in break through results in RL. 

## Video Game Playing AI
  
Let’s say we are trying to teach an RL agent to play the game of Mario. Agent
just means the “brain” of the AI algorithm that is controlling the character
on the screen. In RL our AI algorithm learns by doing. We place our agent in
the virtual world of Mario just like a human who is playing the game. The
agent is able to move our Mario character left, right and jump just like a
human holding the physical controller would. Our RL agent also gets to see
the screen like our human eyes would. However, instead of the light from the
screen entering eyes, the numerical RGB values are fed to the RL agent as
input. We reward our RL agent for doing good things, like completing levels
in Mario and punish it for doing bad things, like falling in a pit in Mario.
Over thousands of trial and error plays of the game, eventually our RL
algorithm will learn how to control the Mario character to beat levels.

The concept of learning from trial and error is simple but of course it is
not as easy for a computer as it is for a human. For instance, the RL agent
in our Mario world has to learn to interpret the RGB numerical values that
are being fed into it. When humans play this game they are playing with the
prior information about the world around them. In one glance, a human is able
to understand that they are controlling the little Italian man named Mario at
the center of the screen. We are able to understand a platform can be jumped
on and to watch out for menacing looking characters without thinking about
it. The RL algorithm needs to learn all of this from scratch. Figure 1
demonstrates how a game might look to an RL algorithm that has no prior
knowledge about the world such as visual and object consistencies [[1]](#sources).

{{< image src="/img/curiosity/game_prior.gif" class="center-image" width="400px" caption="Figure 1: Top, playing a game with human priors. Bottom, playing the same game without priors such as objects or visual consistencies.  This is the input an RL policies must learn from. See [[1]](#caption1) for more information and where these GIFs are from." >}}
{{< image src="/img/curiosity/game_no_prior.gif" class="center-image" width="400px" >}}


Starting from nothing means it is difficult for RL algorithms to understand the world around
them. Having to learn entirely from scratch means they need millions of plays
to learn. Another issue is that this RL agent has no idea of what it should
do in this environment. The only way to measure how well it is doing is the
reward or score it receives from the environment. However, this is
problematic because this reward is just a number and our RL agent does not
understand what this number really means. 

How does our agent know this reward number it has been given is a good number? What
if we could achieve more reward? Consider the following story. You are at a
casino playing on the slot machines and you have found a machine which you
seem to consistently be making money on. Would you keep playing this same
machine? But because this machine seems to be in your favor maybe you could
win even more on another slot machine. Would your answer change if this was
the first machine you had played at? What if you had already tried almost all
of them? This story highlights the exploration versus exploitation problem,
which at its core is how to trade off using a best solution versus looking
for something better in an unknown environment.
Our agent controlling the virtual Mario character faces the same problem with
not knowing if it's behavior should be exploited to get the same reward or if

This issue is made even worse when the rewards we give the agent for doing a
good job are infrequent, we call this a "sparse reward setting". Sparse
rewards are common for humans. You're only told you did a good job by the
video game when you actually beat the game of Mario. If you only make it 50%
of the way through you'll get the same game over screen as if you died only
making it through 2% of the level. Sparse rewards are difficult for RL to
handle as it can be very rare that the agent gets told it's doing a good job
meaning the agent is lost, having little idea on how to improve. The issue
comes down to a problem of exploration. If we are able to efficiently explore
the environment maybe we can get lucky and stumble onto some sparse rewards.

## Introducing Curiosity

Curiosity can be responsible for wanting to explore the world. From this AI
researchers have tried to make RL algorithms explicitly curious about their
environment and want to explore their environment. Curiosity and exploration
are tightly linked as curiosity is just a method to explore the environment. However, the full
definition of curiosity could include more than just a desire to see new
things but also to understand why they work. We will start with curiosity
about seeing states of the environment and build up to curiosity about deeper
mechanisms of the environment.

## Epsilon Greedy

The epsilon greedy exploration algorithm is the simplest and most commonly
used exploration algorithms. The idea is that every time the agent decides
what to do, with some small probability named epsilon, it chooses an action
at random instead of what it thinks is best [[2]](#sources). Epsilon is set to be a small
value, meaning the agent only explores a random action's outcome very
infrequently and often follows what it thinks is best. Overall, this type of
curiosity is too random and does not explicitly seek new understanding of the
environment. 

## Count Based Exploration

If we are trying to explore everything in our environment then why not reward
the agent for seeing new things? Count based exploration gives the agent a
good job reward every time it sees a new state. We add this reward reward
onto the normal reward from the environment. This added reward represents how
curious our agent is about the state. In count based exploration the added
reward is the inverse of the number of times the agent has seen the state
[[3]](#sources). States with many visits will garner little curiosity while those with
few will be rewarded.

Humans are curious to see new things but they aren't curious to see
everything. While interested, we don't stare endlessly at every new snowflake
that falls from the sky. We are typically most curious to see states that are
semantically different. One pixel being a slightly different color shouldn't
matter to us. Thankfully, the RL algorithm #Explore, released in 2016 by
researchers at AI research group DeepMind,
addresses this issue. In #Explore, the RL agent learns to distinguish what
major things it sees are important and appear to new to it [[4]](#sources). Now
our Mario agent will only be curious when it sees an entirely new scene, such
as a tunnel section or a enemy it hasn't seen.

## Being Curious About How Things Work

However, simply trying to see new states is not all there is to curiosity.
For humans our curiosity motivates us to not just see new things but to also
understand the world around us. For instance, what will happen when Mario
jumps on the Goomba (Pictured in Figure 2)? A player who has never played
Mario will be surprised to find when Mario flattens the Goomba. This player
has done a good thing, finding some surprising new behavior that turns out to
be crucial for passing levels where they need to flatten enemies to get
around them.

{{< image src="/img/curiosity/jump.jpg" class="center-image" width="400px" caption="Figure 2: Mario jumping on a Goomba enemy" >}}

We can provide an agent incentive for finding this new type of behavior by
first learning a world model. A world model is a predictive model that
tries to predict what will happen. In this the RL agent tries to predict
what will happen next given what it currently sees. So as Mario is falling
on the Goomba, the RL agent will try to predict what will happen. If the
agent has not seen Mario squash a Goomba already the AI will receive a
curiosity reward when it is unable to predict this happening. This will
drive the AI to find results which it cannot explain and then seek to model
and understand these behaviors as well.

The issue with this world modelling approach is known as the Noisy TV
Problem. If the goal of the AI agent is to model everything in the
environment, how will it deal with modelling randomness that the agent should
not try to model? Consider an RL agent trying to solve a maze, all we need to
do is put a TV in the maze and suddenly the AI agent is mesmerized. 
You can see this in action in Figure 3. 

{{< image src="/img/curiosity/noisy_tv_problem.gif" class="center-image" width="400px" caption="Figure 3: maze navigation agent getting stuck by a tv that has nothing to do with reaching the end of the maze. See [[5]](#sources) for source of this GIF." >}}

The reason why the TV captures all curiosity, is the agent is trying to model
things that are outside of its control. The TV can display random static and
the agent will try to understand it. It's impossible to understand the static
as it is by definition random so the agent will be transfixed by it forever. 
The solution of this is to only be
curious about things that the agent can actually affect. This is the idea of
the paper "Curiosity Driven Exploration by Self-supervised Prediction"
produced by students at the University of California, Berkeley. In
this paper they found that their proposed method is able to solve many levels of the Mario
environment without giving the agent any sort of reward for solving the
levels at all and relied purely on the curiosity reward [[6]](#sources). Instead the
Mario RL agent is curious to discover new things and in the process
progresses through many of the levels without explicit understanding that it
is beating the game.

## Random Curiosity

Random Network Distillation, another recent method for augmenting standard RL
with curiosity, took the reinforcement learning research community by
surprise with how deceivingly simple it is. Before discussing the engineering
behind it let's first look at a long standing game that has gone unsolved by
RL, Montezuma's Revenge.

Montezuma's Revenge is a classic Atari game where you control an explorer and
must gather keys to open doors and avoid obstacles along the way. When RL
agents first started to beat Atari games Montezuma's Revenge was the one game
where not a single algorithm could complete even one level. The game is so
difficult because of how difficult exploration is in it. The agent must learn
that a key opens up a door before getting any sort of reward for progressing
to the next level, all the while doing complex obstacle avoidance which would
require a slight of hand for most humans with a physical controller. An image
from the game is shown in figure 4.

{{< image src="/img/curiosity/mr.png" class="center-image" width="400px" caption="Figure 4: A picture taken from the game Montezuma's Revenge. " >}}

Random Network Distillation, released out of the research company OpenAI was
the first to completely solve Montezuma's Revenge. The key insight of the
OpenAI researchers was the RL agent could learn to predict the output of a
fixed random transformation on the visual input. This just means that this
random transformation computes some sort of "secret" in every image input
[[7]](#sources). This secret could be what shape is in the top left corner or what colors
are in the middle. The agent then has to predict the correct output of this
transformation as before with predicting the world model. When the agent is
surprised at the output of the random transformation that means it has
stumbled across something new and is given a reward for being curious. It
then stops being curious when it understands what the "secret" to this scene
is.

The important aspect of this approach is the transformation agent uses to
understand the environment is random. This has two advantages. The first is
that the random transformation does not need to model anything. Before when
predicting the world model the world model had to correspond to actual
reality. This world model could be difficult to represent as modelling
reality is a hard task. However, the random network does not need to
learn to represent anything as these "secrets" they extract are completely
random. Furthermore, it turns out through experimentation done at the OpenAI
labs, extracting these random features keeps the RL agent curious about the
important aspects of the scene and ignores trivial details. In
experimentation this was shown to keep the agent focused on discovering new
rooms in Montezuma's revenge and was able to beat human performance on the
game [[7]](#sources).

## Go Explore

Recently, a new algorithm from Uber AI labs named GoExplore crushed Random
Network Distillation's high score of 17,000 with a super
human performance of 400,000 [[8]](#sources).

All of the curiosity methods discussed so far face a problem called curiosity
detachment. An agent might stumble into new areas and get curious about those
areas. However, they might then randomly stumble into another area that once
again sparks their curiosity and then forget about the previous area they
were just curious about. Essentially the agent forgets about promising areas
it had already seen that could have led to the solution. The idea of the
GoExplore algorithm is to keep track of all the promising areas of the
environment where the curiosity of the agent was stimulated. The agent
remembers the sequence of actions that got it to that state and randomly
decides to go back to some of the states it was curious about and explore
them more in depth. This results in GoExplore constantly focusing on
expanding the limits of its knowledge and being curious about all aspects of
the game it is playing, not just what it is looking at right now. Overall,
GoExplore is a different method than what was previously seen and remains the
best exploration algorithm since the time of this writing.

{{< image src="/img/curiosity/detach.png" class="center-image" width="400px" caption="Figure 5: Image taken from Uber's blog post on GoExplore located at [[8]](#sources). Note that intrinsic reward is just the same as curiosity." >}}

## Conclusion

Overall, in the last year there have been numerous methods that have
revolutionized the field of curiosity driven exploration in AI. These methods
have pushed the best AI agents in a variety of video games which were
previously unsolvable.

However, the fight for better exploration continues. Solving sparse reward
tasks with difficult control is a challenging task with RL. In all the video
games we discussed, the actions are simple; move your character any direction
or jump. The screen was also 2D and only had simple shapes. The story is
different if we are trying to control a robotic arm that has 7 different
joints on it. Things become even harder when we need to deal with the real
world complexity of 3D images. Despite this, the main motivation of curiosity
will likely still apply. 

Curiosity has been a key motivating factor for the solution to the
exploration problem for decades and it will likely remain so for decades to
come. RL has a long way to go from being able to solve real world tasks on
its own. However, modelling curiosity is a step in the right direction.
Agents that are curious about the world will be able to derive knowledge
without humans explicitly telling them to do so. It's hard to give a real
world agent a reward for doing something. After all humans are able to do
things without rewards, or at least work for very sparse rewards. Curiosity
is making the same possible with AI.

## Sources
1.  Dubey, Rachit, et al. "Investigating human priors for playing video games."
    arXiv preprint arXiv:1802.10217 (2018).
2.  Salakhutdinov, Russ. “Deep Reinforcement Learning and Control.”
    www.cs.cmu.edu/~rsalakhu/10703/Lecture_Exploration.pdf.
3.  Martin, Jarryd, et al. "Count-based exploration in feature space for
    reinforcement learning." arXiv preprint arXiv:1706.08090 (2017).
4. Tang, Haoran, et al. "# Exploration: A study of count-based exploration for
    deep reinforcement learning." Advances in neural information processing
    systems. 2017.
5. Vincent, James. “How Teaching AI to Be Curious Helps Machines Learn for
    Themselves.” The Verge, The Verge, 1 Nov. 2018,
    www.theverge.com/2018/11/1/18051196/ai-artificial-intelligence-curiosity-openai-montezumas-revenge-noisy-tv-problem.
6. Pathak, Deepak, et al. "Curiosity-driven exploration by self-supervised
    prediction." Proceedings of the IEEE Conference on Computer Vision and
    Pattern Recognition Workshops. 2017.
7. Burda, Yuri, et al. "Exploration by random network distillation." arXiv
    preprint arXiv:1810.12894 (2018).
8. Ecoffet, Adrien, et al. "Montezuma’s revenge solved by go-explore, a new
    algorithm for hard-exploration problems (sets records on pitfall, too)."
    Uber Engineering Blog, Nov (2018).
