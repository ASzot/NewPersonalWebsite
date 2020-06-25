---
title: Setting up OpenAI Gym with MuJoCo
date: 2018-04-05
---
## Introduction
MuJoCo is a powerful physics simulator that you can run experiments in. 
OpenAI Gym makes it a useful environment to train reinforcement learning agents
in.

Before doing this, I didn't have a lot of experience with RL, MuJoCo, or OpenAI gym. I
wanted to get more involved in RL and wanted to solve a custom physics problem
I had in mind using RL. If you're in the same boat of wanting to solve an RL problem
that uses physics simulation then this post is for you. 

## Installing MuJoCo with Gym

Installing MuJoCo and Gym was a total pain. Getting MuJoCo to run alone was
difficult enough but often the version did not match up with the version of
Gym. (If you don't have a .edu email or MuJoCo will cost you). 

It turns out MuJoCo 0.5.7 and OpenAI Gym 0.7.4 are versions that are compatible.
This is accurate as of March 6th, 2018. In the future other versions may be
compatible but these are guaranteed to work.

Now that you should have everything installed run the "Humanoid" environment to
test that everything works.

## Setting Up a MuJoCo Scene
Surprisingly, I could not find a lot of tutorials on how to write custom MuJoCo
Gym environments online. The source code for the existing MuJoCo environments at 
[https://github.com/openai/gym/blob/master/gym/envs/mujoco/](https://github.com/openai/gym/blob/master/gym/envs/mujoco/).
are really the only help. 

The first step is to define the MuJoCo scene itself. You do this through an XML
file detailing how the scene is laid out. The best way for getting this
done is to read the existing Gym MuJoCo scenes at 
https://github.com/openai/gym/tree/master/gym/envs/mujoco/assets
and the MuJoCo XML documentation 
http://mujoco.org/book/modeling.html.
Start with an existing MuJoCo scene in the Gym repository and go from there.

For instance placing a box would look like 

```python
&lt body name="goal" pos="0 6 -0.24" &gt
  &lt geom type="box" rgba="1 0 0 1" size="0.25 0.25 0.25" / &gt
&lt /body &gt
```

Now there's a couple things I had trouble figuring out.
First of all, objects will <b>not</b> move if they don't have a joint on
them. 

The different types of joints affect which way objects can be moved. For
instance, if you are trying to have an object to move normally (be affected by
forces in every direction) then use the `type="free"` joint.
I wanted a ball to fall down and roll normally so I put the
`"free"` joint on it.
See the full documentation on joints go <a href="http://mujoco.org/book/modeling.html#joint">here</a>.

Watch out for the options in the `<option>` tag. Many of the
custom Gym MuJoCo environments will specify the `timestep` attribute
in the option tag. A higher timestep value will mean faster simulations but
less accuracy. Many of the Gym MuJoCo environments sacrifice accuracy for
speed. When starting out I would recommend making sure this value is the
default of `0.002`.

## Setting Up a Custom Gym MuJoCo Environment

As mentioned before, looking at the existing definitions for the default Gym MuJoCo
environments is very helpful. Here is the general format

```python
class MyEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        utils.EzPickle.__init__(self)
        FILE_PATH = '' # Absolute path to your .xml MuJoCo scene file.
        # For instance if I had the file in a subfolder of the project where I
        # defined this custom environment I could say 
        # FILE_PATH = os.getcwd() + '/custom_envs/assets/simple_env.xml'
        mujoco_env.MujocoEnv.__init__(self, FILE_PATH, 5)

    def step(self, a):
        # Carry out one step 
        # Don't forget to do self.do_simulation(a, self.frame_skip)

    def viewer_setup(self):
        # Position the camera

    def reset_model(self):
        # Reset model to original state. 
        # This is called in the overall env.reset method
        # do not call this method directly. 

    def _get_obs(self):
      # Observation of environment feed to agent. This should never be called
      # directly but should be returned through reset_model and step
```

Now you have to register your environment. In the package that you defined your
custom environment class in, put the following in `__init__.py`.

```python
from gym.envs.registration import register

register(
  id='MyEnv-v0',
  entry_point='custom_envs.my_env:MyEnv',
)
```

Where `custom_envs` is the name of the package and
`my_env` is the file where the custom environment `MyEnv`
is defined.

Now when actually programming your environment there are a number of methods
that you can use to figure out what's going on in the physics simulation. To
get the position of a body in the scene simply refer to its name attribute.

```python
x, y, z = self.get_body_com("name of body here")
```

Likewise you can get the velocity of an object through
`get_body_comvel`.

<h3>Setting the Position of an Object in a Scene</h3>
For the purpose of my experiment I wanted the agent to determine the position
of an object in the scene when the scene was reset. It turns out that a
position can only be set if the body has a joint on it. Otherwise the
position information is ready only. The steps to setting the position of an
object are somewhat convoluted through what I found. First get all the
positions and velocities of the objects.

```python
qvel = np.array(self.model.data.qvel).flatten()
qpos = np.array(self.model.data.qpos).flatten()
```

Next you have to manually change the positions of bodies through editing this
array. For each body there are 7 attributes. The first 3 are the position and
the next 4 are the rotation quaternion. The array only stores this information
for bodies that have joints. So if you had 3 bodies with joints there would be
3 * 7 = 21 elements in this array. So simply index by the index of the body
with
the joint offset by 0 or 3 depending on position or rotation. Set
the elements that you desire then set the state of the bodies. 

```python
self.set_state(qpos, qvel)
```

## Miscellaneous Tasks
There were a number of things I wanted to do in MuJoCo but didn't know how
to. First of all, I wanted to be able to set the position of an object but not
have it be affected by any forces once set. To set the position of the object arbitrarily
I had to put a free joint on the body. However, this meant that gravity was
acting on it. The solution to this was to manually set the position of the body I
wanted to fix every single step.

I wanted to make a bouncy ball that would bounce off of a surface. Set the
solref parameter to achieve this on the `geom` tag of the body.
Lower values for the second value of solref correspond to bouncier surfaces.

Here's what the combination of the bouncy ball and fixed surface looked like!
(The agent decides how to orient the bounce surface).

{{< image src="/img/mujoco/show3.gif" class="center-image" width="400px" >}}
{{< image src="/img/mujoco/show.gif" class="center-image" width="400px" >}}

Also if you see in the second gif that it looks like the platform is slowly
moving downwards that's because I wasn't setting the velocity to 0 on every time
step when I was fixing the object.


## Conclusion
These are the primary things I have ran into while using MuJoCo so far. I will
update this post if I run into anything more. If you
have any questions or if anything is incorrect be sure send me an email "m e [at] andrewszot [dot] com"
(obviously no spaces).


