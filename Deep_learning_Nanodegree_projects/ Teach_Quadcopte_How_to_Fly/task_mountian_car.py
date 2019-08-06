#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 14:08:49 2019

@author: khaled
"""

import numpy as np

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self,env):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.env=env
        self.action_low = self.env.action_space.low
        self.action_high = self.env.action_space.high
        self.action_size = 1
        self.state_size=2
        self.target=0.5
        # Goal

    

    def step(self, action):
        """Uses action to obtain next state, reward, done."""
        next_state, reward, done, _ = self.env.step(action)
        #self.env.render()
        #reward=reward-abs(reward)*abs(next_state[0]-self.target)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        return  self.env.reset()

        #return state