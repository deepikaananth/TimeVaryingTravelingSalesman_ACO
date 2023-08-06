#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""

  Ant Colony heuristic based Optimization (ACO) for the Dynamic Traveling Salesman Problem (DTSP)

"""
import numpy as np
import cvxpy as cp
import matplotlib
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
import scipy
from scipy.integrate import odeint
from tqdm import tqdm
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
from matplotlib.animation import PillowWriter
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from random import randint
from itertools import permutations


# import ellipse_dynamics_3d


# In[2]:


# Paramteric function of a 2D ellipse in 3D space

def ellipse_dynamics_3d(state, t, u, v):
    x, y, z = state
    
    # 2D Ellipse in 3D parametric equations
    # x = c0 + cos(t)u0 + sin(t)v0
    # y = c1 + cos(t)u1 + sin(t)v1
    # z = c2 + cos(t)u2 + sin(t)v2
    
    dxdt = -np.sin(t)*u[0] + np.cos(t)*v[0]
    dydt = -np.sin(t)*u[1] + np.cos(t)*v[1]
    dzdt = -np.sin(t)*u[2] + np.cos(t)*v[2]
    
    return [dxdt, dydt, dzdt]


def construct_ellipse(u, v, c):
    
    # no need to integrate
    T_lim = 15;    T_pts = 500
    t = np.linspace(0, T_lim, T_pts)
    
    x = -np.sin(t)*u[0] + np.cos(t)*v[0] + c[0]
    y = -np.sin(t)*u[1] + np.cos(t)*v[1] + c[1]
    z = -np.sin(t)*u[2] + np.cos(t)*v[2] + c[2]
    
    return np.vstack((x, y, z)).T

print('Done defining the method ellipse_dynamics_3d')


# In[3]:


# Creating an orbit using the ellipse_dynamics_3d method

# Creating a random Orbit (orbit1)
u1 = np.array([1, 2, 4])
v1 = np.array([5, 3, 0])
c1 = np.array([0, 0, 0])       # Center of the ellipse in 3d space
init_pos1 = [c1[0]+u1[0], c1[1]+u1[1], c1[2]+u1[2]]    # Substituting the values of the known constants along with t = 0 into the dynamics
T_lim = 15
T_pts = 500
t = np.linspace(0, T_lim, T_pts)


#### No need to intergrate, since we already have the parametric equations. Integration is only necessary if we had
#### ... to calculate the AREA of the ellipse
#### state1 = scipy.integrate.odeint(ellipse_dynamics_3d, init_pos1, t, args=(u1,v1))
####for i in range(state1.shape[0]):
####    # print('state[i,0] = ', state[i,0])
####    state1[i, 0] = state1[i, 0] + c1[0]
####    state1[i, 1] = state1[i, 1] + c1[1]
####    state1[i, 2] = state1[i, 2] + c1[2]


state1 = construct_ellipse(u1, v1, c1)


# Plotting
ax = plt.figure(figsize=(10,10)).add_subplot(projection='3d')
ax.plot(state1[:,0], state1[:,1], state1[:,2], linewidth=2.15, label='Orbit 1')
plt.title('Randomly created elliptical orbit centered at = ({},{},{})'.format(c1[0], c1[1], c1[2]))
ax.legend()
ax.scatter3D(c1[0], c1[1], c1[2], s=50, color=(1,0.2,0.2))
ax.text(c1[0], c1[1], c1[2], s="center")

# Plotting the u and v vectors:
ax.plot([0,u1[0]], [0,u1[1]], [0,u1[2]])
ax.plot([0,v1[0]], [0,v1[1]], [0,v1[2]])

# Plotting the shadows (projections)
# 1. Shadow below (on XY plane)
ax.plot(state1[:,0], state1[:,1], -6*np.ones_like(state1[:,0]), linewidth=1.5, color='grey')
ax.scatter3D(c1[0], c1[1], -6, s=20, color=(1,0.2,0.2))
ax.plot([c1[0], c1[0]], [c1[1],c1[1]], [c1[2], -6], linewidth=1, color='grey', linestyle='--')

# 2. Shadown on left (YZ plane)
ax.plot(-6*np.ones_like(state1[:,0]), state1[:,1], state1[:,2], linewidth=1.5, color='grey')
ax.scatter3D( -6, c1[1], c1[2], s=20, color=(1,0.2,0.2))
ax.plot([c1[0], -6], [c1[1], c1[1]], [c1[2], c1[2]], linewidth=1, color='grey', linestyle='--')

# 3. Shadown on right (XZ plane)
ax.plot(state1[:,0],  6*np.ones_like(state1[:,0]), state1[:,2], linewidth=1.5, color='grey')
ax.scatter3D(c1[0],    6, c1[2], s=20, color=(1,0.2,0.2))
ax.plot([c1[0], c1[0]], [c1[1], 6], [c1[2], c1[2]], linewidth=1, color='grey', linestyle='--')


ax.set_xlabel('x', fontweight='bold')
ax.set_ylabel('y', fontweight='bold')
ax.set_zlabel('z', fontweight='bold')
plt.show()    


# In[4]:


"""

################################################
################################################
                Class "orbit"
################################################
################################################

"""

class orbit:
    
    def __init__(self, center, T_lim, T_pts, u_vectors, v_vectors, num_orbits, init_pos, num_ants, iters, rho, tau0, alpha, beta, del_t_W, q_ph, ph_min, ph_max, q0):
        self.T_lim = T_lim      # max lim of the t_vector to be used intergrate.odeint()
        self.T_pts = T_pts      # representation of the orbital periods of each of the pieces of debris
        self.c = center         # center of the orbits (can be changed to focus/ii)
        self.u = u_vectors      # num_orbits x 3 array of the parametric vectors defining the orbits
        self.v = v_vectors      # num_orbits x 3 array of the parametric vectors defining the orbits orthogonal to the corresponding u's
        self.n = num_orbits     # Number of orbits
        self.init = init_pos    # initial positions of all pieces of debris in their orbits
        self.iters = iters      # number of iterations we want the algo to run for
        self.rho = rho          # evaporation rate for ACO
        self.tau0 = tau0        # pheromone trails initialization constant
        self.ants = num_ants    # omega parameter in paper
        self.alpha = alpha      # hyperparamter exponential for pheromone trail value
        self.beta = beta        # hyperparamter exponential for reciprocal of weights value 
        self.delta = del_t_W    # simulation time step at every update - the iterable
        self.delta_t = del_t_W  # simulation time step at every update - the time step size
        self.q_ph = q_ph        # prob. threshold for choice between ib and bs soln path for ph trails update
        self.ph_min = ph_min    # lower limit on phermone trails value
        self.ph_max = ph_max    # upper limit on phermone trails value
        self.q0 = q0            # prob hyperparameter to decide whether the most liekely or random node is picked as the next node to go for an ant
        self.w_sum_bs = 1000000 # best sum over weights at some time point
        self.time_bs = 0.0      # the time point at which best sum over weights was achieved
        
        # Weight/Edge matrix    # Initialized with initial position distances
        self.W = scipy.spatial.distance.cdist(self.init, self.init, metric='euclidean')      # Initialized with the 
        
        # Pheromone trail matrix
        self.ph = np.ones((self.n, self.n)) * self.tau0   # where tau0 is the initialization value
        
        # two paths: ib and bs : a sequence of nodes of the graph each representing
        self.path_ib = np.arange(0, self.n, 1)    # Iteration best ant's path randomly initialized to be in increasing order of the nodes/pieces of debris
        self.path_bs = np.arange(0, self.n, 1)    # Best so far ant's path randomly initialized to be in increasing order of the nodes/pieces of debris
        
        ## The Paths of each of the ants: initialized with each ant randomly assigned a node
        self.paths = np.zeros((self.ants, self.n))
        self.paths[:,0] = np.random.randint(0, self.n-1, (self.ants,))
        
        print('Initialized self.paths = ')
        print(self.paths)

        
       
    def ellipse_dynamics_3d(self, t, u, v):
        """
        ### To be used for scipy.integrate.odeint; defines the dynamics given by the 3d ellipse 
        ### parametric equations
        ### Inputs: state, time vector and arguments: u and v vector parameterizing the orbital trajectory 
                    being integrated over
        ### Output: stacked vector of the partials over time of the state (wrt the three coords)
        """
        
        # x, y, z = state
        
        # dxdt = -np.sin(t)*u[0,0] + np.cos(t)*v[0,0]
        # dydt = -np.sin(t)*u[0,1] + np.cos(t)*v[0,1]
        # dzdt = -np.sin(t)*u[0,2] + np.cos(t)*v[0,2]
        
        x = -np.cos(t)*u[0,0] + np.sin(t)*v[0,0] + self.c[0]
        y = -np.cos(t)*u[0,1] + np.sin(t)*v[0,1] + self.c[1]
        z = -np.cos(t)*u[0,2] + np.sin(t)*v[0,2] + self.c[2]
        
        # return [dxdt, dydt, dzdt]
        return x, y, z
    
    
    
    def ellipse_dynamics(self, t, u, v):   # t_v: time vector
        """
        ### Inputs: time vector and arguments: u and v vector parameterizing the orbital trajectory 
        ### Output: current orbital position at time t
        """

        dt = [self.c[0] + np.cos(t)*u[0] + np.sin(t)*v[0], self.c[1] + np.cos(t)*u[1] + np.sin(t)*v[1], self.c[2] + np.cos(t)*u[2] + np.sin(t)*v[2]]
        return dt
    
    
    """
    
    def ellipse_dynamics_non_orthogonal(self, t, u, v):
        
        # 2D Ellipse in 3D parametric equations for non orthogonal u and v
        # x = c0 + sin(t)u0 + sin(t+theta)v0
        # y = c1 + sin(t)u1 + sin(t+theta)v1
        # z = c2 + sin(t)u2 + sin(t+theta)v2
        print('u = ', u)
        print('v = ', v)
        print('t = ', t)
        u = np.reshape(u, (3,))
        v = np.reshape(v, (3,))
        theta = np.arccos(np.dot(u, v)/(np.linalg.norm(u)*np.linalg.norm(v)))
        if theta > np.pi:
            theta = theta % np.pi
        
        if np.linalg.norm(u) > np.linalg.norm(v):    # u is now the semi major axis vector
            dd = math.sqrt(((np.linalg.norm(u,2))**2 * np.sin(t)**2) + ((np.linalg.norm(v,2))**2 * np.sin(t)**2))
        else:                                        # v is the semi major axis vector in this case
            dd = math.sqrt(((np.linalg.norm(v,2))**2 * np.sin(t)**2) + ((np.linalg.norm(u,2))**2 * np.sin(t)**2))
        
        return dd
        
    """
        
    
    def ellipse_dynamics_non_orthogonal(self, state, t, u, v):
        
        d = state        # passed as argument but unused; its integral is returned by the function
        
        theta = np.arccos(np.dot(u, v)/(np.linalg.norm(u)*np.linalg.norm(v)))
        
        if theta > np.pi:
            theta = theta % np.pi
        
        if np.linalg.norm(u) > np.linalg.norm(v):    # u is now the semi major axis vector
            dd = math.sqrt(((np.linalg.norm(u,2))**2 * np.sin(t)**2) + ((np.linalg.norm(v,2))**2 * np.sin(t)**2))
        else:                                        # v is the semi major axis vector in this case
            dd = math.sqrt(((np.linalg.norm(v,2))**2 * np.sin(t)**2) + ((np.linalg.norm(u,2))**2 * np.sin(t)**2))
        
        return dd
    
    

    def create_elliptical_orbit(self, u, v, T_pts):   # creates one elliptic orbit for an input u,v
        """
        
        ### Creates an elliptical orbit defined by vectors u and v and center c
        ### Input arguments: the two orthogonal vectors u and v which define the plane on which the 
            ellipse lies and the center of the ellipse:
            u = array([u1, u2, u3]); v = array([v1, v2, v3]); c = array([c1, c2, c3])
        ### Returns: points that make up a discrete elliptical trajectory
        
        """        
        t_v = np.linspace(0, self.T_lim, T_pts)
        
        state = []
        
        for t in t_v:
            d = [self.c[0] + np.cos(t)*u[0] + np.sin(t)*v[0], self.c[1] + np.cos(t)*u[1] + np.sin(t)*v[1], self.c[2] + np.cos(t)*u[2] + np.sin(t)*v[2]]
            state = np.append(d, state)
        
        state = np.asarray(state)
        state = np.reshape(state, (int(state.shape[0]/3), 3))
        
        return state
    
    
    
    
    
    def plot_orbits(self):
        """
        
        ### Plots the orbits passed in the arguments
        ### Input arguments: the discrete orbital trajectories returned by create_elliptical_orbit
        
        """
        
        # A function just for Plotting all the orbits currently in question
        ax = plt.figure(figsize=(10,10)).add_subplot(projection='3d')
        # ax.text(self.c[0], self.c[1], self.c[2], s="center")
        ax.scatter3D(self.c[0], self.c[1], self.c[2], color=(1,0.2,0.2))
        plt.title('Elliptical orbits centered at = ({},{},{})'.format(self.c[0], self.c[1], self.c[2]))
        
        for i in range(self.n):  # considers only the first dimension
            state = self.create_elliptical_orbit(self.u[i, :],self.v[i, :], int(self.T_pts[i]))
            
            ax.plot(state[:,0], state[:,1], state[:,2], label='Orbit {}'.format(i))
            
            ax.scatter(self.u[:, 0], self.u[:, 1], self.u[:, 2], s=50, c='k', marker='o')
            ax.scatter(self.v[:, 0], self.v[:, 1], self.v[:, 2], s=50, c='k', marker='o')
            
            ax.legend()
            
        plt.show()
    
    
    
    
    
    def weight_matrix_at_t(self, t):            
        # DISTANCE ON AN ELLIPTICAL TRAJECTORY as a distance metric
    
        """
        ### Updates the Weight matrix self.W but using elliptical orbit (centered at [0,0,0]) for
            the distance metric instead of straight line distances
        ### input arguments: time point "t"    
        """
        
        for i in range(self.n):
            start = i
            for j in range(self.n):
                goal = j
                
                if start != goal:
                
                    t_start = int(t % self.T_pts[start])
                    t_goal  = int(t_start + self.delta_t)   # int(t % (self.T_pts[goal]) + self.delta_t)
                    
                    u_traj = self.ellipse_dynamics(t_start, self.u[start,:], self.v[start, :])
                    
                    v_traj = self.ellipse_dynamics(t_goal, self.u[goal,:], self.v[goal, :])
                    
                    # Now using the u_traj and v_traj vectors to compute the ellipse parametric they together define
                    t_eval = np.linspace(t_start, t_goal, 3)
                    t_eval = np.asarray(t_eval)
                    initial_pos = math.sqrt(((np.linalg.norm(u_traj,2))**2 * np.sin(t_eval[0])**2) + ((np.linalg.norm(v_traj,2))**2 * np.sin(t_eval[0])**2))
                    
                    d_init = [math.sqrt(((np.linalg.norm(u_traj,2))**2 * np.sin(t_start)**2) + ((np.linalg.norm(v_traj,2))**2 * np.sin(t_start)**2))]
                    
                    # print('initial condition = ', d_init)
                    distance = scipy.integrate.odeint(self.ellipse_dynamics_non_orthogonal, d_init, np.linspace(t_start, t_goal, 30),  args=(u_traj, v_traj))
                    
                    # dist_vec = scipy.integrate.odeint(self.ellipse_dynamics_non_orthogonal, initial_pos, t_eval, args=(u_traj, v_traj))
                    self.W[start][goal] = abs(distance[-1] - distance[0])
                    
                        
    
    
    
    
    def update_ant_paths(self, t_iter):  
        """
        
        ### adds the next most likely node (to a matrix of paths) that every ant will take based on the 
            "pseudo-random proportional scheme" - updates self.paths (does not return anything)
        ### inputs: iteration number t_iter
        
        """
        
        t = t_iter    
        
        """
            # For EVERY ANT do : 
            # a) find out which node the ant is currently at : node "i"
            # b) calculate the probabilities of going to every other self.n - 1 possible nodes "j" from 
                 current node "i"
            # c) retain node "j" with the highest prob. and advance to it (most likely outcome chosen)
            # d) add this "j" to matrix paths for ant "k" - make sure the same "j" does not already exist 
                 in the path so far, if so pick next highest probability "j" value instead
            # e) go on to the next ant (and repeat a - d)
            
        """
        
        for k in range(self.ants):
            
            # current node the k-th ant is at "i"
            i = int(self.paths[k, t])            # last value of the "k"-th ant's row in the matrix
            
            # probability vector to store the computations
            prob_vector = np.ones((self.n))
            prob_vector[i] = 0                   # zero probability of picking the same node again

            for j in range(self.n):
                if j != i:
                    prob_vector[j] = self.ph[i][j]**self.alpha * (1/self.W[i][j])**self.beta 
            
            # Now calculating the sum over all possible next nodes "i" 
            sum_prob = 0.
            for i_ in range(self.n):
                for j in range(self.n):
                    if j != i:
                        sum_prob += self.ph[i][j]**self.alpha * (1/self.W[i][j])**self.beta
            
            # Find probabilities (by dividing with the sum over all possible "i"'s)
            prob_vector = prob_vector/sum_prob
            prob_vector = prob_vector/np.max(prob_vector)
            
            count = 0
            j = -1
            flag = 1
            #print('path so far = ', self.paths[k, :t+1])
            while flag == 1:
                flag = 0
                
                if np.random.uniform(1,1) < self.q0:
                    # randomly pick next node to visit
                    j = np.random.randint(0, self.n-1)
                else:
                    # Pick the argmax node
                    j = np.argmax(prob_vector)
                
                # Checking if this position already exists in the path so far
                # CHECK FOR DUPLICITY
                for pos in self.paths[k, 0:t+1]:
                    if int(j) == int(pos):
                        prob_vector[j] = 0
                        flag = 1
                        break                     # exit this for loop: {pos in self.paths[k, 0:t+1]}
            
            # Updating the next node this "k"-th ant will go to at time "t+1"
            self.paths[k, t+1] = int(j)
            
            # update the time element until intersection with the other piece of debris!!
            self.delta = self.delta + self.delta_t
            



        
    def build_solution_paths(self):
        """
        
        ### Computes the metrics to calculate the solution paths as iteration best and best so far
        ### no inputs, no return values, updates self.path_ib and self.path_bs
        
        """
            
        t_n = 0
        while t_n < self.n-1:
            self.update_ant_paths(t_n)
            t_n += 1           
            self.delta += self.delta_t
            self.weight_matrix_at_t(self.delta)
        
        # Now that "self.ants" number of full solutions paths exist, we evaluate them all - assign them Eq. 2. metric phi() based scores
        phi_vector = np.zeros((self.ants))   # because there are self.ants number of paths
        
        for k in range(self.ants):   # "k" - identifies one ant
            phi = 0
            p = 0
            while p < self.n-1:     # complete solution paths are already built??
                phi += self.W[int(self.paths[k, int(p)]), int(self.paths[k, int(p+1)])]
                p += 1
            phi_vector[k] = phi
                
        # Find the best-phi path!
        path_ind = np.argmin(phi_vector)
        
        # Find if this best path has a shorter sum of weights compared to any calculated so far! If so, save it along with the optimum starting time!
        if np.min(phi_vector) < self.w_sum_bs:
            self.w_sum_bs = np.min(phi_vector)
            self.time_bs = self.delta - (self.n-1)*self.delta_t
        
        # This is the "iteration-best" or "ib" path!
        self.path_ib = self.paths[path_ind, :]
        
        # Find the sum of the weights in the iteration best path for comparison with the best so far path 
        p = 0
        weights_ib = 0
        while p < self.n-1:
            weights_ib += self.W[int(self.path_ib[int(p)]), int(self.path_ib[int(p+1)])]   # only one best-so-far path for all ants together
            p += 1
        
        p = 0
        weights_bs = 0
        while p < self.n-1:
            weights_bs += self.W[int(self.path_bs[int(p)]), int(self.path_bs[int(p+1)])]   # only one best-so-far path for all ants together
            p += 1
        
        
        # Compare to the best-so-far path
        # If the new iteration best has a lesser path cost than that of the best so far, then the 
        # new best_so_far (bs) becomes the current iteration best (ib)
        if weights_ib < weights_bs:
            self.path_bs = self.path_ib

        
    
            
            

    def plot_solution_paths(self):
        """
        ### Plotting the current iteration best and the best so far solution paths
        ### no input arguments, no return values
        ### plot using a self.n node (regular polygon) based graph
        """        
        ######################################################################################
        # PLOTTING THE BEST SO FAR PATH
        
        # interior angle of a regular self.n sided polygon:  360/self.n  degrees
        int_angle = 2*np.pi/orb.n                 # number of orbits: self.n
        # radius    = 1                           # (IMPLICIT since no scaling is applied)
        nodes = [[np.cos(int_angle * i), np.sin(int_angle * i)] for i in range(orb.n)]
        nodes = np.asarray(nodes)
        nodes = np.reshape(nodes, (orb.n, 2))
        
        plt.figure(figsize=(7,7))
        plt.scatter(nodes[:,0], nodes[:,1], c='k') 
        plt.scatter(nodes[int(orb.path_bs[0]),0], nodes[int(orb.path_bs[0]),1], s=150, c = 'r', )
        plt.title('Best Solution Path')
        
        x_pts = []
        y_pts = []
        for e in range(len(orb.path_bs)):
            x_pts = np.append(x_pts, nodes[int(orb.path_bs[e]), 0])
            y_pts = np.append(y_pts, nodes[int(orb.path_bs[e]), 1])
            plt.text(nodes[int(orb.path_bs[e]),0], nodes[int(orb.path_bs[e]),1], s=str(int(orb.path_bs[e])))    
        
        plt.plot(x_pts, y_pts, 'g-')
        plt.show() 
        
        ########################################################################################
        # NOW ITERATION BEST PATH
        
        plt.figure(figsize=(7,7))
        plt.scatter(nodes[:,0], nodes[:,1], c='k') 
        plt.scatter(nodes[int(orb.path_ib[0]),0], nodes[int(orb.path_ib[0]),1], s=150, c = 'r', )
        plt.title('Iteration Best Solution Path')
        
        x_pts = []
        y_pts = []
        for e in range(len(orb.path_ib)):
            x_pts = np.append(x_pts, nodes[int(orb.path_ib[e]), 0])
            y_pts = np.append(y_pts, nodes[int(orb.path_ib[e]), 1])
            plt.text(nodes[int(orb.path_ib[e]),0], nodes[int(orb.path_ib[e]),1], s=str(int(orb.path_ib[e])))    
        
        plt.plot(x_pts, y_pts, 'g-')
        plt.show() 
        
        
        
        
    
    def pheromone_trails_update(self):
        """
        
        ### Updating all the pheromone trails after every iteration
        ### no inputs, no return values
        
        """
        
        # First decrease all trails by the evaporation rate amount
        self.ph = self.ph * (1-self.rho)
        
        # Choose between the iteration best and the best so far solution paths
        if np.random.uniform(1,1) < self.q_ph:
            chosen_path = self.path_bs
        else:
            chosen_path = self.path_ib
        
        # next increase only those along the best so far path by a stipulated amount
        for node in range(len(chosen_path)-1):
            self.ph[int(chosen_path[node])][int(chosen_path[node+1])] += (1/self.W[int(chosen_path[node])][int(chosen_path[node+1])])*1000
            self.ph[int(chosen_path[node+1])][int(chosen_path[node])] += (1/self.W[int(chosen_path[node+1])][int(chosen_path[node])])*1000  # Since it is symmetric
            
        self.ph = np.clip(self.ph, self.ph_min, self.ph_max)    # clipped to stay within limits

        
        
    
    
    def plot_pheromone_trails(self):
        """
        
        ### Plotting the pheromone trails
        ### no input arguments, no return values
        ### plot using a self.n node (regular polygon) based graph
        
        """
        
        # interior angle of a regular self.n sided polygon:  360/self.n  degrees
        int_angle = 2*np.pi/self.n                # number of orbits: self.n
        radius    = 5                             
        nodes = [[radius*np.cos(int_angle * i), radius*np.sin(int_angle * i)] for i in range(self.n)]
        nodes = np.asarray(nodes)
        #print('nodes:')
        #print(nodes)
        
        # Normalizing the pheromone trail values for plotting - linewidth
        pher = self.ph             # assignment to a temp variable to leave the original undisturbed
        pher = pher/np.max(pher)
        
        plt.figure(figsize=(7,7))
        plt.scatter(nodes[:,0], nodes[:,1], c='k') 
        plt.scatter(nodes[int(orb.path_ib[0]),0], nodes[int(orb.path_ib[0]),1], s=150, c = 'r', )
        plt.title('Pheromone Trails')
        
        pt_x = []
        pt_y = []
        for e in range(self.n):
            plt.text(nodes[int(orb.path_ib[e]),0], nodes[int(orb.path_ib[e]),1], s=str(int(orb.path_ib[e])))    
            for e_rest in range(self.n):
                pt_x = []
                pt_y = []
                if(e != e_rest):
                    pt_x = nodes[e, 0]
                    pt_x = np.append(pt_x, nodes[e_rest, 0])
                    pt_y = nodes[e, 1]
                    pt_y = np.append(pt_y, nodes[e_rest, 1])
                    plt.plot(pt_x, pt_y, 'b', linewidth=pher[e][e_rest]*8)
        
        plt.show()    
        

    
    def create_orbit(self):
        
        orbit = np.zeros((self.n, self.T_pts, 3))   # 5 x 500 x 3  (3 : x,y,z coordinates)
        
        # for i in range(self.n):
        orbit = self.create_elliptical_orbit(np.reshape(self.u[i, :], (1,3)), np.reshape(self.v[i, :],(1,3)), self.T_lim[i])
        
        return orbits
    
    # def weight_matrix(self, u_vectors, v_vectors)


# In[5]:


# testing out the functions in class orbit
n = 9 
T_lim = 10
u_vectors = np.array([[3, 4, 5],  [3, -2, 2],  [1, 2, 5],    [1, 7, 0],  [7, -7, 14], [3, 4, 8],     [9, 7, 8],  [6, 1, 5],  [4, 7, 1]])
v_vectors = np.array([[4, -8, 4], [2, -3, -6], [2, -5, 8/5], [-7, 1, 1], [7, -7, -7], [-1, -1, 7/8], [3, -5, 1], [-10,10,10], [8,4,-60]])
center = [0,0,0]




# Calculating the orbital periods using Kepler's laws:
mu = 1           # Standard Gravitational constant {assumed to be a constant (accurately will be == GM)}
T_pts = np.zeros((n,))
for i in range(n):
    if np.linalg.norm(u_vectors[i,:], 2) > np.linalg.norm(v_vectors[i,:], 2):
        a = np.linalg.norm(u_vectors[i,:], 2)
        T_pt = np.ceil(2*math.pi*math.sqrt(a**3 / mu))
        T_pts[i] = int(T_pt)
        
    else:
        a = np.linalg.norm(u_vectors[i,:], 2)
        T_pt = np.ceil(2*math.pi*math.sqrt(a**3 / mu))
        T_pts[i] = int(T_pt)

        
        

print('T_pts vector = ', T_pts)
init_pos = v_vectors    # can be arbitrary and will affect final solution
n_ants = 100             # must be a big number to explore MANY more solutions
iters = 40             # Number of iterations: 250 - 300
rho = 0.9               # evaporation rate
tau0 = 10               # pheromone trail initialization value
alpha = 1               # for now == 1
beta = 5                # for now == 1   have to be unequal for the relative inluences to change the way paths are chosen
time_step = 25          # to update W in time steps differing by...
q_ph = 0.20             # threshold to pick between the best so far and the iteration best paths for pheromone trail update
ph_min=0.0000000000001  # lower limit on the pheromone trail values
ph_max = 15             # upper limit on the pheromone trail values
q0 = 0.35               # 0.25:0.75 prob of randomly picking the next node and picking the most likely one to visit




# initializing the problem
orb = orbit(center, T_lim, T_pts, u_vectors, v_vectors, n, init_pos, n_ants, iters, rho, tau0, alpha, beta, time_step, q_ph, ph_min, ph_max, q0)


# In[6]:


# Plotting the orbits
print('Plotting the test case orbits')
orb.plot_orbits()


# In[7]:


"""
MAIN.py function

"""



# Plotting the orbits
print('Plotting the test case orbits')
orb.plot_orbits()

# Plotting the initial pheromone trails
print('\nPlotting the (uniformly) initialized pheromone trails')
orb.plot_pheromone_trails()

# Running the algorithm!
for i in range(20):     # iterations chosen to be 500!
    
    print('\n\n\nITERATION: ', i)
    
    orb.build_solution_paths()   # compute "ant" number of solution paths
    
    orb.pheromone_trails_update()
    
    ## Update time step and use it to update the weight matrix 
    orb.delta = orb.delta + orb.delta_t
    orb.weight_matrix_at_t(orb.delta)
    
    if i % 4 == 0:
        print('\nUpdated pheromone trails')
        orb.plot_pheromone_trails()
        print('\nPlotting solution paths')
        orb.plot_solution_paths()
    
    # Re-initialize paths
    orb.paths = np.zeros((orb.ants, orb.n))
    orb.paths[:,0] = np.random.randint(0, orb.n-1, (orb.ants,))


print('\nPlotting the LAST updated pheromone trails')
orb.plot_pheromone_trails()
print('\nPlotting solution paths')
orb.plot_solution_paths()


print('\n\nBEST OVERALL solution path = ', orb.path_bs)
print('\nStarting time for optimum results (in multiples of time unit \"orb.delta_t\" elapsed from 0:00): ', orb.time_bs)


# In[8]:


# Calculating knot points for multiple shooting

knot_points = np.zeros((orb.n, 3))   # empty array for the knot points
knot_points_in_sequence = np.zeros((orb.n, 3))   # empty array for the knot points
t_start = orb.time_bs

for t in range(orb.n):
    junk_piece = int(orb.path_bs[t])    # the index of the orbit to be considered
    print('\nJunk piece number: {}'.format(junk_piece))
    t_pt = int((t_start + (orb.delta_t*t)) % orb.T_pts[junk_piece])
    print('time point in orbit: {}'.format(t_pt))
    u_orb = np.reshape(orb.u[junk_piece,:], (3,))
    v_orb = np.reshape(orb.v[junk_piece,:], (3,))
    knot_points[junk_piece,:] = orb.ellipse_dynamics(t_pt, u_orb, v_orb)
    knot_points_in_sequence[t,:] = orb.ellipse_dynamics(t_pt, u_orb, v_orb)
    
print('\nKnot points in sequence')
for t in range(orb.n):
    junk_piece = int(orb.path_bs[t])    # the index of the orbit to be considered
    print('\nOrbit: {}  Knot point: [{}, {}, {}]'.format(junk_piece, knot_points_in_sequence[t,0], knot_points[t,1], knot_points[t,2]))


# In[9]:


print(orb.W)
print(orb.ph)


# In[ ]:





# In[ ]:





# In[ ]:





# In[10]:


############################################## LOG ########################################################
# 1st Run:
# Parameters: 
"""
init_pos = v_vectors    # Later changed to the u_vectors
n = 9                   # 5 pieces of debris (5 u and v vectors also!)
n_ants = 25             # must be a big number to explore MANY more solutions
iters = 10              # Number of iterations
rho = 0.8               # evaporation rate
tau0 = 10               # pheromone trail initialization value
alpha = 1               # for now == 1
beta = 5                # for now == 1   have to be unequal for the relative inluences to change the way paths are chosen
time_step = 500         # to update W in time steps differing by...
q_ph = 0.23             # threshold to pick between the best so far and the iteration best paths for pheromone trail update
ph_min = 2              # lower limit on the pheromone trail values
ph_max = 15             # upper limit on the pheromone trail values
q0 = 0.5
"""
# IT 2: [7. 3. 5. 2. 4. 0. 6. 1. 8.]
# IT 3: [7. 3. 5. 2. 4. 0. 6. 1. 8.]
# IT 4: [7. 3. 5. 2. 6. 0. 4. 1. 8.]
# IT 5: [7. 3. 5. 2. 6. 0. 4. 1. 8.]
# IT 6: [7. 3. 5. 2. 6. 0. 4. 1. 8.]
# IT 7: [7. 3. 5. 2. 6. 0. 4. 1. 8.]
# IT 8: [7. 3. 5. 2. 6. 0. 4. 1. 8.]
# IT 9: 
# Best solution path =  [7. 3. 5. 2. 0. 6. 1. 4. 8.]




# 500 iteration run with slightly different parameters
# Best solution path =  [7. 3. 5. 2. 6. 0. 1. 4. 8.]



# How does the same thing work but with the u_vectors as the initial positions?
# optimized solution for delta = 50 and for 50 iterations



"""  MOST CREDIBLE SOLUTION SO FAR!  """
# For UPDATED TIME STEP with every iteration moving the system delta_t forward in time, 
# Best solution path =  [7. 3. 5. 4. 0. 2. 6. 1. 8.]


""" RANDOMIZED OUTPUTS FINALLY OBTAINED: AS THEY SHOULD BE """
# Different Outputs obtained for every run



"""  250 ITERATION SOLUTION FROM COLAB NOTEBOOK AFTER ELLIPTICAL DISTANCE METRIC UPDATE  """
# BEST OVERALL solution path =  [4. 5. 2. 1. 6. 7. 3. 0. 8.]
# Starting time for optimum results (in multiples of time unit "orb.delta_t" elapsed from 0:00):  1029025


# In[ ]:




