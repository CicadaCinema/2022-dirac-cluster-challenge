#!/usr/bin/env python3

import numpy as np
from numpy.random import default_rng
from scipy.constants import gravitational_constant as G
import matplotlib.pyplot as plt
from time import perf_counter
import numba
from numba import cuda


N_YEARS = 0.1
FP_TYPE = np.float64
SEED = 12345


def random(num, a=0., b=1.):
    """
    Generate a random number between a and b
    """
    rng = default_rng(SEED)
    return rng.random(num, dtype=FP_TYPE)*(b-a) + a


def calc_stable_orbit(r, theta):
    """
    Given a position (in polar coords) calculate the velocity required to keep the body in a stable, circular orbit around the origin. Also returns Cartesian positions.

    Args:
        r: list of radii (one per body)
        theta: list of angles (one per body)

    Returns:
        List of Cartesian positions and velocities
    """
    v_mag = 1./np.sqrt(r)

    pos = np.zeros((r.shape[0], 2))
    vel = np.zeros((r.shape[0], 2))

    pos[:,0] = r*np.sin(theta)
    pos[:,1] = r*np.cos(theta)

    vel[:,0] = -v_mag*np.cos(theta)
    vel[:,1] =  v_mag*np.sin(theta)

    return pos, vel


def generate_random_star_system(num, min_radius=0.4, max_radius=20, min_mass=1./6000000, max_mass=1./1000):
    """
    Generate positions, velocities and masses for a star system similar to our own. Assumes star is massive (with mass=1) with zero velocity at coords (0,0).

    Args:
        num: Number of bodies to simulate (including central star)
        min_radius: sets the minimum radius possible
        max_radius: sets the maximum radius possible
        min_mass: sets the minimum mass possible
        max_mass: sets the maximum mass possible

    Returns:
        Lists of positions, velocities and masses, one per body
    """
    r = random(num, min_radius, max_radius)
    mass = random(num, min_mass, max_mass)

    theta = random(num, 0., np.pi)
    pos, vel = calc_stable_orbit(r, theta)

    # Add central star
    pos[0,:] = 0.
    vel[0,:] = 0.
    mass[0] = 1.

    return pos, vel, mass


def create_solar_system():
    """
    Generate positions, velocities and masses for our own solar system. Assumes star is massive (with mass=1) with zero velocity at coords (0,0).
    """

    # Solar system data from https://physics.stackexchange.com/questions/441608/solar-system-position-and-velocity-data

    names = ["Sun", "Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Neptune", "Uranus"]

    mass = np.array((1,1/6023600,1/408524,1/332946.038,1/3098710,1/1047.55,1/3499,1/22962,1/19352))
    r = np.array((0.1, 0.4, 0.7, 1, 1.5, 5.2, 9.5, 19.2, 30.1))
    theta = random(len(r), 0., np.pi)
    pos, vel = calc_stable_orbit(r, theta)

    pos[0,:] = 0.
    vel[0,:] = 0.
    mass[0] = 1.

    return pos, vel, mass

"""
@cuda.jit
def vec_gpu(acc, pos, mass, epsilon):
    i = cuda.grid(1)
    if i < pos.size:
        r = pos[:, :] - pos[i, :]
        acc[i, :] = np.sum(r.T * mass / (r[:, 0] ** 2 + r[:, 1] ** 2 + epsilon ** 2) ** (1.5), axis=1)


def calc_acc(acc, pos, mass):
    epsilon = 1.1*np.power(len(pos), -0.48)

    threadsperblock = 256
    blockspergrid = (pos.size + threadsperblock - 1)
    print("threadsperblock, blockspergrid")
    print(threadsperblock, blockspergrid)
    #vec_gpu[blockspergrid, threadsperblock](acc, pos, mass, epsilon)

    for i in range(len(pos)):
        r = pos[:, :] - pos[i, :]
        acc[i, :] = np.sum(r.T * mass / (r[:, 0] ** 2 + r[:, 1] ** 2 + epsilon ** 2) ** (1.5), axis=1)
"""


def calc_acc(acc, pos, mass):
    epsilon_squared = (1.1*np.power(len(pos), -0.48)) ** 2
    for i in range(len(pos)):
        r = pos[:, :] - pos[i, :]

        # dr=(r[:,0]**2 + r[:,1]**2+ epsilon_squared)**(1.5)

        # acc[i,:] = np.sum(r.T*mass/dr, axis=1)
        rx = 0
        ry = 0
        for j in range(len(pos)):
            rx += r[j, 0] * mass[j] / (r[j, 0] ** 2 + r[j, 1] ** 2 + epsilon_squared) ** (1.5)
            ry += r[j, 1] * mass[j] / (r[j, 0] ** 2 + r[j, 1] ** 2 + epsilon_squared) ** (1.5)

        acc[i, 0] = rx
        acc[i, 1] = ry
        # acc_sum[:]=0
        # for j in numba.prange(len(pos))
        #    acc_sum[:]=
        # acc[i,:]=acc_sum[:]
        # acc[i,:] = np.sum(r.T*mass/(r[:,0]**2 + r[:,1]**2 + epsilon**2)**(1.5), axis=1)

def advance_pos(acc, pos, pos_prev, pos_temp, dt):
    """
    Advance positions of all bodies based on previous position and current acceleration. This uses the Verlet method which is useful for simulations of simple equations of motion. It is 4th-order accurate in dt (compared to only 1st-order for the Euler method) making it suitable for longer-running simulations like this one.

    Args:
        acc: list of accelerations
        pos: current positions of all bodies
        pos_prev: previous positions of all bodies
        pos_temp: array to temporarily hold positions during calculation
        dt: simulation timestep
    """
    pos_temp[:,:] = pos[:,:]
    pos[:,:] = 2.0 * pos[:,:] - pos_prev[:,:] + acc[:,:] * dt**2
    pos_prev[:,:] = pos_temp[:,:]


def run(is_solar_system=False, plot=False, n_particles=8):
    """
    Run simulation
    """

    if is_solar_system:
        print("Running regular solar system")
        pos, vel, mass = create_solar_system()
    else:
        print(f"Running with {n_particles} particles")
        pos, vel, mass = generate_random_star_system(n_particles)

    dt = 0.01
    total_time = 10000*dt
    # total_time = N_YEARS*2*np.pi
    pos_tracker = []

    # Load initial conditions
    acc = np.zeros_like(pos)
    pos_temp = np.zeros_like(pos)
    pos_prev = np.zeros_like(pos)

    # Figure out initial acceleration
    calc_acc(acc, pos, mass)

    # Calculate previous position from velocity
    pos_prev[:] = pos - vel*dt - 0.5*acc*dt**2

    start = perf_counter()

    t = 0
    while t < total_time:
        pos_tracker += [pos.copy()]
        calc_acc(acc, pos, mass)
        advance_pos(acc, pos, pos_prev, pos_temp, dt)
        t += dt

    end = perf_counter()
    completion_time = end-start
    print(f"Time to complete: {completion_time:.4f} s")

    if plot:
        positions_for_plotting = np.array(pos_tracker)

        fig, ax = plt.subplots()
        xmin, xmax = 0.0, 0.0
        ymin, ymax = 0.0, 0.0
        for i in range(len(pos)):
            xdata = positions_for_plotting[:,i,0]
            ydata = positions_for_plotting[:,i,1]
            xmin = min(np.min(xdata), xmin)
            xmax = max(np.max(xdata), xmax)
            ymin = min(np.min(ydata), ymin)
            ymax = max(np.max(ydata), ymax)
            ax.plot(xdata, ydata)

        xmax = max(abs(xmax), abs(xmin))
        ymax = max(abs(ymax), abs(ymin))
        xmin = -xmax
        ymin = -ymax
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)
        plt.savefig("out.png")

    return completion_time


if __name__ == "__main__":
    #numba.set_num_threads(4)

    n_particle_range = [8, 16, 32, 64, 128]
    #runtimes = [run(n_particles=n) for n in n_particle_range]
    run(plot=True, n_particles=30)
    print(n_particle_range)
    #print(runtimes)