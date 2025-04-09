import gymnasium as gym
from JSAnimation.IPython_display import display_animation
from IPython.display import display
from IPython.display import HTML
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np


env_cartpole = gym.make('CartPole-v1', render_mode="rgb_array")

def display_frames_as_gif(frames):
    """
    Render jupyter embed video by taking a list of frames.
    referred to http://nbviewer.jupyter.org/github/patrickmineault/xcorr-notebooks/blob/master/Render%20OpenAI%20gym%20as%20GIF.ipynb
    """
    fig = plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi = 64)
    ax = plt.gca()
    patch = ax.imshow(frames[0])

    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])
        return (patch,)

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    
    plt.close()
    return anim

def goalAchieved(scores, episodes=500):
    """
    Prevent the game to run forever.
    """
    if np.mean(scores[-episodes:]) > 200 and scores.size > 500:
        return True
    else:
        return False
    
def test_cartpole(env, num, agent=None, **kwargs):
    """
    Generate game frames. Use other module such as matplotlib to generate other displayable format (see display_frames_as_gif.)
    """
    state, _ = env.reset()
    frames = []
    done = False
    count = 0
    for i in range(num):
        frames.append(env.render())
        if agent is None:
            a = env.action_space.sample()
        if isinstance(agent, object):
            a = int(agent.act(state, train=False))
        state, reward, done, info, _ = env.step(a)
        if done and i%41==0: # add 40 frames after done
            break
    frames.append(env.render())
    print(len(frames))
    return frames