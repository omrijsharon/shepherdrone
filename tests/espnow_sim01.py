import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
import networkx as nx



# fix numpy seed
np.random.seed(0)

# Number of agents
n_agents = 200
com_radius = 3  # km
max_radius = 15 # km
# Initialize the message and start with agent 23
sender = 14
receiver = 32
# assert reciver and sender must be within the range of n_agents
assert receiver < n_agents and sender < n_agents, 'receiver and sender must be within the range of n_agents'

# # Generate random radii and angles
# radii = np.random.uniform(0, max_radius, n_agents)  # in km
# angles = np.random.uniform(0, 2*np.pi, n_agents)
#
# # Convert polar coordinates to Cartesian coordinates
# x = radii * np.cos(angles)
# y = radii * np.sin(angles)

x = np.random.uniform(-max_radius, max_radius, n_agents)
y = np.random.uniform(-max_radius, max_radius, n_agents)

# Initialize agent colors ('red' for not transmitted, 'green' for transmitting, 'blue' for already transmitted)
colors = ['red'] * n_agents
colors_next = ['red'] * n_agents

# Initialize set to keep track of agents that have transmitted the message
transmitted_agents = set()


# Function to plot agents at each step
def plot_agents(step, sender=23, receiver=42):
    plt.clf()

    for i, (x_coord, y_coord, color) in enumerate(zip(x, y, colors)):
        plt.scatter(x_coord, y_coord, c=color)
        plt.text(x_coord, y_coord, str(i), fontsize=8)

        # circle = plt.Circle((x_coord, y_coord), com_radius, color=color, fill=True, alpha=0.05)
        circle = plt.Circle((x_coord, y_coord), 0.5, color=color, fill=True, alpha=0.2)
        plt.gca().add_artist(circle)

    plt.title(f'Step {step}: Message Progress from Agent {sender} to Agent {receiver}')
    plt.xlabel('X Coordinate (km)')
    plt.ylabel('Y Coordinate (km)')
    plt.axis('equal')
    plt.grid(True)
    plt.pause(0.0000001)


# Function to plot agents at each step
def plot_state(state, step, sender=23, receiver=42):
    plt.clf()

    for i, (x_coord, y_coord, s_coord) in enumerate(zip(x, y, state)):
        if s_coord[0] == 1 and s_coord[1] == 0:
            color = 'blue'
        elif s_coord[0] == 0 and s_coord[1] == 1:
            color = 'green'
        else:
            color = 'red'
        plt.scatter(x_coord, y_coord, c=color)
        plt.text(x_coord, y_coord, str(i), fontsize=8)

        # circle = plt.Circle((x_coord, y_coord), com_radius, color=color, fill=True, alpha=0.05)
        circle = plt.Circle((x_coord, y_coord), 0.5, color=color, fill=True, alpha=0.2)
        plt.gca().add_artist(circle)

    plt.title(f'Step {step}: Message Progress from Agent {sender} to Agent {receiver}')
    plt.xlabel('X Coordinate (km)')
    plt.ylabel('Y Coordinate (km)')
    plt.axis('equal')
    plt.grid(True)
    plt.pause(0.0000001)


def create_connectivity_matrix(x, y, com_radius):
    """
    Create a sparse matrix representing the connectivity between agents.

    Parameters:
    - x, y: Coordinates of the agents
    - com_radius: Communication radius

    Returns:
    - csr_matrix: Sparse matrix where a 1 at (i, j) indicates a connection from agent i to agent j
    """
    n_agents = len(x)
    data = []
    rows = []
    cols = []

    for i, (xi, yi) in enumerate(zip(x, y)):
        for j, (xj, yj) in enumerate(zip(x, y)):
            if i != j:  # An agent is not connected to itself
                distance = np.sqrt((xi - xj) ** 2 + (yi - yj) ** 2)

                if distance <= com_radius:
                    data.append(1)
                    rows.append(i)
                    cols.append(j)

    return csr_matrix((data, (rows, cols)), shape=(n_agents, n_agents))


# Define the function to plot the connectivity graph
def plot_connectivity_graph(connectivity_matrix, x, y):
    """
    Plot the connectivity graph based on the sparse matrix and agent coordinates.

    Parameters:
    - connectivity_matrix: csr_matrix representing agent connectivity
    - x, y: Coordinates of the agents
    """
    G = nx.from_numpy_array(connectivity_matrix, create_using=nx.DiGraph())

    # Create a dictionary of positions from x, y coordinates
    pos = {i: (x_coord, y_coord) for i, (x_coord, y_coord) in enumerate(zip(x, y))}

    plt.figure(figsize=(10, 10))

    nx.draw(G, pos, with_labels=True, node_color='blue', font_weight='bold', node_size=700, font_size=18, alpha=0.5)
    plt.title("Connectivity Graph of Agents")
    plt.show()


connectivity_matrix = create_connectivity_matrix(x, y, com_radius).toarray()
# Use the previously created connectivity_matrix to plot the graph
plot_connectivity_graph(connectivity_matrix, x, y)

state = np.zeros((n_agents, 2), dtype=int)
state[sender, 1] = 1

step = 0
# Loop through steps to propagate the message
# Break the loop when the message reaches the receiver
while state[receiver, 1] == 0:
    # Update memory based on current activity
    state[:, 0] = np.bitwise_or(state[:, 0], state[:, 1])

    # Predict next activity based on connectivity
    next_state_activity = np.clip(connectivity_matrix.dot(state[:, 1]), 0, 1)

    # Inhibit next activity for agents that have already forwarded the message
    next_state_activity = np.bitwise_xor(next_state_activity, state[:, 0])

    # Update the state matrix
    state[:, 1] = next_state_activity
    step += 1
    plot_state(state, step, sender=sender, receiver=receiver)


message_received = False
step = 0

# Set the initial sender to 'green'
colors[sender] = 'green'
colors_next[sender] = 'green'
transmitted_agents.add(sender)

plt.figure(figsize=(10, 10))
plot_agents(step)

# Loop through steps to propagate the message
while not message_received:
    step += 1

    # Find agents within com_radius km of the sender (or any agent that is currently transmitting)
    for i, (xi, yi) in enumerate(zip(x, y)):
        if colors[i] == 'green':
            for j, (xj, yj) in enumerate(zip(x, y)):
                if i != j:
                    distance = np.sqrt((xi - xj) ** 2 + (yi - yj) ** 2)

                    if distance <= com_radius and j not in transmitted_agents:
                        colors_next[j] = 'green'
                        transmitted_agents.add(j)

                        if j == receiver:
                            message_received = True
            colors_next[i] = 'blue'
    colors = colors_next.copy()
    plot_agents(step)
    print(step)
plt.show()
