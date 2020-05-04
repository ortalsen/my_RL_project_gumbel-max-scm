# %%
"""
This notebook is used to generate the true MDP parameters, by sampling 10k times from the transitions of every state/action pair using the underlying simulator.  In the paper, this is used to construct an "optimal" physician policy against which we compare the RL policy
"""

# %%
import itertools as it
import numpy as np
from sepsisSimDiabetes.State import State
from sepsisSimDiabetes.Action import Action
from sepsisSimDiabetes.MDP import MDP
import pickle as pkl
from tqdm import tqdm_notebook as tqdm

# %%
# Samples per component/state/action pair
np.random.seed(1)
n_iter = 10000
n_actions = Action.NUM_ACTIONS_TOTAL
n_states = State.NUM_OBS_STATES
n_components = 2

states = range(n_states)
actions = range(n_actions)
components = [0, 1]

# %%
## TRANSITION MATRIX
tx_mat = np.zeros((n_components, n_actions, n_states, n_states))

# Not used, but a required argument
dummy_pol = np.ones((n_states, n_actions)) / n_actions

# %%
# WARNING: This takes about 2 hours to run on my laptop
tx_mat = np.zeros((n_components, n_actions, n_states, n_states))
for (c, s0, a, _) in tqdm(it.product(components, states, actions, range(n_iter)), total=n_components*n_actions*n_states*n_iter):
    this_mdp = MDP(init_state_idx=s0, policy_array=dummy_pol, p_diabetes=c)
    r = this_mdp.transition(Action(action_idx=a))
    s1 = this_mdp.state.get_state_idx()
    tx_mat[c, a, s0, s1] += 1

# %%
est_tx_mat = tx_mat / n_iter
# Extra normalization
est_tx_mat /= est_tx_mat.sum(axis=-1, keepdims=True)

# %%
## REWARD MATRIX
np.random.seed(1)

# Calculate the reward matrix explicitly, only based on state
est_r_mat = np.zeros_like(est_tx_mat)
for s1 in states:
    this_mdp = MDP(init_state_idx=s1, policy_array=dummy_pol, p_diabetes=1)
    r = this_mdp.calculateReward()
    est_r_mat[:, :, :, s1] = r

# %%
## PRIOR ON INITIAL STATE
np.random.seed(1)
prior_initial_state = np.zeros((n_components, n_states))

for c in components:
    this_mdp = MDP(p_diabetes=c)
    for _ in range(n_iter):
        s = this_mdp.get_new_state().get_state_idx()
        prior_initial_state[c, s] += 1
    
prior_initial_state = prior_initial_state / n_iter
# Extra normalization
prior_initial_state /= prior_initial_state.sum(axis=-1, keepdims=True)

# %%
prior_mx_components = np.array([0.8, 0.2])

# %%
mat_dict = {"tx_mat": est_tx_mat,
            "r_mat": est_r_mat,
            "p_initial_state": prior_initial_state,
            "p_mixture": prior_mx_components}
with open('data/diab_txr_mats-replication.pkl', 'wb') as f:
    pkl.dump(mat_dict, f)