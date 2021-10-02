use std::ops::Index;

use slotmap::SecondaryMap;

use crate::markov::{ActionError, Environment, MDPEnvironment, Reward, StateKey, MDP};

use crate::miscellaneous::ArgOrd;
use crate::probability::{throw_coin, Distribution};

pub struct MDPPolicy<'a> {
    mdp: &'a MDP,
    policy: SecondaryMap<StateKey, usize>,
}

impl<'a> Index<StateKey> for MDPPolicy<'a> {
    type Output = usize;

    fn index(&self, index: StateKey) -> &usize {
        &self.policy[index]
    }
}

impl<'a> MDPPolicy<'a> {
    pub fn new(mdp: &'a MDP, policy: SecondaryMap<StateKey, usize>) -> Self {
        Self { mdp, policy }
    }

    pub fn sample_action_result(&self, state: StateKey) -> Result<(StateKey, Reward), ActionError> {
        self.mdp.sample_transition(state, self.policy[state])
    }

    /// Returns a value function, using the TD(0) algorithm
    pub fn td_zero(&self, epoch_size: usize, learning_rate: f32) -> SecondaryMap<StateKey, f32> {
        let mut value_mapping: SecondaryMap<StateKey, f32> = SecondaryMap::new();

        for state_key in self.mdp.states().keys() {
            value_mapping.insert(state_key, 0.0);
        }

        for starting_state in self.mdp.states().keys() {
            let mut simulation = MDPEnvironment::new(&self.mdp, starting_state);

            for _ in 0..epoch_size {
                self.perform_tdzero_update(&mut simulation, &mut value_mapping, learning_rate);
            }
        }

        value_mapping
    }

    fn perform_tdzero_update(
        &self,
        environment: &mut MDPEnvironment,
        value_mapping: &mut SecondaryMap<StateKey, f32>,
        learning_rate: f32,
    ) {
        let cur_state = *environment.cur_state();

        let reward = environment.perform_action(&self[cur_state]);
        let next_state = *environment.cur_state();

        let expected_reward = reward.value() + self.mdp.gamma() * value_mapping[next_state];

        value_mapping[cur_state] =
            (1.0 - learning_rate) * value_mapping[cur_state] + learning_rate * expected_reward;
    }
}

impl MDP {
    pub fn perform_q_learning(
        &self,
        epoch_size: usize,
        learning_rate: f32,
        epsilon: f32,
    ) -> SecondaryMap<StateKey, Vec<f32>> {
        let mut q_func: SecondaryMap<StateKey, Vec<f32>> = SecondaryMap::new();
        let mut num_seen: SecondaryMap<StateKey, Vec<usize>> = SecondaryMap::new();

        for (state_key, state) in self.states() {
            let mut q = Vec::new();
            let mut seen = Vec::new();

            for _ in 0..state.transitions.len() {
                q.push(0.0);
                seen.push(0);
            }

            q_func.insert(state_key, q);
            num_seen.insert(state_key, seen);
        }

        for starting_state in self.states().keys() {
            let mut simulation = MDPEnvironment::new(&self, starting_state);

            for _ in 0..epoch_size {
                self.perform_q_update(
                    &mut simulation,
                    &mut q_func,
                    &mut num_seen,
                    learning_rate,
                    epsilon,
                );
            }
        }

        q_func
    }

    fn perform_q_update(
        &self,
        environment: &mut MDPEnvironment,
        q_function: &mut SecondaryMap<StateKey, Vec<f32>>,
        num_seen: &mut SecondaryMap<StateKey, Vec<usize>>,
        learning_rate: f32,
        epsilon: f32,
    ) {
        let cur_state = *environment.cur_state();

        let action = num_seen[cur_state].arg_min().unwrap();

        let reward = environment.perform_action(&action).value();
        let new_state = *environment.cur_state();

        let future_reward = if throw_coin(epsilon) {
            let len = q_function[new_state].len();
            let itr_item = 0..len;
            let itr_weight = (0..len).map(|_| 1.0);

            let distribution = Distribution::from(itr_item.zip(itr_weight)).unwrap();

            q_function[new_state][distribution.sample()]
        } else {
            q_function[new_state].max_val()
        };

        let expected_reward = reward + self.gamma() * future_reward;
        q_function[cur_state][action] =
            (1.0 - learning_rate) * q_function[cur_state][action] + learning_rate * expected_reward;
    }
}

#[cfg(test)]
mod tests {
    use slotmap::SecondaryMap;

    use crate::{
        markov::{Reward, MDP},
        probability::Distribution,
    };

    use super::MDPPolicy;

    #[test]
    fn test_cycle_td_zero() {
        let epsilon = 0.01;
        let num_states = 13;
        let reward = 1.0;
        let gamma = 0.9;
        let epoch_size = 250_000;
        let learning_rate = 0.001;

        let mut mdp = MDP::new(gamma);

        let mut states = Vec::new();
        for _ in 0..num_states {
            states.push(mdp.add_new_state());
        }

        for (i, &state) in states.iter().enumerate() {
            mdp.add_transition(
                state,
                Distribution::new(
                    vec![(states[(i + 1) % states.len()], Reward::new(reward))],
                    vec![1.0],
                )
                .unwrap(),
            );
            mdp.add_transition(
                state,
                Distribution::new(
                    vec![(
                        states[(i + states.len() - 1) % states.len()],
                        Reward::new(2.0),
                    )],
                    vec![2.0],
                )
                .unwrap(),
            );
        }

        let mut policy_map = SecondaryMap::new();
        for state in states {
            policy_map.insert(state, 0);
        }

        let policy = MDPPolicy::new(&mdp, policy_map);

        for (_, val) in policy.td_zero(epoch_size, learning_rate) {
            assert!((val - (reward / (1.0 - gamma))).abs() < epsilon);
        }
    }

    #[test]
    fn test_complex_td_zero() {
        let epsilon = 0.1;
        let gamma = 0.9;
        let epoch_size = 50_000_000;
        let learning_rate = 0.001;

        let mut mdp = MDP::new(gamma);

        let state = mdp.add_new_state();

        mdp.add_transition(
            state,
            Distribution::new(
                vec![(state, Reward(1.0)), (state, Reward(2.0))],
                vec![0.75, 0.25],
            )
            .unwrap(),
        );

        let mut policy_map = SecondaryMap::new();
        policy_map.insert(state, 0);

        let policy = MDPPolicy::new(&mdp, policy_map);

        let value_func = policy.td_zero(epoch_size, learning_rate);
        let val = value_func.get(state).unwrap();

        assert!(
            (val - (1.25) / (1.0 - gamma)).abs() < epsilon,
            "value at state computed: {:}, expected value is: {:}",
            val,
            (1.25) / (1.0 - gamma)
        );
    }
}
