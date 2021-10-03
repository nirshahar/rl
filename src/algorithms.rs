use std::ops::Index;

use crate::markov::{Environment, MDPEnvironment, Reward, MDP};

use crate::miscellaneous::ArgOrd;
use crate::probability::{throw_coin, Distribution};

pub struct MDPPolicy<'a, const S: usize, const A: usize> {
    mdp: &'a MDP<S, A>,
    policy: [usize; S],
}

impl<'a, const S: usize, const A: usize> Index<usize> for MDPPolicy<'a, S, A> {
    type Output = usize;

    fn index(&self, index: usize) -> &usize {
        &self.policy[index]
    }
}

impl<'a, const S: usize, const A: usize> MDPPolicy<'a, S, A> {
    pub fn new(mdp: &'a MDP<S, A>, policy: [usize; S]) -> Self {
        Self { mdp, policy }
    }

    pub fn from_q(mdp: &'a MDP<S, A>, q_func: [[f32; A]; S]) -> Self {
        let mut action_chosen = [0; S];
        for (state, &action_values) in q_func.iter().enumerate() {
            action_chosen[state] = action_values.arg_max();
        }

        MDPPolicy::new(mdp, action_chosen)
    }

    pub fn sample_action_result(&self, state: usize) -> (usize, Reward) {
        self.mdp.sample_transition(state, self.policy[state])
    }

    /// Returns a value function, using the TD(0) algorithm
    pub fn td_zero(&self, epoch_size: usize, learning_rate: f32) -> [f32; S] {
        let mut value_mapping = [0.0; S];

        for starting_state in 0..self.mdp.num_states() {
            let mut simulation = MDPEnvironment::new(&self.mdp, starting_state);

            for _ in 0..epoch_size {
                self.perform_tdzero_update(&mut simulation, &mut value_mapping, learning_rate);
            }
        }

        value_mapping
    }

    fn perform_tdzero_update(
        &self,
        environment: &mut MDPEnvironment<S, A>,
        value_mapping: &mut [f32; S],
        learning_rate: f32,
    ) {
        let cur_state = environment.cur_state();

        let reward = environment.perform_action(self[cur_state]);
        let next_state = environment.cur_state();

        let expected_reward = reward.value() + self.mdp.gamma() * value_mapping[next_state];

        value_mapping[cur_state] =
            (1.0 - learning_rate) * value_mapping[cur_state] + learning_rate * expected_reward;
    }
}

impl<const S: usize, const A: usize> MDP<S, A> {
    pub fn perform_q_learning(
        &self,
        epoch_size: usize,
        learning_rate: f32,
        epsilon: f32,
    ) -> [[f32; A]; S] {
        let mut q_func = [[0.0; A]; S];
        let mut num_seen = [[0usize; A]; S];

        for starting_state in 0..self.num_states() {
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
        environment: &mut MDPEnvironment<S, A>,
        q_function: &mut [[f32; A]; S],
        num_seen: &mut [[usize; A]; S],
        learning_rate: f32,
        epsilon: f32,
    ) {
        let cur_state = environment.cur_state();

        let action = num_seen[cur_state].arg_min();

        let reward = environment.perform_action(action).value();
        let new_state = environment.cur_state();

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

    use crate::{
        markov::{Reward, MDP},
        probability::Distribution,
    };

    use super::MDPPolicy;

    #[test]
    fn test_cycle_td_zero() {
        const NUM_STATES: usize = 13;
        let epsilon = 0.01;
        let reward = 1.0;
        let gamma = 0.9;
        let epoch_size = 250_000;
        let learning_rate = 0.001;

        let mut mdp = MDP::<NUM_STATES, 2>::new(gamma);

        for state in 0..NUM_STATES {
            mdp.set_transition(
                state,
                0,
                Distribution::new(
                    vec![((state + 1) % NUM_STATES, Reward::new(reward))],
                    vec![1.0],
                )
                .unwrap(),
            );
            mdp.set_transition(
                state,
                1,
                Distribution::new(
                    vec![((state + NUM_STATES - 1) % NUM_STATES, Reward::new(2.0))],
                    vec![2.0],
                )
                .unwrap(),
            );
        }

        let policy_map = [0; NUM_STATES];

        let policy = MDPPolicy::new(&mdp, policy_map);

        for val in policy.td_zero(epoch_size, learning_rate) {
            assert!((val - (reward / (1.0 - gamma))).abs() < epsilon);
        }
    }

    #[test]
    fn test_complex_td_zero() {
        let epsilon = 0.1;
        let gamma = 0.9;
        let epoch_size = 50_000_000;
        let learning_rate = 0.001;

        let mut mdp = MDP::<1, 2>::new(gamma);

        mdp.set_transition(
            0,
            0,
            Distribution::new(vec![(0, Reward(1.0)), (0, Reward(2.0))], vec![0.75, 0.25]).unwrap(),
        );

        let policy_map = [0; 1];

        let policy = MDPPolicy::new(&mdp, policy_map);

        let value_func = policy.td_zero(epoch_size, learning_rate);
        let val = value_func.get(0).unwrap();

        assert!(
            (val - (1.25) / (1.0 - gamma)).abs() < epsilon,
            "value at state computed: {:}, expected value is: {:}",
            val,
            (1.25) / (1.0 - gamma)
        );
    }
}
