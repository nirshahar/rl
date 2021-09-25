use slotmap::SecondaryMap;

use crate::markov::{ActionError, Environment, MDPEnvironment, Reward, StateKey, MDP};

pub struct MDPPolicy {
    mdp: MDP,
    policy: SecondaryMap<StateKey, usize>,
}

impl MDPPolicy {
    fn do_action(&self, state: StateKey) -> Result<(StateKey, Reward), ActionError> {
        self.mdp.sample_transition(state, self.policy[state])
    }

    // Returns a value function
    pub fn td_zero(
        &mut self,
        epoch_size: usize,
        start_learning_rate: f32,
    ) -> SecondaryMap<StateKey, f32> {
        let mut value_mapping: SecondaryMap<StateKey, f32> = SecondaryMap::new();
        let mut learning_rate: SecondaryMap<StateKey, f32> = SecondaryMap::new();

        for state_key in self.mdp.states().keys() {
            value_mapping.insert(state_key, 0.0);
            learning_rate.insert(state_key, start_learning_rate);
        }

        for starting_state in self.mdp.states().keys() {
            let mut simulation = MDPEnvironment::new(&self.mdp, starting_state);

            for _ in 0..epoch_size {
                self.perform_update(&mut simulation, &mut value_mapping, &mut learning_rate);
            }
        }

        value_mapping
    }

    fn perform_update(
        &self,
        environment: &mut MDPEnvironment,
        value_mapping: &mut SecondaryMap<StateKey, f32>,
        learning_rate: &mut SecondaryMap<StateKey, f32>,
    ) {
        let cur_state = *environment.cur_state();

        let (next_state, reward) = self
            .do_action(cur_state)
            .expect("Action does not exist in the MDP");

        let expected_reward = reward.value() + self.mdp.gamma() * value_mapping[next_state];
        let lr = learning_rate[cur_state];

        value_mapping[cur_state] = (1.0 - lr) * value_mapping[cur_state] + lr * expected_reward;

        learning_rate[cur_state] = 1.0 / (1.0 / lr + 1.0);
    }
}
