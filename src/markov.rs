use std::ops::{Deref, DerefMut};

use slotmap::{new_key_type, SlotMap};

use crate::probability::Distribution;

#[derive(Debug)]
pub enum ActionError {
    ActionDoesNotExist,
}

pub trait Environment<S, A> {
    fn perform_action(&mut self, action: &A) -> Reward;

    fn cur_state(&self) -> &S;
}

#[derive(Clone, Copy)]
pub struct Reward(f32);

impl Reward {
    pub fn new(val: f32) -> Reward {
        Reward(val)
    }

    pub fn value(&self) -> f32 {
        self.0
    }
}

impl Deref for Reward {
    type Target = f32;

    fn deref(&self) -> &f32 {
        &self.0
    }
}

impl DerefMut for Reward {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

pub struct State {
    transitions: Vec<Distribution<(StateKey, Reward)>>,
}

impl State {
    pub fn new() -> State {
        State {
            transitions: Vec::new(),
        }
    }

    fn do_action(&self, action: usize) -> Result<(StateKey, Reward), ActionError> {
        if let Some(distribution) = self.transitions.get(action) {
            Ok(distribution.sample())
        } else {
            Err(ActionError::ActionDoesNotExist)
        }
    }
}

new_key_type! { pub struct StateKey; }

pub struct MDP {
    states: SlotMap<StateKey, State>,
    gamma: f32,
}

impl MDP {
    pub fn new(gamma: f32) -> MDP {
        if !gamma.is_finite() {
            panic!("Cannot create an MDP with a NaN / Infinite gamma (discounting) value");
        }
        if gamma <= 0.0 || gamma >= 1.0 {
            panic!("The discounting factor gamma must be in the range (0,1) (inclusive)");
        }

        MDP {
            states: SlotMap::with_key(),
            gamma: gamma,
        }
    }

    pub fn gamma(&self) -> f32 {
        self.gamma
    }

    pub fn add_state(&mut self, state: State) -> StateKey {
        self.states.insert(state)
    }

    pub fn add_new_state(&mut self) -> StateKey {
        self.add_state(State::new())
    }

    pub fn add_transition(
        &mut self,
        state: StateKey,
        target_distribution: Distribution<(StateKey, Reward)>,
    ) {
        self.states[state].transitions.push(target_distribution);
    }

    pub fn sample_transition(
        &self,
        state: StateKey,
        action: usize,
    ) -> Result<(StateKey, Reward), ActionError> {
        self.states[state].do_action(action)
    }

    pub fn states(&self) -> &SlotMap<StateKey, State> {
        &self.states
    }
}

pub struct MDPEnvironment<'a> {
    mdp: &'a MDP,
    cur_state: StateKey,
}

impl<'a> MDPEnvironment<'a> {
    pub fn new(mdp: &'a MDP, starting_state: StateKey) -> MDPEnvironment<'a> {
        MDPEnvironment {
            mdp,
            cur_state: starting_state,
        }
    }

    pub fn reset(&mut self, starting_state: StateKey) {
        self.cur_state = starting_state;
    }
}

impl<'a> Deref for MDPEnvironment<'a> {
    type Target = MDP;

    fn deref(&self) -> &MDP {
        &self.mdp
    }
}

impl<'a> Environment<StateKey, usize> for MDPEnvironment<'a> {
    fn perform_action(&mut self, action: &usize) -> Reward {
        let action = *action;

        let (new_state, reward) = self
            .mdp
            .sample_transition(self.cur_state, action)
            .expect("Action does not exist in the MDP");

        self.cur_state = new_state;

        reward
    }

    fn cur_state(&self) -> &StateKey {
        &self.cur_state
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        markov::{Environment, MDPEnvironment, Reward},
        probability::Distribution,
    };

    use super::MDP;

    #[test]
    fn test_cycle_ddp() {
        let prob_weight = 1.0;
        let num_states = 13;
        let num_steps = 10000;

        let mut mdp = MDP::new(0.9);

        let mut states = Vec::new();
        for _ in 0..num_states {
            states.push(mdp.add_new_state());
        }

        for (i, &state) in states.iter().enumerate() {
            mdp.add_transition(
                state,
                Distribution::new(
                    vec![(states[(i + 1) % states.len()], Reward::new(i as f32))],
                    vec![prob_weight],
                )
                .unwrap(),
            );
            mdp.add_transition(
                state,
                Distribution::new(
                    vec![(
                        states[(i + states.len() - 1) % states.len()],
                        Reward::new(i as f32),
                    )],
                    vec![prob_weight],
                )
                .unwrap(),
            );
        }

        let mut mdp_environment = MDPEnvironment::new(&mdp, states[0]);

        for i in 0..num_steps {
            assert_eq!(*mdp_environment.cur_state(), states[i % states.len()]);
            assert_eq!(
                mdp_environment.perform_action(&0).value(),
                (i % states.len()) as f32
            );
        }

        mdp_environment.reset(states[0]);

        for i in 0..num_steps {
            assert_eq!(
                *mdp_environment.cur_state(),
                states[(num_steps * states.len() - i) % states.len()]
            );
            assert_eq!(
                mdp_environment.perform_action(&1).value(),
                ((num_steps * states.len() - i) % states.len()) as f32
            );
        }

        mdp_environment.reset(states[0]);

        for i in 0..num_steps {
            assert_eq!(*mdp_environment.cur_state(), states[i % 2]);
            mdp_environment.perform_action(&(i % 2)).value();
        }
    }
}
