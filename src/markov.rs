use slotmap::{new_key_type, SlotMap};

use crate::probability::Distribution;

#[derive(Debug)]
pub enum ActionError {
    ActionDoesNotExist,
}

#[derive(Clone, Copy)]
pub struct Reward(f32);

impl Reward {
    fn new(val: f32) -> Reward {
        Reward(val)
    }

    fn value(&self) -> f32 {
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

    fn do_action(&mut self, action: usize) -> Result<(StateKey, Reward), ActionError> {
        if let Some(distribution) = self.transitions.get_mut(action) {
            Ok(distribution.sample())
        } else {
            Err(ActionError::ActionDoesNotExist)
        }
    }
}

new_key_type! { pub struct StateKey; }

pub struct MDP {
    states: SlotMap<StateKey, State>,
}

impl MDP {
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
        &mut self,
        state: StateKey,
        action: usize,
    ) -> Result<(StateKey, Reward), ActionError> {
        self.states[state].do_action(action)
    }
}
}
