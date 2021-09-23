use slotmap::{new_key_type, SlotMap};

struct State {}

new_key_type! { struct StateKey; }

struct MDP {
    states: SlotMap<StateKey, State>,
}

impl MDP {
    fn add_state(&mut self, state: State) -> StateKey {
        self.states.insert(state)
    }
}
