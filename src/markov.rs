use std::ops::{Deref, DerefMut};

use crate::probability::Distribution;

pub trait Environment {
    fn perform_action(&mut self, action: usize) -> Reward;

    fn cur_state(&self) -> usize;
}

#[derive(Clone, Copy)]
pub struct Reward(pub f32);

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

pub struct MDP<const S: usize, const A: usize> {
    transitions: [[Distribution<(usize, Reward)>; A]; S],
    gamma: f32,
}

impl<const S: usize, const A: usize> MDP<S, A> {
    pub fn new(gamma: f32) -> Self {
        if !gamma.is_finite() {
            panic!("Cannot create an MDP with a NaN / Infinite gamma (discounting) value");
        }
        if gamma <= 0.0 || gamma >= 1.0 {
            panic!("The discounting factor gamma must be in the range (0,1) (inclusive)");
        }

        MDP {
            transitions: [0usize; S].map(|state| {
                [0; A].map(|_| Distribution::new(vec![(state, Reward(0.0))], vec![1.0]).unwrap())
            }),
            gamma,
        }
    }

    pub fn gamma(&self) -> f32 {
        self.gamma
    }

    pub fn set_transition(
        &mut self,
        state: usize,
        action: usize,
        target_distribution: Distribution<(usize, Reward)>,
    ) {
        self.transitions[state][action] = target_distribution;
    }

    pub fn sample_transition(&self, state: usize, action: usize) -> (usize, Reward) {
        self.transitions[state][action].sample()
    }

    pub fn num_states(&self) -> usize {
        self.transitions.len()
    }
}

pub struct MDPEnvironment<'a, const S: usize, const A: usize> {
    mdp: &'a MDP<S, A>,
    cur_state: usize,
}

impl<'a, const S: usize, const A: usize> MDPEnvironment<'a, S, A> {
    pub fn new(mdp: &'a MDP<S, A>, starting_state: usize) -> Self {
        MDPEnvironment {
            mdp,
            cur_state: starting_state,
        }
    }

    pub fn reset(&mut self, starting_state: usize) {
        self.cur_state = starting_state;
    }
}

impl<'a, const S: usize, const A: usize> Deref for MDPEnvironment<'a, S, A> {
    type Target = MDP<S, A>;

    fn deref(&self) -> &MDP<S, A> {
        &self.mdp
    }
}

impl<'a, const S: usize, const A: usize> Environment for MDPEnvironment<'a, S, A> {
    fn perform_action(&mut self, action: usize) -> Reward {
        let (new_state, reward) = self.mdp.sample_transition(self.cur_state, action);

        self.cur_state = new_state;

        reward
    }

    fn cur_state(&self) -> usize {
        self.cur_state
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
        const NUM_STATES: usize = 13;
        let prob_weight = 1.0;

        let num_steps = 10000;

        let mut mdp = MDP::<NUM_STATES, 2>::new(0.9);

        for state in 0..NUM_STATES {
            mdp.set_transition(
                state,
                0,
                Distribution::new(
                    vec![((state + 1) % NUM_STATES, Reward::new(state as f32))],
                    vec![prob_weight],
                )
                .unwrap(),
            );
            mdp.set_transition(
                state,
                1,
                Distribution::new(
                    vec![(
                        (state + NUM_STATES - 1) % NUM_STATES,
                        Reward::new(state as f32),
                    )],
                    vec![prob_weight],
                )
                .unwrap(),
            );
        }

        let mut mdp_environment = MDPEnvironment::new(&mdp, 0);

        for i in 0..num_steps {
            assert_eq!(mdp_environment.cur_state(), i % NUM_STATES);
            assert_eq!(
                mdp_environment.perform_action(0).value(),
                (i % NUM_STATES) as f32
            );
        }

        mdp_environment.reset(0);

        for i in 0..num_steps {
            assert_eq!(
                mdp_environment.cur_state(),
                (num_steps * NUM_STATES - i) % NUM_STATES
            );
            assert_eq!(
                mdp_environment.perform_action(1).value(),
                ((num_steps * NUM_STATES - i) % NUM_STATES) as f32
            );
        }

        mdp_environment.reset(0);

        for i in 0..num_steps {
            assert_eq!(mdp_environment.cur_state(), i % 2);
            mdp_environment.perform_action(i % 2).value();
        }
    }
}
