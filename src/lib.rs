use slotmap::{new_key_type, SlotMap};

pub mod probability {
    use rand::{distributions::Uniform, prelude::ThreadRng, thread_rng, Rng};

    pub enum ArgumentError {
        NonPositive,
        NotFinite,
    }

    pub struct Distribution<V> {
        distribution: Vec<(f32, V)>,
        rng: ThreadRng,
    }

    impl<V> Distribution<V> {
        pub fn new(distribution: impl Iterator<Item = (f32, V)>) -> Result<Self, ArgumentError> {
            let mut distribution: Vec<_> = distribution.collect();

            let mut sum = 0.0;
            for (weight, val) in distribution.iter_mut() {
                if !weight.is_finite() {
                    return Err(ArgumentError::NotFinite);
                }
                if *weight <= 0.0 {
                    return Err(ArgumentError::NonPositive);
                }
                let temp = *weight;
                *weight += sum;
                sum += *weight;
            }

            for (weight, val) in distribution.iter_mut() {
                *weight /= sum;
            }

            Ok(Distribution {
                distribution,
                rng: thread_rng(),
            })
        }
    }

    impl<K: Copy> Distribution<K> {
        pub fn gen_random_element(&mut self) -> K {
            let rnd = self.rng.sample(Uniform::new(0.0, 1.0));

            let val_idx = self
                .distribution
                .binary_search_by(|&(weight, val)| weight.partial_cmp(&rnd).unwrap());

            let val_idx = val_idx.unwrap_or_else(|idx| idx);

            self.distribution[val_idx].1
        }
    }
}

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
