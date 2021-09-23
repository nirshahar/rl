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
            sum += temp;
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
    pub fn gen_element(&mut self) -> K {
        let rnd = self.rng.sample(Uniform::new(0.0, 1.0));

        let val_idx = self
            .distribution
            .binary_search_by(|&(weight, val)| weight.partial_cmp(&rnd).unwrap());

        let val_idx = val_idx.unwrap_or_else(|idx| idx - 1) + 1;

        self.distribution[val_idx].1
    }
}
