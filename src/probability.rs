use std::fmt::Debug;

use rand::{distributions::Uniform, thread_rng, Rng};

#[derive(Debug)]
pub enum ArgumentError {
    NonPositive,
    NotFinite,
    SizeMismatch,
}

pub struct Distribution<V: Copy> {
    distribution: Vec<(f32, V)>,
}

impl<V: Copy> Distribution<V> {
    pub fn new(items: Vec<V>, weights: Vec<f32>) -> Result<Self, ArgumentError> {
        if items.len() != weights.len() {
            return Err(ArgumentError::SizeMismatch);
        }

        Distribution::from(weights.into_iter().zip(items.into_iter()))
    }

    pub fn from(distribution: impl Iterator<Item = (f32, V)>) -> Result<Self, ArgumentError> {
        let mut distribution: Vec<_> = distribution.collect();

        let mut sum = 0.0;
        for (weight, _) in distribution.iter_mut() {
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

        for (weight, _) in distribution.iter_mut() {
            *weight /= sum;
        }

        Ok(Distribution { distribution })
    }
}

impl<K: Copy> Distribution<K> {
    pub fn sample(&self) -> K {
        let rnd = thread_rng().sample(Uniform::new(0.0, 1.0));

        let val_idx = self
            .distribution
            .binary_search_by(|&(weight, _)| weight.partial_cmp(&rnd).unwrap());

        let val_idx = match val_idx {
            Ok(val_idx) => val_idx,
            Err(val_idx) => val_idx,
        };

        self.distribution[val_idx].1
    }
}
