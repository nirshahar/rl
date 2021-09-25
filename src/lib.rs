pub mod probability;

pub mod algorithms;

pub mod markov;

#[cfg(test)]
mod tests {
    use crate::probability::Distribution;

    fn test_given_distribution(items: Vec<usize>, weights: Vec<f32>) {
        let weight_sum: f64 = weights.iter().sum::<f32>() as f64;

        let distribution = Distribution::new(items, weights.clone()).unwrap();

        let mut sampled_distribution = Vec::new();
        for _ in 0..weights.len() {
            sampled_distribution.push(0.0);
        }

        let s = 50000000;
        let epsilon = 0.001;

        for _ in 0..s {
            sampled_distribution[distribution.sample()] += 1.0;
        }

        for i in 0..sampled_distribution.len() {
            sampled_distribution[i] /= s as f64;
        }

        for i in 0..sampled_distribution.len() {
            let prob = sampled_distribution[i];
            let expected_prob = weights[i] as f64 / weight_sum;

            assert!(
                (expected_prob - prob).abs() < epsilon,
                "prob diff for current item: {:?}, prob: {:#?}",
                (expected_prob - prob).abs(),
                sampled_distribution
            );
        }
    }

    #[test]
    fn test_simple_distribution() {
        let items = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
        let weights = vec![1f32; 10];
        test_given_distribution(items, weights);
    }

    #[test]
    fn test_complex_distribution() {
        let items = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
        let weights = vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0];
        test_given_distribution(items, weights);
    }

    #[test]
    fn test_long_complex_distribution() {
        let len = 1000;
        let mut items = Vec::new();
        let mut weights = Vec::new();

        for i in 0..len {
            items.push(i);
            weights.push(1.0 + (i % 2) as f32);
        }
        test_given_distribution(items, weights);
    }
}
