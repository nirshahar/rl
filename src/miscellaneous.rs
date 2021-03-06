pub trait ArgOrd<T: PartialOrd> {
    fn arg_max(&self) -> usize;
    fn arg_min(&self) -> usize;

    fn min_val(&self) -> T;
    fn max_val(&self) -> T;
}

impl<T: PartialOrd + Copy> ArgOrd<T> for Vec<T> {
    fn arg_max(&self) -> usize {
        self.iter()
            .enumerate()
            .max_by(|(_, v1), (_, v2)| v1.partial_cmp(v2).unwrap())
            .map(|(idx, _)| idx)
            .unwrap()
    }

    fn arg_min(&self) -> usize {
        self.iter()
            .enumerate()
            .min_by(|(_, v1), (_, v2)| v2.partial_cmp(v1).unwrap())
            .map(|(idx, _)| idx)
            .unwrap()
    }

    fn min_val(&self) -> T {
        *self
            .iter()
            .min_by(|v1, v2| v2.partial_cmp(v1).unwrap())
            .unwrap()
    }

    fn max_val(&self) -> T {
        *self
            .iter()
            .max_by(|v1, v2| v1.partial_cmp(v2).unwrap())
            .unwrap()
    }
}
