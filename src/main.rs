#[derive(Debug)]
struct LabeledPoint<'a> {
    /// The data category.
    label: &'a str,

    /// The data point's features.
    point: Vec<f64>,
}

trait LinearAlgebra<T>
where
    T: std::ops::Add + std::ops::Sub,
{
    /// For two vectors `v` and `w` of the same length,
    /// the dot product of `v` and `w` is the result of coupling
    /// up each corresponding element in `v` and `w`, multiplying those two
    /// elements together, and adding each result.
    fn dot(&self, w: &[T]) -> T;

    fn subtract(&self, w: &[T]) -> Vec<T>;

    fn sum_of_squares(&self) -> T;

    fn distance(&self, w: &[T]) -> f64;
}

impl LinearAlgebra<f64> for Vec<f64> {
    fn dot(&self, w: &[f64]) -> f64 {
        assert_eq!(self.len(), w.len());

        self.iter()
            .zip(w.iter())
            .map(|(self_element, w_element)| *self_element * *w_element)
            .sum()
    }

    fn subtract(&self, w: &[f64]) -> Vec<f64> {
        todo!()
    }

    fn sum_of_squares(&self) -> f64 {
        todo!()
    }

    fn distance(&self, w: &[f64]) -> f64 {
        todo!()
    }
}

#[cfg(test)]
#[test]
fn test_linear_algebra_f64_dot() {
    let v = vec![1.0, 5.0, -3.0];
    let w = vec![0.5, 2.0, 3.0];

    assert_eq!(v.dot(&w), 1.5);
}

fn main() {
    println!("Hello, world!");
}
