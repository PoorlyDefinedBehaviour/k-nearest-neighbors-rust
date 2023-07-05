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

    /// For two vectors of the same length `v` and `w`,
    /// subtract the element at `w[i]` from the element at `v[i]`.
    fn subtract(&self, w: &[T]) -> Vec<T>;

    /// A vector's sum of squares is the result of squaring each of its elements and adding everything up.
    fn sum_of_squares(&self) -> T;

    /// The distance between two vectors `v` and `w` is defined as:
    /// sqrt( (v1 - w1)^2 + ... + (vn - wn)^2 )
    fn distance(&self, w: &[T]) -> f64;
}

impl LinearAlgebra<f64> for Vec<f64> {
    fn dot(&self, w: &[f64]) -> f64 {
        assert_eq!(self.len(), w.len());

        self.iter()
            .zip(w.iter())
            .map(|(self_element, w_element)| self_element * w_element)
            .sum()
    }

    fn subtract(&self, w: &[f64]) -> Vec<f64> {
        assert_eq!(self.len(), w.len());

        self.iter()
            .zip(w.iter())
            .map(|(self_element, w_element)| self_element - w_element)
            .collect()
    }

    fn sum_of_squares(&self) -> f64 {
        self.iter().map(|x| x * x).sum()
    }

    fn distance(&self, w: &[f64]) -> f64 {
        assert_eq!(self.len(), w.len());

        self.iter()
            .zip(w.iter())
            .map(|(self_element, w_element)| (self_element - w_element).powf(2.0))
            .sum::<f64>()
            .sqrt()
    }
}

#[cfg(test)]
#[test]
fn test_linear_algebra_vec_f64_dot() {
    let v = vec![1.0, 5.0, -3.0];
    let w = vec![0.5, 2.0, 3.0];

    assert_eq!(v.dot(&w), 1.5);
}

#[cfg(test)]
#[test]
fn test_linear_algebra_vec_f64_subtract() {
    let v = vec![1.0, 5.0, -3.0];
    let w = vec![0.5, 2.0, 3.0];

    assert_eq!(v.subtract(&w), vec![0.5, 3.0, -6.0]);
}

#[cfg(test)]
#[test]
fn test_linear_algebra_vec_f64_sum_of_squares() {
    let v = vec![1.0, 5.0, -3.0];

    assert_eq!(v.sum_of_squares(), 35.0);
}

#[cfg(test)]
#[test]
fn test_linear_algebra_vec_f64_sum_of_distance() {
    let v = vec![1.0, 5.0, -3.0];
    let w = vec![0.5, 2.0, 3.0];

    assert_eq!(v.distance(&w), 45.25_f64.sqrt());
}

fn main() {
    println!("Hello, world!");
}
