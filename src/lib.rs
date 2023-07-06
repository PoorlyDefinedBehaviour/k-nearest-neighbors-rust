use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct LabeledPoint<'a> {
    /// The data category.
    pub label: &'a str,

    /// The data point's features.
    pub point: Vec<f64>,
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

/// Returns the predicted label for `new_point`.
///
/// `k` is the number of neighbors we use to classify our new data points.
/// `data_points` are our labeled data points.
/// `new_point` is the point we want to classify.
pub fn knn_classify(
    k: u8,
    mut data_points: Vec<LabeledPoint>,
    new_point: &[f64],
) -> Option<String> {
    data_points.sort_unstable_by(|data_point_a, data_point_b| {
        let distance_from_a = data_point_a.point.distance(new_point);
        let distance_from_b = data_point_b.point.distance(new_point);

        distance_from_a
            .partial_cmp(&distance_from_b)
            .expect("nan is impossible")
    });

    let k_nearest_labels: Vec<LabeledPoint> =
        data_points.iter().take(k as usize).cloned().collect();

    find_most_common_label(&k_nearest_labels)
}

fn find_most_common_label(labels: &[LabeledPoint]) -> Option<String> {
    let mut count = HashMap::new();

    for labeled_point in labels.iter() {
        let entry = count.entry(&labeled_point.label).or_insert(0);
        *entry += 1;
    }

    // TODO: break ties by decreasing the number of labels.
    count
        .into_iter()
        .max_by_key(|(_label, occurrences)| *occurrences)
        .map(|(label, _occurrences)| label.to_string())
}

#[cfg(test)]
#[tokio::test]
async fn test_knn() -> anyhow::Result<()> {
    use anyhow::Context;
    use rand::seq::SliceRandom;

    fn parse_data_set(data_set: &str) -> anyhow::Result<Vec<LabeledPoint>> {
        data_set
            .split('\n')
            .filter(|data_point| !data_point.is_empty())
            .map(|data_point| -> anyhow::Result<LabeledPoint> {
                let columns = data_point.split(',').collect::<Vec<&str>>();

                let (label, point) = columns
                    .split_last()
                    .context("splitting data point columns")?;

                let point: Vec<f64> = point
                    .iter()
                    .map(|feature| feature.parse::<f64>())
                    .collect::<Result<Vec<f64>, std::num::ParseFloatError>>()?;

                Ok(LabeledPoint { label, point })
            })
            .collect::<anyhow::Result<Vec<LabeledPoint>>>()
    }

    fn training_set_and_testing_set<T>(mut data: Vec<T>, prob: f64) -> (Vec<T>, Vec<T>)
    where
        T: Clone,
    {
        data.shuffle(&mut rand::thread_rng());

        let split_index = ((data.len() as f64) * prob).round() as usize;

        let (left, right) = data.split_at(split_index);

        (left.to_vec(), right.to_vec())
    }

    fn count_correct_classificatrions(
        training_set: Vec<LabeledPoint>,
        testing_set: &[LabeledPoint],
        k: u8,
    ) -> u32 {
        let mut num_correct = 0;

        for iris in testing_set.iter() {
            let predicted = knn_classify(k, training_set.clone(), &iris.point);

            let actual = iris.label;

            if let Some(predicted) = predicted {
                if predicted == actual {
                    num_correct += 1;
                }
            }
        }

        num_correct
    }

    let body =
        reqwest::get("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data")
            .await?
            .text()
            .await?;

    let points = parse_data_set(&body)?;

    let (training_set, testing_set) = training_set_and_testing_set(points, 0.70);

    let k = 5;
    let num_correct = count_correct_classificatrions(training_set, &testing_set, k);
    let percent_correct = num_correct as f32 / testing_set.len() as f32;

    assert!(percent_correct > 0.9);

    Ok(())
}
