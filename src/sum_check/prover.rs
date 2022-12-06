use crate::{
    poly::multilinear::MultilinearPolynomial,
    sum_check::VirtualPolynomial,
    util::{
        arithmetic::{BooleanHypercube, PrimeField},
        expression::{CommonPolynomial, Rotation},
        num_threads, parallelize_iter, Itertools,
    },
};
use num_integer::Integer;
use std::{
    borrow::Cow,
    cmp::Ordering,
    collections::{BTreeMap, HashMap},
    ops::Range,
};

#[derive(Debug)]
pub struct ProvingState<'a, F: PrimeField> {
    virtual_poly: &'a VirtualPolynomial<'a, F>,
    lagranges: HashMap<(i32, usize), (bool, F)>,
    eq_xys: Vec<MultilinearPolynomial<F>>,
    identities: Vec<F>,
    polys: Vec<Cow<'a, MultilinearPolynomial<F>>>,
    round: usize,
    next_map: Vec<usize>,
}

impl<'a, F: PrimeField> ProvingState<'a, F> {
    pub fn new(virtual_poly: &'a VirtualPolynomial<F>) -> Self {
        let bh = BooleanHypercube::new(virtual_poly.info.num_vars());
        let idx_map = bh.idx_map();
        let expression = virtual_poly.info.expression();
        let lagranges = expression
            .used_langrange()
            .into_iter()
            .map(|i| {
                let b = idx_map[i.rem_euclid((1 << virtual_poly.info.num_vars()) as i32) as usize];
                ((i, b >> 1), (b.is_even(), F::one()))
            })
            .collect();
        let eq_xys = virtual_poly
            .ys
            .iter()
            .map(|y| MultilinearPolynomial::eq_xy(y))
            .collect_vec();
        let identities = (0..)
            .map(|idx| F::from((idx as u64) << virtual_poly.info.num_vars()))
            .take(
                expression
                    .used_identity()
                    .into_iter()
                    .max()
                    .unwrap_or_default()
                    + 1,
            )
            .collect_vec();
        let polys = {
            let query_idx_to_poly = expression
                .used_query()
                .into_iter()
                .map(|query| (query.index(), query.poly()))
                .collect::<BTreeMap<_, _>>();
            (0..)
                .map(|idx| {
                    query_idx_to_poly
                        .get(&idx)
                        .map(|poly| Cow::Borrowed(virtual_poly.polys[*poly]))
                        .unwrap_or_else(|| Cow::Owned(MultilinearPolynomial::zero()))
                })
                .take(query_idx_to_poly.keys().max().cloned().unwrap_or_default() + 1)
                .collect_vec()
        };
        Self {
            virtual_poly,
            lagranges,
            eq_xys,
            identities,
            polys,
            round: 0,
            next_map: bh.next_map(),
        }
    }

    pub fn sample_evals(&self) -> Vec<F> {
        let size = 1 << (self.virtual_poly.info.num_vars() - self.round - 1);
        let points = self
            .virtual_poly
            .info
            .sample_points()
            .into_iter()
            .map(F::from)
            .collect_vec();

        let evaluate = |range: Range<usize>, point: &F| {
            if self.round == 0 {
                self.evaluate::<true>(range, point)
            } else {
                self.evaluate::<false>(range, point)
            }
        };

        if size < 32 {
            points
                .iter()
                .map(|point| evaluate(0..size, point))
                .collect()
        } else {
            let num_threads = num_threads();
            let chunk_size = Integer::div_ceil(&size, &num_threads);
            points
                .iter()
                .map(|point| {
                    let mut partials = vec![F::zero(); num_threads];
                    parallelize_iter(
                        partials.iter_mut().zip((0..).step_by(chunk_size)),
                        |(partial, start)| {
                            *partial = evaluate(start..start + chunk_size, point);
                        },
                    );
                    partials
                        .into_iter()
                        .reduce(|acc, partial| acc + &partial)
                        .unwrap()
                })
                .collect()
        }
    }

    pub fn next_round(&mut self, challenge: F) {
        self.lagranges = self
            .lagranges
            .drain()
            .into_iter()
            .map(|((i, b), (is_even, value))| {
                let mut output = value * &challenge;
                if is_even {
                    output = value - &output;
                }
                ((i, b >> 1), (b.is_even(), output))
            })
            .collect();
        self.eq_xys
            .iter_mut()
            .for_each(|eq_xy| *eq_xy = eq_xy.fix_variables(&[challenge]));
        self.identities
            .iter_mut()
            .for_each(|constant| *constant += challenge * F::from(1 << self.round));
        if self.round == 0 {
            let query_idx_to_rotation = self
                .virtual_poly
                .info
                .expression()
                .used_query()
                .into_iter()
                .map(|query| (query.index(), query.rotation()))
                .collect::<BTreeMap<_, _>>();
            self.polys = self
                .polys
                .iter()
                .enumerate()
                .map(|(idx, poly)| {
                    match (poly.is_zero(), query_idx_to_rotation.get(&idx).copied()) {
                        (true, _) => Cow::Owned(MultilinearPolynomial::zero()),
                        (false, Some(Rotation(0))) => Cow::Owned(poly.fix_variables(&[challenge])),
                        (false, Some(rotation)) => {
                            let poly = MultilinearPolynomial::new(
                                (0..1 << self.virtual_poly.info.num_vars())
                                    .map(|b| poly[self.rotate(b, rotation)])
                                    .collect_vec(),
                            );
                            Cow::Owned(poly.fix_variables(&[challenge]))
                        }
                        _ => unreachable!(),
                    }
                })
                .collect_vec();
        } else {
            self.polys.iter_mut().for_each(|poly| {
                if !poly.is_zero() {
                    *poly = Cow::Owned(poly.fix_variables(&[challenge]));
                }
            });
        }
        self.next_map =
            BooleanHypercube::new(self.virtual_poly.info.num_vars() - self.round - 1).next_map();
        self.round += 1;
    }

    fn evaluate<const IS_FIRST_ROUND: bool>(&self, range: Range<usize>, point: &F) -> F {
        let partial_identity_eval = F::from((1 << self.round) as u64) * point;

        let mut sum = F::zero();
        for b in range {
            sum += self.virtual_poly.info.expression().evaluate(
                &|scalar| scalar,
                &|poly| match poly {
                    CommonPolynomial::Lagrange(i) => self.lagrange(i, b, point),
                    CommonPolynomial::EqXY(idx) => {
                        let poly = &self.eq_xys[idx];
                        poly[b << 1] + (poly[(b << 1) + 1] - poly[b << 1]) * point
                    }
                    CommonPolynomial::Identity(idx) => {
                        self.identities[idx]
                            + F::from((b << (self.round + 1)) as u64)
                            + &partial_identity_eval
                    }
                },
                &|query| {
                    let (b_0, b_1) = if IS_FIRST_ROUND {
                        (
                            self.rotate(b << 1, query.rotation()),
                            self.rotate((b << 1) + 1, query.rotation()),
                        )
                    } else {
                        (b << 1, (b << 1) + 1)
                    };
                    let poly = &self.polys[query.index()];
                    poly[b_0] + (poly[b_1] - poly[b_0]) * point
                },
                &|idx| self.virtual_poly.challenges[idx],
                &|scalar| -scalar,
                &|lhs, rhs| lhs + &rhs,
                &|lhs, rhs| lhs * &rhs,
                &|value, scalar| scalar * &value,
            );
        }
        sum
    }

    fn lagrange(&self, i: i32, b: usize, point: &F) -> F {
        self.lagranges
            .get(&(i, b))
            .map(|(is_even, value)| {
                let output = *value * point;
                if *is_even {
                    *value - &output
                } else {
                    output
                }
            })
            .unwrap_or_else(F::zero)
    }

    fn rotate(&self, mut b: usize, rotation: Rotation) -> usize {
        match rotation.0.cmp(&0) {
            Ordering::Less => unimplemented!("Negative roation is not supported yet"),
            Ordering::Equal => b,
            Ordering::Greater => {
                for _ in 0..rotation.0 as usize {
                    b = self.next_map[b];
                }
                b
            }
        }
    }
}