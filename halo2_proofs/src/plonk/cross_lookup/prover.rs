use std::iter;
use std::ops::Mul;

use super::super::{
    circuit::Expression, ChallengeBeta, ChallengeGamma, ChallengeRforCrossLookup, ChallengeTheta,
    ChallengeX, Error, ProvingKey,
};
use crate::arithmetic::{num_threads, parallelize, parallelize_iter, product, CurveAffine,eval_polynomial};
use crate::plonk;
use crate::poly::commitment::{Blind, Params};
use crate::poly::{Coeff, EvaluationDomain, ExtendedLagrangeCoeff, LagrangeCoeff, Polynomial, ProverQuery};
use crate::transcript::{EncodedChallenge, TranscriptWrite};
use crate::poly::Rotation;
use ark_std::{end_timer, start_timer};
use group::{
    ff::{BatchInvert, Field},
    Curve,
};
use num_integer::Integer;
use rand_core::RngCore;

use super::Argument;

pub(crate) struct CommittedSet<C: CurveAffine> {
    pub(crate) cross_lookup_s_inv_poly: Polynomial<C::Scalar, Coeff>,
    pub(crate) cross_lookup_s_inv_coset: Polynomial<C::Scalar, ExtendedLagrangeCoeff>,
    cross_lookup_s_inv_blind: Blind<C::Scalar>,
    pub(crate) cross_lookup_z_product_poly: Polynomial<C::Scalar, Coeff>,
    pub(crate) cross_lookup_z_product_coset: Polynomial<C::Scalar, ExtendedLagrangeCoeff>,
    cross_lookup_z_product_blind: Blind<C::Scalar>,
}

#[derive(Default)]
pub(crate) struct Committed<C: CurveAffine> {
    pub(crate) sets: Vec<CommittedSet<C>>,
}

pub(crate) struct Evaluated<C: CurveAffine> {
    pub(in crate::plonk) cross_lookup_s_inv_poly: Polynomial<C::Scalar, Coeff>,
    cross_lookup_s_inv_blind: Blind<C::Scalar>,
    pub(in crate::plonk) cross_lookup_z_product_poly: Polynomial<C::Scalar, Coeff>,
    cross_lookup_z_product_blind: Blind<C::Scalar>,
}

impl Argument {
    /// 给定涉及跨系统查找证明的列[A_0,A_1,...,A_{n-1}]：
    /// 求出包含乘积逆元的多项式[s_inv_0,s_inv_1,...,s_inv_{n-1}]以及相应的z多项式[z_0,z_1,...,z_{n-1}]，并且把承诺写进transcript
    /// TODO:作RLC，加上timer
    pub(in crate::plonk) fn commit<
        'a,
        'params: 'a,
        C: CurveAffine,
        P: Params<'params, C>,
        E: EncodedChallenge<C>,
        R: RngCore,
        T: TranscriptWrite<C, E>,
        const ZK: bool,
    >(
        &self,
        pk: &ProvingKey<C>,
        params: &P,
        domain: &EvaluationDomain<C::Scalar>,
        theta: ChallengeTheta<C>,
        r_for_cross_lookup: ChallengeRforCrossLookup<C>,
        advice_values: &'a [Polynomial<C::Scalar, LagrangeCoeff>],
        fixed_values: &'a [Polynomial<C::Scalar, LagrangeCoeff>],
        instance_values: &'a [Polynomial<C::Scalar, LagrangeCoeff>],
        challenges: &'a [C::Scalar],
        mut rng: R,
        transcript: &mut T,
    ) -> Result<Committed<C>, Error> {
        assert!(
            !ZK,
            "the codes which support zk property haven't been implemented."
        );

        //得到s和z的表达式
        let chunk_size = Integer::div_ceil(&(params.n() as usize), &num_threads());
        let num_chunks = Integer::div_ceil(&(params.n() as usize), &chunk_size);
        let mut sets: Vec<CommittedSet<C>> = vec![];

        for column in self.columns.iter() {
            //s_inv polys

            let mut products = vec![C::Scalar::ONE; chunk_size];
            parallelize_iter(
                products
                    .iter_mut()
                    .zip(advice_values[column.index()].chunks(chunk_size)),
                |(part_product, values)| {
                    let randomized_values = values.iter().map(|value| *value + *r_for_cross_lookup);
                    *part_product = product::<C::Scalar>(randomized_values);
                },
            );
            let s = product::<C::Scalar>(products);

            let mut s_poly_inv_values = vec![C::Scalar::ONE; params.n() as usize];
            s_poly_inv_values[0] = s.invert().unwrap();

            let s_inv_poly_lagrange = pk.vk.domain.lagrange_from_vec(s_poly_inv_values.clone());
            let blind = Blind(C::Scalar::random(&mut rng));
            let s_inv_commitment_projective = params.commit_lagrange(&s_inv_poly_lagrange, blind);
            let s_inv_blind = blind;
            let s_inv_poly = domain.lagrange_to_coeff(s_inv_poly_lagrange);
            let s_inv_coset = domain.coeff_to_extended(s_inv_poly.clone());
            let s_inv_commitment = s_inv_commitment_projective.to_affine();

            //end_timer!(timer);
            
            
            //z_polys    z_i = z_{i-1}  * (values_{i-1}+r) * s_inv_{i-1}
            let mut step_product = s_poly_inv_values;//TODO:optimize this using parallel
            parallelize(&mut step_product, |step_product, start| {
                for (step_product, value) in step_product
                    .iter_mut()
                    .zip(advice_values[column.index()][start..].iter())
                {
                    *step_product *= (*value+*r_for_cross_lookup);
                }
            });

            let z = iter::once(C::Scalar::ONE)
            .chain(step_product)
            .scan(C::Scalar::ONE, |state, cur| {
                *state *= &cur;
                Some(*state)
            })
            .take(params.n() as usize)
            .collect::<Vec<_>>();

            assert_eq!(z.len(), params.n() as usize);
            let z_lagrange = pk.vk.domain.lagrange_from_vec(z);

            let z_blind = Blind(C::Scalar::random(&mut rng));
            let z_commitment = params.commit_lagrange(&z_lagrange, z_blind).to_affine();
            let z_poly = pk.vk.domain.lagrange_to_coeff(z_lagrange);
            let z_coset = domain.coeff_to_extended(z_poly.clone());

            //write commitments to transcript
            transcript.write_point(s_inv_commitment)?;
            transcript.write_point(z_commitment)?; // 注意，这里s_inv和z的承诺是交替写进transcript的

            // push elements to CommittedSet
            sets.push(CommittedSet{
                cross_lookup_s_inv_poly: s_inv_poly,
                cross_lookup_s_inv_coset: s_inv_coset,
                cross_lookup_s_inv_blind: s_inv_blind,
                cross_lookup_z_product_poly: z_poly,
                cross_lookup_z_product_coset: z_coset,
                cross_lookup_z_product_blind: z_blind,
            })
        }

        Ok(Committed {
            sets
        })
    }
}


impl<C: CurveAffine> Committed<C> {
    pub(in crate::plonk) fn evaluate<
        E: EncodedChallenge<C>,
        T: TranscriptWrite<C, E>,
    >(
        self,
        pk: &plonk::ProvingKey<C>,
        x: ChallengeX<C>,
        transcript: &mut T,
    ) -> Result<Vec<Evaluated<C>>, Error> {
        
        let domain = &pk.vk.domain;

        let mut sets = self.sets.iter();
        let mut ans_sets = vec![];

        while let Some(set) = sets.next() {
            let s_inv_eval = eval_polynomial(&set.cross_lookup_s_inv_poly, *x);
            let z_eval = eval_polynomial(&set.cross_lookup_z_product_poly, *x);
            let z_next_eval = eval_polynomial(&set.cross_lookup_z_product_poly, domain.rotate_omega(*x, Rotation::next()));

            // Hash cross_lookup's evals
            for eval in iter::empty()
                .chain(Some(&s_inv_eval))
                .chain(Some(&z_eval))
                .chain(Some(&z_next_eval))
            {
                transcript.write_scalar(*eval)?;//这里也是先按set的顺序，再按s_inv_eval、z_eval、z_next_eval的顺序写进transcript
            }
            
            ans_sets.push(Evaluated { 
                cross_lookup_s_inv_poly: set.cross_lookup_s_inv_poly.clone(),
                cross_lookup_s_inv_blind: set.cross_lookup_s_inv_blind,
                cross_lookup_z_product_poly: set.cross_lookup_z_product_poly.clone(),
                cross_lookup_z_product_blind: set.cross_lookup_z_product_blind,
            });
        }
        
        Ok(ans_sets)
    }
}


impl<C: CurveAffine> Evaluated<C> {
    pub(in crate::plonk) fn open<'a>(
        &'a self,
        pk: &'a ProvingKey<C>,
        x: ChallengeX<C>,
    ) -> impl Iterator<Item = ProverQuery<'a, C>> + Clone {
        let x_next = pk.vk.domain.rotate_omega(*x, Rotation::next());

        iter::empty()
            // Open cross-lookup's s_inv poly commitment at x
            .chain(Some(ProverQuery{
                point :*x,
                poly: &self.cross_lookup_s_inv_poly,
                blind: self.cross_lookup_s_inv_blind,
            }))
            //Open cross-lookup's z poly commitment at x
            .chain(Some(ProverQuery{
                point:*x,
                poly: &self.cross_lookup_z_product_poly,
                blind: self.cross_lookup_z_product_blind,
            }))
            //Open cross-lookup's z poly commitment at x_next
            .chain(Some(ProverQuery{
                point: x_next,
                poly: &self.cross_lookup_z_product_poly,
                blind: self.cross_lookup_z_product_blind,
            }))
    }
}