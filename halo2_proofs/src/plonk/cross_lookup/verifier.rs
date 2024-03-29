
use std::iter;

use super::super::{
    circuit::Expression,ChallengeRforCrossLookup,
};
use super::Argument;
use crate::plonk::{self, ChallengeX};
use crate::{
    arithmetic::CurveAffine,
    plonk::{Error, VerifyingKey},
    poly::{commitment::MSM, Rotation, VerifierQuery},
    transcript::{EncodedChallenge, TranscriptRead},
};
use ff::Field;

pub struct CommittedSet<C: CurveAffine> {
    s_inv_commitment: C,
    product_z_commitment: C,
}

pub struct Committed<C:CurveAffine>{
    sets: Vec<CommittedSet<C>>,
}

pub struct EvaluatedSet<C: CurveAffine> {
    s_inv_commitment: C,
    product_z_commitment: C,
    s_inv_eval: C::Scalar,
    z_eval: C::Scalar,
    z_next_eval: C::Scalar,
}

pub struct Evaluated<C: CurveAffine>{
    sets: Vec<EvaluatedSet<C>>,
}


impl Argument {
    pub(in crate::plonk) fn read_commitments<
        C: CurveAffine,
        E: EncodedChallenge<C>,
        T: TranscriptRead<C, E>,
    >(
        &self,
        transcript: &mut T,
    ) -> Result<Committed<C>, Error> {
        let len = self.columns.len();
        let mut sets = vec![];

        for _ in (0..len){
            let s_inv_commitment = transcript.read_point()?;
            let product_z_commitment = transcript.read_point()?;
            sets.push(CommittedSet {
                s_inv_commitment,
                product_z_commitment,
            });
        }

        Ok(Committed {
            sets,
        })
    }
}

impl<C: CurveAffine> Committed<C> {
    pub(crate) fn evaluate<E: EncodedChallenge<C>, T: TranscriptRead<C, E>>(
        self,
        transcript: &mut T,
    ) -> Result<Evaluated<C>, Error> {
        let mut sets_ans = vec![];

        for set in self.sets.iter(){
            let s_inv_eval = transcript.read_scalar()?;
            let z_eval = transcript.read_scalar()?;
            let z_next_eval = transcript.read_scalar()?;
            sets_ans.push(EvaluatedSet{
                s_inv_commitment: set.s_inv_commitment,
                product_z_commitment: set.product_z_commitment,
                s_inv_eval,
                z_eval,
                z_next_eval,
            })
        }
        Ok(Evaluated {
            sets: sets_ans,
        })
    }
}




//对于每个circuit，各个值需要chain起来

impl<C: CurveAffine> Evaluated<C> {
    pub(in crate::plonk) fn expressions<'a, const ZK: bool>(
        &'a self,
        vk: &'a plonk::VerifyingKey<C>,
        p: &'a Argument,
        advice_evals: &'a [C::Scalar],
        fixed_evals: &'a [C::Scalar],
        instance_evals: &'a [C::Scalar],
        l_0: C::Scalar,
        //l_last: C::Scalar,
        //l_blind: C::Scalar // use these when zk property's codes have been implemented.
        r_for_cross_lookup: ChallengeRforCrossLookup<C>,
    ) -> impl Iterator<Item = C::Scalar> + 'a {
        //注意这里写进的值是先按set排序，再按约束排序的
        iter::empty()
        .chain(
            self.sets.iter().zip(p.columns.iter()).enumerate().flat_map(move |(index,(set,column))|{ //possible bug: move
                //l_0(X) * (1-z(X)) = 0
                let expression1 = l_0 * &(C::Scalar::ONE - set.z_eval);
                // z(\omega X) -z(X) * s_inv(X) * (v(X)+r) = 0
                let value_plus_r = advice_evals[vk.cs.get_advice_query_index(*column, Rotation::cur())] + *r_for_cross_lookup; //possible bug
                let expression2 = set.z_next_eval - set.z_eval * set.s_inv_eval *  value_plus_r;

                std::iter::once(expression1).chain(std::iter::once(expression2))
            })
        )
    }

    pub(in crate::plonk) fn queries<'r, M: MSM<C> + 'r>(
        &'r self,
        vk: &'r VerifyingKey<C>,
        x: ChallengeX<C>,
    ) -> impl Iterator<Item = VerifierQuery<'r, C, M>> + Clone {
        let x_next = vk.domain.rotate_omega(*x, Rotation::next());

        iter::empty()
        .chain(self.sets.iter().flat_map(move |set| {
            iter::empty()
                .chain(Some(VerifierQuery::new_commitment(
                    &set.s_inv_commitment,
                    *x,
                    set.s_inv_eval,
                )))
                .chain(Some(VerifierQuery::new_commitment(
                    &set.product_z_commitment,
                    *x,
                    set.z_eval,
                )))
                .chain(Some(VerifierQuery::new_commitment(
                    &set.product_z_commitment,
                    x_next,
                    set.z_next_eval,
                )))
        }))
    }
}