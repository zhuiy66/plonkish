use ff::{Field, FromUniformBytes, WithSmallOrderMulGroup};
use group::Curve;
use rand_core::RngCore;
use std::iter;

use super::{
    vanishing, ChallengeBeta, ChallengeGamma, ChallengeTheta,ChallengeRforCrossLookup, ChallengeX, ChallengeY, Error,
    VerifyingKey,
};
use crate::arithmetic::{compute_inner_product, CurveAffine};
use crate::poly::commitment::{CommitmentScheme, Verifier};
use crate::poly::VerificationStrategy;
use crate::poly::{
    commitment::{Blind, Params, MSM},
    Guard, VerifierQuery,
};
use crate::transcript::{read_n_points, read_n_scalars, EncodedChallenge, TranscriptRead};

#[cfg(feature = "batch")]
mod batch;
#[cfg(feature = "batch")]
pub use batch::BatchVerifier;

use crate::poly::commitment::ParamsVerifier;

/// Returns a boolean indicating whether or not the proof is valid
pub fn verify_proof<
    'params,
    Scheme: CommitmentScheme,
    V: Verifier<'params, Scheme>,
    E: EncodedChallenge<Scheme::Curve>,
    T: TranscriptRead<Scheme::Curve, E>,
    Strategy: VerificationStrategy<'params, Scheme, V>,
    const ZK: bool,
>(
    params: &'params Scheme::ParamsVerifier,
    vk: &VerifyingKey<Scheme::Curve>,
    strategy: Strategy,
    instances: &[&[&[Scheme::Scalar]]],
    transcript: &mut T,
) -> Result<Strategy::Output, Error>
where
    Scheme::Scalar: WithSmallOrderMulGroup<3> + FromUniformBytes<64>,
{
    assert!(!ZK,"codes for zk property haven't been implemented");
    // Check that instances matches the expected number of instance columns
    for instances in instances.iter() {
        if instances.len() != vk.cs.num_instance_columns {
            return Err(Error::InvalidInstances);
        }
    }

    let instance_commitments = if V::QUERY_INSTANCE {
        instances
            .iter()
            .map(|instance| {
                instance
                    .iter()
                    .map(|instance| {
                        if ZK
                            && instance.len()
                                > params.n() as usize - (vk.cs.blinding_factors::<ZK>() + 1)
                        {
                            return Err(Error::InstanceTooLarge);
                        }
                        let mut poly = instance.to_vec();
                        poly.resize(params.n() as usize, Scheme::Scalar::ZERO);
                        let poly = vk.domain.lagrange_from_vec(poly);

                        Ok(params.commit_lagrange(&poly, Blind::default()).to_affine())
                    })
                    .collect::<Result<Vec<_>, _>>()
            })
            .collect::<Result<Vec<_>, _>>()?
    } else {
        vec![vec![]; instances.len()]
    };

    let num_proofs = instance_commitments.len();

    // Hash verification key into transcript
    vk.hash_into(transcript)?;

    if V::QUERY_INSTANCE {
        for instance_commitments in instance_commitments.iter() {
            // Hash the instance (external) commitments into the transcript
            for commitment in instance_commitments {
                transcript.common_point(*commitment)?
            }
        }
    } else {
        for instance in instances.iter() {
            for instance in instance.iter() {
                for value in instance.iter() {
                    transcript.common_scalar(*value)?;
                }
            }
        }
    }

    // Hash the prover's advice commitments into the transcript and squeeze challenges
    let (advice_commitments, challenges) = {
        let mut advice_commitments =
            vec![vec![Scheme::Curve::default(); vk.cs.num_advice_columns]; num_proofs];
        let mut challenges = vec![Scheme::Scalar::ZERO; vk.cs.num_challenges];

        for current_phase in vk.cs.phases() {
            for advice_commitments in advice_commitments.iter_mut() {
                for (phase, commitment) in vk
                    .cs
                    .advice_column_phase
                    .iter()
                    .zip(advice_commitments.iter_mut())
                {
                    if current_phase == *phase {
                        *commitment = transcript.read_point()?; // 把advice列的commitments读出来
                    }
                }
            }
            for (phase, challenge) in vk.cs.challenge_phase.iter().zip(challenges.iter_mut()) {
                if current_phase == *phase {
                    *challenge = *transcript.squeeze_challenge_scalar::<()>(); //把challenges读出来
                }
            }
        }

        (advice_commitments, challenges)
    };

    // Sample theta challenge for keeping lookup columns linearly independent
    let theta: ChallengeTheta<_> = transcript.squeeze_challenge_scalar();

    let lookups_permuted = (0..num_proofs)
        .map(|_| -> Result<Vec<_>, _> {
            // Hash each lookup permuted commitment
            vk.cs
                .lookups
                .iter()
                .map(|argument| argument.read_permuted_commitments(transcript))
                .collect::<Result<Vec<_>, _>>()
        })
        .collect::<Result<Vec<_>, _>>()?; //所有lookup对应的压缩input和压缩table的commitment

    // Sample beta challenge
    let beta: ChallengeBeta<_> = transcript.squeeze_challenge_scalar();

    // Sample gamma challenge
    let gamma: ChallengeGamma<_> = transcript.squeeze_challenge_scalar();

    let permutations_committed = (0..num_proofs)
        .map(|_| {
            // Hash each permutation product commitment
            vk.cs
                .permutation
                .read_product_commitments::<_, _, _, ZK>(vk, transcript)
        })
        .collect::<Result<Vec<_>, _>>()?; //把permutation的z多项式的承诺读出来

    let lookups_committed = lookups_permuted
        .into_iter()
        .map(|lookups| {
            // Hash each lookup product commitment
            lookups
                .into_iter()
                .map(|lookup| lookup.read_product_commitment(transcript))
                .collect::<Result<Vec<_>, _>>()
        })
        .collect::<Result<Vec<_>, _>>()?;

    //cross_lookups_committed
    let r_for_cross_lookup: ChallengeRforCrossLookup<_> = transcript.squeeze_challenge_scalar();
    //let r_for_cross_lookup: ChallengeRforCrossLookup<_> = ChallengeRforCrossLookup::<Scheme::Curve>::getone();

    let cross_lookups_committed = (0..num_proofs)
    .map(|_|{
        vk.cs.cross_lookup_columns.read_commitments(transcript)
    })
    .collect::<Result<Vec<_>,_>>()?;

    let vanishing = vanishing::Argument::read_commitments_before_y::<_, _, ZK>(transcript)?; //这个指的是vanishing用到的random_poly，如果<ZK>是false，这个是一个默认的承诺值

    // Sample y challenge, which keeps the gates linearly independent.
    let y: ChallengeY<_> = transcript.squeeze_challenge_scalar();

    let vanishing = vanishing.read_commitments_after_y(vk, transcript)?; //把所有h_pieces多项式的承诺读出来（组成一个向量）

    // Sample x challenge, which is used to ensure the circuit is
    // satisfied with high probability.
    let x: ChallengeX<_> = transcript.squeeze_challenge_scalar();
    let instance_evals = if V::QUERY_INSTANCE {
        (0..num_proofs)
            .map(|_| -> Result<Vec<_>, _> {
                read_n_scalars(transcript, vk.cs.instance_queries.len())
            })
            .collect::<Result<Vec<_>, _>>()?
    } else {
        let xn = x.pow(&[params.n() as u64, 0, 0, 0]);
        let (min_rotation, max_rotation) =
            vk.cs
                .instance_queries
                .iter()
                .fold((0, 0), |(min, max), (_, rotation)| {
                    if rotation.0 < min {
                        (rotation.0, max)
                    } else if rotation.0 > max {
                        (min, rotation.0)
                    } else {
                        (min, max)
                    }
                });
        let max_instance_len = instances
            .iter()
            .flat_map(|instance| instance.iter().map(|instance| instance.len()))
            .max_by(Ord::cmp)
            .unwrap_or_default();
        let l_i_s = &vk.domain.l_i_range(
            *x,
            xn,
            -max_rotation..max_instance_len as i32 + min_rotation.abs(),
        );
        instances
            .iter()
            .map(|instances| {
                vk.cs
                    .instance_queries
                    .iter()
                    .map(|(column, rotation)| {
                        let instances = instances[column.index()];
                        let offset = (max_rotation - rotation.0) as usize;
                        compute_inner_product(instances, &l_i_s[offset..offset + instances.len()])
                        //这里inner product是干什么的，还没搞明白，后面再看（因为测试不需要instances）
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>()
    };

    let advice_evals = (0..num_proofs)
        .map(|_| -> Result<Vec<_>, _> { read_n_scalars(transcript, vk.cs.advice_queries.len()) })
        .collect::<Result<Vec<_>, _>>()?;

    let fixed_evals = read_n_scalars(transcript, vk.cs.fixed_queries.len())?;

    let vanishing = vanishing.evaluate_after_x::<_, _, ZK>(transcript)?; //新加了一个叫做random_eval的变量（<ZK>为false的时候，这个变量是F::ZERO）

    let permutations_common = vk.permutation.evaluate(transcript)?; //从transcript里面读出所有permutation多项式在x处的值

    let permutations_evaluated = permutations_committed
        .into_iter()
        .map(|permutation| permutation.evaluate::<_, _, ZK>(transcript)) //从transcript中读出permutation的多项式z在x和x_next的值，和承诺放一起构成向量
        .collect::<Result<Vec<_>, _>>()?;

    let lookups_evaluated = lookups_committed
        .into_iter()
        .map(|lookups| -> Result<Vec<_>, _> {
            lookups
                .into_iter()
                .map(|lookup| lookup.evaluate(transcript))
                .collect::<Result<Vec<_>, _>>()
        })
        .collect::<Result<Vec<_>, _>>()?; // 在承诺集合的基础上，加上lookup的z多项式在x和x_next处的值、A'在x和x_prev处的值、S’在x处的值

    //cross-lookups evaluated
    let cross_lookups_evaluated = cross_lookups_committed
    .into_iter()
    .map(|cross_lookups| cross_lookups.evaluate(transcript))
    .collect::<Result<Vec<_>,_>>()?;

    // This check ensures the circuit is satisfied so long as the polynomial
    // commitments open to the correct values.
    let vanishing = {
        // x^n
        let xn = x.pow(&[params.n() as u64, 0, 0, 0]);

        let blinding_factors = vk.cs.blinding_factors::<ZK>();
        let l_evals = vk
            .domain
            .l_i_range(*x, xn, (-((blinding_factors + 1) as i32))..=0);
        assert_eq!(l_evals.len(), 2 + blinding_factors);
        let l_last = l_evals[0];
        let l_blind: Scheme::Scalar = l_evals[1..(1 + blinding_factors)]
            .iter()
            .fold(Scheme::Scalar::ZERO, |acc, eval| acc + eval);
        let l_0 = l_evals[1 + blinding_factors];

        // Compute the expected value of h(x)
        let expressions = advice_evals
            .iter()
            .zip(instance_evals.iter())
            .zip(permutations_evaluated.iter())
            .zip(lookups_evaluated.iter())
            .zip(cross_lookups_evaluated.iter())
            .flat_map(|((((advice_evals, instance_evals), permutation), lookups),cross_lookups)| {
                let challenges = &challenges;
                let fixed_evals = &fixed_evals;
                std::iter::empty()
                    // Evaluate the circuit using the custom gates provided
                    .chain(vk.cs.gates.iter().flat_map(move |gate| {
                        gate.polynomials().iter().map(move |poly| {
                            poly.evaluate(
                                &|scalar| scalar,
                                &|_| panic!("virtual selectors are removed during optimization"),
                                &|query| fixed_evals[query.index.unwrap()],
                                &|query| advice_evals[query.index.unwrap()],
                                &|query| instance_evals[query.index.unwrap()],
                                &|challenge| challenges[challenge.index()],
                                &|a| -a,
                                &|a, b| a + &b,
                                &|a, b| a * &b,
                                &|a, scalar| a * &scalar,
                            )
                        }) //门约束中等号左边的值
                    }))
                    .chain(permutation.expressions::<ZK>(
                        vk,
                        &vk.cs.permutation,
                        &permutations_common,
                        advice_evals,
                        fixed_evals,
                        instance_evals,
                        l_0,
                        l_last,
                        l_blind,
                        beta,
                        gamma,
                        x, //复制约束带来的新等式中等号左边的值
                    ))
                    .chain(
                        lookups
                            .iter()
                            .zip(vk.cs.lookups.iter())
                            .flat_map(move |(p, argument)| {
                                p.expressions::<ZK>(
                                    l_0,
                                    l_last,
                                    l_blind,
                                    argument,
                                    theta,
                                    beta,
                                    gamma,
                                    advice_evals,
                                    fixed_evals,
                                    instance_evals,
                                    challenges, //lookup约束带来的新等式中等号左边的值 (中间包含RLC运算)
                                )
                            })
                            .into_iter(),
                    )
                    .chain(cross_lookups.expressions::<ZK>(
                        vk,
                        &vk.cs.cross_lookup_columns,
                        advice_evals,
                        fixed_evals,
                        instance_evals,
                        l_0,
                        r_for_cross_lookup,
                    ))
            });

        vanishing.verify(params, expressions, y, xn)
    }; //得到从左侧求出的h多项式的值，以及聚合后的h多项式的承诺

    let queries = instance_commitments
        .iter()
        .zip(instance_evals.iter())
        .zip(advice_commitments.iter())
        .zip(advice_evals.iter())
        .zip(permutations_evaluated.iter())
        .zip(lookups_evaluated.iter())
        .zip(cross_lookups_evaluated.iter())
        .flat_map(
            |((
                (
                    (((instance_commitments, instance_evals), advice_commitments), advice_evals),
                    permutation,
                ),
                lookups,
            ),cross_lookups)| {
                iter::empty()
                    .chain(
                        V::QUERY_INSTANCE
                            .then_some(vk.cs.instance_queries.iter().enumerate().map(
                                move |(query_index, &(column, at))| {
                                    VerifierQuery::new_commitment(
                                        &instance_commitments[column.index()],
                                        vk.domain.rotate_omega(*x, at),
                                        instance_evals[query_index],
                                    )
                                },
                            ))
                            .into_iter()
                            .flatten(),
                    )
                    .chain(vk.cs.advice_queries.iter().enumerate().map(
                        move |(query_index, &(column, at))| {
                            VerifierQuery::new_commitment(
                                &advice_commitments[column.index()],
                                vk.domain.rotate_omega(*x, at),
                                advice_evals[query_index],
                            )
                        },
                    ))
                    .chain(permutation.queries::<_, ZK>(vk, x))
                    .chain(
                        lookups
                            .iter()
                            .flat_map(move |p| p.queries(vk, x))
                            .into_iter(),
                    )
                    .chain(cross_lookups.queries(vk,x))
            },
        )
        .chain(
            vk.cs
                .fixed_queries
                .iter()
                .enumerate()
                .map(|(query_index, &(column, at))| {
                    VerifierQuery::new_commitment(
                        &vk.fixed_commitments[column.index()],
                        vk.domain.rotate_omega(*x, at),
                        fixed_evals[query_index],
                    )
                }),
        )
        .chain(permutations_common.queries(&vk.permutation, x))
        .chain(vanishing.queries::<ZK>(x)); //这里面用到了expected_h_eval，所以在后面能验证值到底对不对

    // We are now convinced the circuit is satisfied so long as the
    // polynomial commitments open to the correct values.

    let verifier = V::new(params);
    strategy.process(|msm| {
        verifier
            .verify_proof(transcript, queries, msm)
            .map_err(|_| Error::Opening)
    })
}
