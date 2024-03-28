use crate::{
    backend::{
        hyperplonk::{
            preprocessor::{batch_size, preprocess},
            prover::{
                instance_polys, lookup_compressed_polys, lookup_h_polys, lookup_m_polys,
                permutation_z_polys, prove_zero_check,
            },
            verifier::verify_zero_check,
        },
        PlonkishBackend, PlonkishCircuit, PlonkishCircuitInfo, WitnessEncoding,
    },
    pcs::PolynomialCommitmentScheme,
    poly::multilinear::MultilinearPolynomial,
    util::{
        arithmetic::{powers, PrimeField},
        chain, end_timer,
        expression::{
            rotate::{BinaryField, Rotatable},
            Expression,
        },
        start_timer,
        transcript::{TranscriptRead, TranscriptWrite},
        Deserialize, DeserializeOwned, Itertools, Serialize,
    },
    Error,
};
use rand::RngCore;
use std::{fmt::Debug, hash::Hash, iter, marker::PhantomData};

use self::prover::{cross_lookup_s_polys, cross_lookup_z_polys};

pub(crate) mod preprocessor;
pub(crate) mod prover;
pub(crate) mod verifier;

#[cfg(any(test, feature = "benchmark"))]
pub mod util;

#[derive(Clone, Debug)]
pub struct HyperPlonk<Pcs>(PhantomData<Pcs>);

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HyperPlonkProverParam<F, Pcs>
where
    F: PrimeField,
    Pcs: PolynomialCommitmentScheme<F>,
{
    pub(crate) pcs: Pcs::ProverParam,
    pub(crate) num_instances: Vec<usize>,
    pub(crate) num_witness_polys: Vec<usize>,
    pub(crate) num_challenges: Vec<usize>,
    pub(crate) lookups: Vec<Vec<(Expression<F>, Expression<F>)>>,
    pub(crate) num_permutation_z_polys: usize,
    pub(crate) num_vars: usize,
    pub(crate) expression: Expression<F>,
    pub(crate) preprocess_polys: Vec<MultilinearPolynomial<F>>,
    pub(crate) preprocess_comms: Vec<Pcs::Commitment>,
    pub(crate) permutation_polys: Vec<(usize, MultilinearPolynomial<F>)>,
    pub(crate) permutation_comms: Vec<Pcs::Commitment>,
    pub(crate) cross_system_polys: Vec<usize>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HyperPlonkVerifierParam<F, Pcs>
where
    F: PrimeField,
    Pcs: PolynomialCommitmentScheme<F>,
{
    pub(crate) pcs: Pcs::VerifierParam,
    pub(crate) num_instances: Vec<usize>,
    pub(crate) num_witness_polys: Vec<usize>,
    pub(crate) num_challenges: Vec<usize>,
    pub(crate) num_lookups: usize, //看一下是否随round更新
    pub(crate) num_permutation_z_polys: usize,
    pub(crate) num_vars: usize,
    pub(crate) expression: Expression<F>,
    pub(crate) preprocess_comms: Vec<Pcs::Commitment>,
    pub(crate) permutation_comms: Vec<(usize, Pcs::Commitment)>,
    pub(crate) cross_system_polys: Vec<usize>,
}

impl<F, Pcs> PlonkishBackend<F> for HyperPlonk<Pcs>
where
    F: PrimeField + Hash + Serialize + DeserializeOwned, // delete
    Pcs: PolynomialCommitmentScheme<F, Polynomial = MultilinearPolynomial<F>>,
{
    type Pcs = Pcs;
    type ProverParam = HyperPlonkProverParam<F, Pcs>;
    type VerifierParam = HyperPlonkVerifierParam<F, Pcs>;

    fn setup(
        circuit_info: &PlonkishCircuitInfo<F>,
        rng: impl RngCore,
    ) -> Result<Pcs::Param, Error> {
        assert!(circuit_info.is_well_formed());

        let num_vars = circuit_info.k;
        let poly_size = 1 << num_vars;
        let batch_size = batch_size(circuit_info);
        Pcs::setup(poly_size, batch_size, rng)
    }

    fn preprocess(
        param: &Pcs::Param,
        circuit_info: &PlonkishCircuitInfo<F>,
    ) -> Result<(Self::ProverParam, Self::VerifierParam), Error> {
        preprocess(param, circuit_info, |pp, polys| {
            let comms = Pcs::batch_commit(pp, &polys)?;
            Ok((polys, comms))
        })
    }

    fn prove(
        pp: &Self::ProverParam,
        circuit: &impl PlonkishCircuit<F>,
        transcript: &mut impl TranscriptWrite<Pcs::CommitmentChunk, F>,
        _: impl RngCore,
    ) -> Result<(), Error> {
        let instance_polys = {
            let instances = circuit.instances();
            for (num_instances, instances) in pp.num_instances.iter().zip_eq(instances) {
                assert_eq!(instances.len(), *num_instances);
                for instance in instances.iter() {
                    transcript.common_field_element(instance)?;
                }
            }
            instance_polys::<_, BinaryField>(pp.num_vars, instances)
        };

        // Round 0..n

        let mut witness_polys = Vec::with_capacity(pp.num_witness_polys.iter().sum()); //把所有轮用到的witness多项式的数量加起来，构造一个相应长度的Vec<MultilinearPolynomial>向量
        let mut witness_comms = Vec::with_capacity(witness_polys.len());
        //let mut challenges = Vec::with_capacity(pp.num_challenges.iter().sum::<usize>() + 4); //为什么加4？这是因为\alpha，\beta,\gamma是额外的。
        //发现后面的prove_zero_check()中用到了challenges向量，看来challenges的前一些值还是有用的
        //但是如果Circuit使用VanillaPlonk，则一开始num_challenges长度为1，唯一的元素值为0（也就是说只有一个phase，phase=0）
        let mut challenges = Vec::with_capacity(pp.num_challenges.iter().sum::<usize>() + 5);
        for (round, (num_witness_polys, num_challenges)) in pp
            .num_witness_polys
            .iter()
            .zip_eq(pp.num_challenges.iter())
            .enumerate()
        {
            let timer = start_timer(|| format!("witness_collector-{round}"));
            let polys = circuit
                .synthesize(round, &challenges)? //在backend.rs中，这个函数仅在round=0时生效
                .into_iter()
                .map(MultilinearPolynomial::new) //用Vec<F>表示的witness列应该是已经经过row_mapping之后的了，因为plonkish_backend/src/frontend/halo2.rs中实现的assign_advice()、assign_fixed()、copy()等函数都对参数row做了一次row_mapping。
                .collect_vec();
            assert_eq!(polys.len(), *num_witness_polys);
            end_timer(timer);

            witness_comms.extend(Pcs::batch_commit_and_write(&pp.pcs, &polys, transcript)?);
            witness_polys.extend(polys);
            challenges.extend(transcript.squeeze_challenges(*num_challenges));
        }
        let polys = chain![&instance_polys, &pp.preprocess_polys, &witness_polys].collect_vec();

        // Round n

        let beta = transcript.squeeze_challenge();

        let timer = start_timer(|| format!("lookup_compressed_polys-{}", pp.lookups.len()));
        let lookup_compressed_polys = {
            let max_lookup_width = pp.lookups.iter().map(Vec::len).max().unwrap_or_default();
            let betas = powers(beta).take(max_lookup_width).collect_vec();
            lookup_compressed_polys::<_, BinaryField>(&pp.lookups, &polys, &challenges, &betas)
        };
        end_timer(timer);

        let timer = start_timer(|| format!("lookup_m_polys-{}", pp.lookups.len()));
        let lookup_m_polys = lookup_m_polys(&lookup_compressed_polys)?; //注意，这个函数也求了X=0^\mu处的情况，但这个不影响lookup的正确性，因为input和table在X=0^\mu处都为0，这是由instance、preprocess和witness等多项式的生成过程决定的。
        end_timer(timer);

        let lookup_m_comms = Pcs::batch_commit_and_write(&pp.pcs, &lookup_m_polys, transcript)?;

        // Round n+1

        let gamma = transcript.squeeze_challenge();

        let timer = start_timer(|| format!("lookup_h_polys-{}", pp.lookups.len()));
        let lookup_h_polys = lookup_h_polys(&lookup_compressed_polys, &lookup_m_polys, &gamma);
        end_timer(timer);

        let timer = start_timer(|| format!("permutation_z_polys-{}", pp.permutation_polys.len()));
        let permutation_z_polys = permutation_z_polys::<_, BinaryField>(
            pp.num_permutation_z_polys,
            &pp.permutation_polys,
            &polys,
            &beta,
            &gamma,
        );
        end_timer(timer);

        let lookup_h_permutation_z_polys =
            chain![lookup_h_polys.iter(), permutation_z_polys.iter()].collect_vec();
        let lookup_h_permutation_z_comms =
            Pcs::batch_commit_and_write(&pp.pcs, lookup_h_permutation_z_polys.clone(), transcript)?;

        //Round n+2
        let r_for_cross_lookup = transcript.squeeze_challenge();

        let timer = start_timer(|| {
            format!(
                "cross_system_lookup_s_polys_z_polys-{}",
                pp.cross_system_polys.len()
            )
        });

        let cross_lookup_s_polys = cross_lookup_s_polys::<_, BinaryField>(
            &pp.cross_system_polys,
            &polys,
            &r_for_cross_lookup,
        );
        let cross_lookup_z_polys = cross_lookup_z_polys::<_, BinaryField>(
            &pp.cross_system_polys,
            &polys,
            &cross_lookup_s_polys,
            &r_for_cross_lookup,
        );
        let cross_lookup_s_comms =
            Pcs::batch_commit_and_write(&pp.pcs, &cross_lookup_s_polys, transcript)?;
        let cross_lookup_z_comms =
            Pcs::batch_commit_and_write(&pp.pcs, &cross_lookup_z_polys, transcript)?;

        end_timer(timer);

        //println!("{}",r_for_cross_lookup);
        // polys[7].evals().iter().zip(cross_lookup_s_polys.evals().iter()).zip(cross_lookup_z_polys.evals().iter()).map(|((a,b),c)|{
        //     println!("{} {} {}",*a,*b,*c);
        // });

        // Round n+3

        let alpha = transcript.squeeze_challenge();
        let y = transcript.squeeze_challenges(pp.num_vars);

        let polys = chain![
            polys,
            pp.permutation_polys.iter().map(|(_, poly)| poly),
            lookup_m_polys.iter(),
            lookup_h_permutation_z_polys,
            cross_lookup_s_polys.iter(),
            cross_lookup_z_polys.iter(),
        ]
        .collect_vec();
        challenges.extend([beta, gamma, r_for_cross_lookup, alpha]);
        let (points, evals) = prove_zero_check(
            pp.num_instances.len(),
            &pp.expression,
            &polys,
            challenges,
            y,
            transcript,
        )?;

        // PCS open

        let dummy_comm = Pcs::Commitment::default();
        let comms = chain![
            iter::repeat(&dummy_comm).take(pp.num_instances.len()),
            &pp.preprocess_comms,
            &witness_comms,
            &pp.permutation_comms,
            &lookup_m_comms,
            &lookup_h_permutation_z_comms,
            &cross_lookup_s_comms,
            &cross_lookup_z_comms,
        ]
        .collect_vec();
        let timer = start_timer(|| format!("pcs_batch_open-{}", evals.len()));
        Pcs::batch_open(&pp.pcs, polys, comms, &points, &evals, transcript)?;
        end_timer(timer);

        Ok(())
    }

    fn generate_prove_polys<'a>(
        pp: &'a Self::ProverParam,
        circuit: &'a impl PlonkishCircuit<F>,
        transcript: &'a mut impl TranscriptWrite<Pcs::CommitmentChunk, F>,
        _: impl RngCore,
    ) -> Result<(F, Vec<MultilinearPolynomial<F>>), Error> {
        let instance_polys = {
            let instances = circuit.instances();
            for (num_instances, instances) in pp.num_instances.iter().zip_eq(instances) {
                assert_eq!(instances.len(), *num_instances);
                for instance in instances.iter() {
                    transcript.common_field_element(instance)?;
                }
            }
            instance_polys::<_, BinaryField>(pp.num_vars, instances)
        };

        // Round 0..n

        let mut witness_polys = Vec::with_capacity(pp.num_witness_polys.iter().sum()); //把所有轮用到的witness多项式的数量加起来，构造一个相应长度的Vec<MultilinearPolynomial>向量
        let mut witness_comms = Vec::with_capacity(witness_polys.len());
        //let mut challenges = Vec::with_capacity(pp.num_challenges.iter().sum::<usize>() + 4); //为什么加4？这是因为\alpha，\beta,\gamma是额外的。
        //发现后面的prove_zero_check()中用到了challenges向量，看来challenges的前一些值还是有用的
        //但是如果Circuit使用VanillaPlonk，则一开始num_challenges长度为1，唯一的元素值为0（也就是说只有一个phase，phase=0）
        let mut challenges = Vec::with_capacity(pp.num_challenges.iter().sum::<usize>() + 5);
        for (round, (num_witness_polys, num_challenges)) in pp
            .num_witness_polys
            .iter()
            .zip_eq(pp.num_challenges.iter())
            .enumerate()
        {
            let timer = start_timer(|| format!("witness_collector-{round}"));
            let polys = circuit
                .synthesize(round, &challenges)? //在backend.rs中，这个函数仅在round=0时生效
                .into_iter()
                .map(MultilinearPolynomial::new) //用Vec<F>表示的witness列应该是已经经过row_mapping之后的了，因为plonkish_backend/src/frontend/halo2.rs中实现的assign_advice()、assign_fixed()、copy()等函数都对参数row做了一次row_mapping。
                .collect_vec();
            assert_eq!(polys.len(), *num_witness_polys);
            end_timer(timer);

            witness_comms.extend(Pcs::batch_commit_and_write(&pp.pcs, &polys, transcript)?);
            witness_polys.extend(polys);
            challenges.extend(transcript.squeeze_challenges(*num_challenges));
        }
        let polys =
            chain![instance_polys, pp.preprocess_polys.clone(), witness_polys].collect_vec();

        // Round n

        let beta = transcript.squeeze_challenge();

        let timer = start_timer(|| format!("lookup_compressed_polys-{}", pp.lookups.len()));
        let lookup_compressed_polys = {
            let max_lookup_width = pp.lookups.iter().map(Vec::len).max().unwrap_or_default();
            let betas = powers(beta).take(max_lookup_width).collect_vec();
            lookup_compressed_polys::<_, BinaryField>(&pp.lookups, &polys, &challenges, &betas)
        };
        end_timer(timer);

        let timer = start_timer(|| format!("lookup_m_polys-{}", pp.lookups.len()));
        let lookup_m_polys = lookup_m_polys(&lookup_compressed_polys)?; //注意，这个函数也求了X=0^\mu处的情况，但这个不影响lookup的正确性，因为input和table在X=0^\mu处都为0，这是由instance、preprocess和witness等多项式的生成过程决定的。
        end_timer(timer);

        let lookup_m_comms = Pcs::batch_commit_and_write(&pp.pcs, &lookup_m_polys, transcript)?;

        // Round n+1

        let gamma = transcript.squeeze_challenge();

        let timer = start_timer(|| format!("lookup_h_polys-{}", pp.lookups.len()));
        let lookup_h_polys = lookup_h_polys(&lookup_compressed_polys, &lookup_m_polys, &gamma);
        end_timer(timer);

        let timer = start_timer(|| format!("permutation_z_polys-{}", pp.permutation_polys.len()));
        let permutation_z_polys = permutation_z_polys::<_, BinaryField>(
            pp.num_permutation_z_polys,
            &pp.permutation_polys,
            &polys,
            &beta,
            &gamma,
        );
        end_timer(timer);

        let lookup_h_permutation_z_polys =
            chain![lookup_h_polys, permutation_z_polys].collect_vec();
        let lookup_h_permutation_z_comms =
            Pcs::batch_commit_and_write(&pp.pcs, &lookup_h_permutation_z_polys, transcript)?;

        //Round n+2
        let mut r_for_cross_lookup = transcript.squeeze_challenge();
        r_for_cross_lookup = F::ONE;

        let timer = start_timer(|| {
            format!(
                "cross_system_lookup_s_polys_z_polys-{}",
                pp.cross_system_polys.len()
            )
        });

        let cross_lookup_s_polys = cross_lookup_s_polys::<_, BinaryField>(
            &pp.cross_system_polys,
            &polys,
            &r_for_cross_lookup,
        );
        let cross_lookup_z_polys = cross_lookup_z_polys::<_, BinaryField>(
            &pp.cross_system_polys,
            &polys,
            &cross_lookup_s_polys,
            &r_for_cross_lookup,
        );

        end_timer(timer);

        let polys = chain![
            polys,
            pp.permutation_polys.iter().map(|(_, poly)| poly.clone()),
            lookup_m_polys,
            lookup_h_permutation_z_polys,
            cross_lookup_s_polys,
            cross_lookup_z_polys,
        ]
        .collect_vec();
        Ok((r_for_cross_lookup, polys))
    }

    fn verify(
        vp: &Self::VerifierParam,
        instances: &[Vec<F>],
        transcript: &mut impl TranscriptRead<Pcs::CommitmentChunk, F>,
        _: impl RngCore,
    ) -> Result<(), Error> {
        for (num_instances, instances) in vp.num_instances.iter().zip_eq(instances) {
            assert_eq!(instances.len(), *num_instances);
            for instance in instances.iter() {
                transcript.common_field_element(instance)?;
            }
        }

        // Round 0..n

        let mut witness_comms = Vec::with_capacity(vp.num_witness_polys.iter().sum());
        let mut challenges = Vec::with_capacity(vp.num_challenges.iter().sum::<usize>() + 4);
        for (num_polys, num_challenges) in
            vp.num_witness_polys.iter().zip_eq(vp.num_challenges.iter())
        {
            witness_comms.extend(Pcs::read_commitments(&vp.pcs, *num_polys, transcript)?);
            challenges.extend(transcript.squeeze_challenges(*num_challenges));
        }

        // Round n

        let beta = transcript.squeeze_challenge();

        let lookup_m_comms = Pcs::read_commitments(&vp.pcs, vp.num_lookups, transcript)?;

        // Round n+1

        let gamma = transcript.squeeze_challenge();

        let lookup_h_permutation_z_comms = Pcs::read_commitments(
            &vp.pcs,
            vp.num_lookups + vp.num_permutation_z_polys,
            transcript,
        )?;

        // Round n+2

        let r_for_cross_lookup = transcript.squeeze_challenge();
        let cross_lookup_s_comms =
            Pcs::read_commitments(&vp.pcs, vp.cross_system_polys.len(), transcript)?;
        let cross_lookup_z_comms =
            Pcs::read_commitments(&vp.pcs, vp.cross_system_polys.len(), transcript)?;

        // Round n+3

        let alpha = transcript.squeeze_challenge();
        let y = transcript.squeeze_challenges(vp.num_vars);

        challenges.extend([beta, gamma, r_for_cross_lookup, alpha]);
        let (points, evals) = verify_zero_check(
            vp.num_vars,
            &vp.expression,
            instances,
            &challenges,
            &y,
            transcript,
        )?;

        // PCS verify

        let dummy_comm = Pcs::Commitment::default();
        let comms = chain![
            iter::repeat(&dummy_comm).take(vp.num_instances.len()),
            &vp.preprocess_comms,
            &witness_comms,
            vp.permutation_comms.iter().map(|(_, comm)| comm),
            &lookup_m_comms,
            &lookup_h_permutation_z_comms,
            &cross_lookup_s_comms,
            &cross_lookup_z_comms,
        ]
        .collect_vec();
        Pcs::batch_verify(&vp.pcs, comms, &points, &evals, transcript)?;

        Ok(())
    }
}

impl<Pcs> WitnessEncoding for HyperPlonk<Pcs> {
    fn row_mapping(k: usize) -> Vec<usize> {
        BinaryField::new(k).usable_indices()
    } //把电路中的行下标映射到乘法子群上的下标，具体到BinaryField上，是1 mod p(X),x mod p(X),x^2 mod p(X)... 的编码结果
}

#[cfg(test)]
mod test {
    use crate::{
        backend::{
            hyperplonk::{
                util::{rand_vanilla_plonk_circuit, rand_vanilla_plonk_w_lookup_circuit},
                HyperPlonk,
            },
            test::run_plonkish_backend,
        },
        pcs::{
            multilinear::{
                Gemini, MultilinearBrakedown, MultilinearHyrax, MultilinearIpa, MultilinearKzg,
                Zeromorph,
            },
            univariate::UnivariateKzg,
        },
        util::{
            code::BrakedownSpec6, expression::rotate::BinaryField, hash::Keccak256,
            test::seeded_std_rng, transcript::Keccak256Transcript,
        },
    };
    use halo2_curves::{
        bn256::{self, Bn256},
        grumpkin,
    };

    macro_rules! tests {
        ($suffix:ident, $pcs:ty, $num_vars_range:expr) => {
            paste::paste! {
                #[test]
                fn [<vanilla_plonk_w_ $suffix>]() {
                    run_plonkish_backend::<_, HyperPlonk<$pcs>, Keccak256Transcript<_>, _>($num_vars_range, |num_vars| {
                        rand_vanilla_plonk_circuit::<_, BinaryField>(num_vars, seeded_std_rng(), seeded_std_rng())
                    });
                }

                #[test]
                fn [<vanilla_plonk_w_lookup_w_ $suffix>]() {
                    run_plonkish_backend::<_, HyperPlonk<$pcs>, Keccak256Transcript<_>, _>($num_vars_range, |num_vars| {
                        rand_vanilla_plonk_w_lookup_circuit::<_, BinaryField>(num_vars, seeded_std_rng(), seeded_std_rng())
                    });
                }
            }
        };
        ($suffix:ident, $pcs:ty) => {
            tests!($suffix, $pcs, 2..16);
        };
    }

    tests!(brakedown, MultilinearBrakedown<bn256::Fr, Keccak256, BrakedownSpec6>);
    tests!(hyrax, MultilinearHyrax<grumpkin::G1Affine>, 5..16);
    tests!(ipa, MultilinearIpa<grumpkin::G1Affine>);
    tests!(kzg, MultilinearKzg<Bn256>);
    tests!(gemini_kzg, Gemini<UnivariateKzg<Bn256>>);
    tests!(zeromorph_kzg, Zeromorph<UnivariateKzg<Bn256>>);
}
