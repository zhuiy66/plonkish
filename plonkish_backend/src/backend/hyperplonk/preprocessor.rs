use crate::{
    backend::{
        hyperplonk::{HyperPlonkProverParam, HyperPlonkVerifierParam},
        PlonkishCircuitInfo,
    },
    pcs::PolynomialCommitmentScheme,
    poly::multilinear::MultilinearPolynomial,
    util::{
        arithmetic::{div_ceil, steps, PrimeField},
        chain,
        expression::{Expression, Query, Rotation},
        Itertools,
    },
    Error,
};
use std::{array, borrow::Cow, mem};

pub(crate) fn batch_size<F: PrimeField>(circuit_info: &PlonkishCircuitInfo<F>) -> usize {
    let num_lookups = circuit_info.lookups.len(); //lookup的数量
    let num_permutation_polys = circuit_info.permutation_polys().len(); //有多少个polynomial涉及permutation
    chain![
        [circuit_info.preprocess_polys.len() + circuit_info.permutation_polys().len()],
        circuit_info.num_witness_polys.clone(),
        [num_lookups],
        [num_lookups + div_ceil(num_permutation_polys, max_degree(circuit_info, None) - 1)],
    ]
    .sum()
}

#[allow(clippy::type_complexity)]
pub(crate) fn preprocess<F: PrimeField, Pcs: PolynomialCommitmentScheme<F>>(
    param: &Pcs::Param,
    circuit_info: &PlonkishCircuitInfo<F>,
    batch_commit: impl Fn(
        &Pcs::ProverParam,
        Vec<MultilinearPolynomial<F>>,
    ) -> Result<(Vec<MultilinearPolynomial<F>>, Vec<Pcs::Commitment>), Error>,
) -> Result<
    (
        HyperPlonkProverParam<F, Pcs>,
        HyperPlonkVerifierParam<F, Pcs>,
    ),
    Error,
> {
    assert!(circuit_info.is_well_formed()); //TODO: add range check for cross_system_polys

    let num_vars = circuit_info.k;
    let poly_size = 1 << num_vars;
    let batch_size = batch_size(circuit_info);
    let (pcs_pp, pcs_vp) = Pcs::trim(param, poly_size, batch_size)?;

    // Compute preprocesses comms
    let preprocess_polys = circuit_info
        .preprocess_polys
        .iter()
        .cloned()
        .map(MultilinearPolynomial::new)
        .collect_vec();
    let (preprocess_polys, preprocess_comms) = batch_commit(&pcs_pp, preprocess_polys)?;

    // Compute permutation polys and comms
    let permutation_polys = permutation_polys(
        num_vars,
        &circuit_info.permutation_polys(),
        &circuit_info.permutations,
    );
    let (permutation_polys, permutation_comms) = batch_commit(&pcs_pp, permutation_polys)?;

    // Compose expression
    let (num_permutation_z_polys, expression) = compose(circuit_info);
    let vp = HyperPlonkVerifierParam {
        pcs: pcs_vp,
        num_instances: circuit_info.num_instances.clone(),
        num_witness_polys: circuit_info.num_witness_polys.clone(),
        num_challenges: circuit_info.num_challenges.clone(),
        num_lookups: circuit_info.lookups.len(),
        num_permutation_z_polys,
        num_vars,
        expression: expression.clone(),
        preprocess_comms: preprocess_comms.clone(),
        permutation_comms: circuit_info
            .permutation_polys()
            .into_iter()
            .zip(permutation_comms.clone())
            .collect(),
        cross_system_polys: circuit_info.cross_system_polys.clone(),
    };
    let pp = HyperPlonkProverParam {
        pcs: pcs_pp,
        num_instances: circuit_info.num_instances.clone(),
        num_witness_polys: circuit_info.num_witness_polys.clone(),
        num_challenges: circuit_info.num_challenges.clone(),
        lookups: circuit_info.lookups.clone(),
        num_permutation_z_polys,
        num_vars,
        expression,
        preprocess_polys,
        preprocess_comms,
        permutation_polys: circuit_info
            .permutation_polys()
            .into_iter()
            .zip(permutation_polys)
            .collect(),
        permutation_comms,
        cross_system_polys: circuit_info.cross_system_polys.clone(),
    };
    Ok((pp, vp))
}

pub(crate) fn compose<F: PrimeField>(
    circuit_info: &PlonkishCircuitInfo<F>,
) -> (usize, Expression<F>) {
    let challenge_offset = circuit_info.num_challenges.iter().sum::<usize>();
    let [beta, gamma, r_for_cross_lookup, alpha] =
        &array::from_fn(|idx| Expression::<F>::Challenge(challenge_offset + idx)); //后续确认一下：beta,gamma和alpha是不是之后会新产生并且加到challenge列表之后？

    let (lookup_constraints, lookup_zero_checks) = lookup_constraints(circuit_info, beta, gamma);

    let max_degree = max_degree(circuit_info, Some(&lookup_constraints));
    let (num_permutation_z_polys, permutation_constraints) = permutation_constraints(
        circuit_info,
        max_degree,
        beta,
        gamma,
        2 * circuit_info.lookups.len(),
    );

    //add cross_lookup constraints
    let cross_system_lookup_constraints = cross_system_lookup_constraints(
        circuit_info,
        r_for_cross_lookup,
        circuit_info.permutation_polys().len()
            + 2 * circuit_info.lookups.len()
            + num_permutation_z_polys,
    );

    let expression = {
        let constraints = chain![
            circuit_info.constraints.iter(),
            lookup_constraints.iter(),
            permutation_constraints.iter(),
            cross_system_lookup_constraints.iter(),
        ]
        .collect_vec();
        let eq = Expression::eq_xy(0);
        let zero_check_on_every_row = Expression::distribute_powers(constraints, alpha) * eq;
        Expression::distribute_powers(
            chain![lookup_zero_checks.iter(), [&zero_check_on_every_row]],
            alpha,
        )
    };

    (num_permutation_z_polys, expression)
}

pub(super) fn max_degree<F: PrimeField>(
    circuit_info: &PlonkishCircuitInfo<F>,
    lookup_constraints: Option<&[Expression<F>]>,
) -> usize {
    let lookup_constraints = lookup_constraints.map(Cow::Borrowed).unwrap_or_else(|| {
        let dummy_challenge = Expression::zero();
        Cow::Owned(self::lookup_constraints(circuit_info, &dummy_challenge, &dummy_challenge).0)
    });
    chain![
        circuit_info.constraints.iter().map(Expression::degree),
        lookup_constraints.iter().map(Expression::degree),
        circuit_info.max_degree,
        [2],
    ]
    .max()
    .unwrap()
}

pub(super) fn lookup_constraints<F: PrimeField>(
    circuit_info: &PlonkishCircuitInfo<F>,
    beta: &Expression<F>,
    gamma: &Expression<F>,
) -> (Vec<Expression<F>>, Vec<Expression<F>>) {
    let m_offset = circuit_info.num_poly() + circuit_info.permutation_polys().len();
    let h_offset = m_offset + circuit_info.lookups.len();
    let constraints = circuit_info
        .lookups
        .iter()
        .zip(m_offset..)
        .zip(h_offset..)
        .flat_map(|((lookup, m), h)| {
            let [m, h] = &[m, h]
                .map(|poly| Query::new(poly, Rotation::cur()))
                .map(Expression::<F>::Polynomial);
            let (inputs, tables) = lookup
                .iter()
                .map(|(input, table)| (input, table))
                .unzip::<_, _, Vec<_>, Vec<_>>();
            let input = &Expression::distribute_powers(inputs, beta);
            let table = &Expression::distribute_powers(tables, beta); //如果有多行的话，作一个RLC压缩成一行
            [h * (input + gamma) * (table + gamma) - (table + gamma) + m * (input + gamma)]
            //m和h相当于这个lookup的一个专属的多项式（具体定义见2022年的cq方案），这些约束
        })
        .collect_vec();
    let sum_check = (h_offset..)
        .take(circuit_info.lookups.len())
        .map(|h| Query::new(h, Rotation::cur()).into())
        .collect_vec(); //把所有的h多项式组成一个向量返回，后续研究一下为什么这样弄 （后面发现所有的h都要恒等于0？那h放在这是干嘛的？）
    (constraints, sum_check)
}

pub(crate) fn permutation_constraints<F: PrimeField>(
    circuit_info: &PlonkishCircuitInfo<F>,
    max_degree: usize,
    beta: &Expression<F>,
    gamma: &Expression<F>,
    num_builtin_witness_polys: usize,
) -> (usize, Vec<Expression<F>>) {
    let permutation_polys = circuit_info.permutation_polys();
    let chunk_size = max_degree - 1;
    let num_chunks = div_ceil(permutation_polys.len(), chunk_size); //因为多项式次数限制，不能一次性跑所有的permutation polynomials，需要分不同的chunk来跑
    let permutation_offset = circuit_info.num_poly();
    let z_offset = permutation_offset + permutation_polys.len() + num_builtin_witness_polys; // z多项式在lookup的m和h之后
    let polys = permutation_polys
        .iter()
        .map(|idx| Expression::Polynomial(Query::new(*idx, Rotation::cur())))
        .collect_vec();
    let ids = (0..polys.len())
        .map(|idx| {
            let offset = F::from((idx << circuit_info.k) as u64);
            Expression::Constant(offset) + Expression::identity()
        })
        .collect_vec(); //permutation_polys的下标（num_vars+log_2(polys.len())）
    let permutations = (permutation_offset..)
        .map(|idx| Expression::Polynomial(Query::new(idx, Rotation::cur())))
        .take(permutation_polys.len())
        .collect_vec();
    let zs = (z_offset..)
        .map(|idx| Expression::Polynomial(Query::new(idx, Rotation::cur())))
        .take(num_chunks)
        .collect_vec();
    let z_0_next = Expression::<F>::Polynomial(Query::new(z_offset, Rotation::next()));
    let l_0 = &Expression::<F>::lagrange(0);
    let one = &Expression::one();
    let constraints = chain![
        zs.first().map(|z_0| l_0 * (z_0 - one)),
        polys
            .chunks(chunk_size)
            .zip(ids.chunks(chunk_size))
            .zip(permutations.chunks(chunk_size))
            .zip(zs.iter())
            .zip(zs.iter().skip(1).chain([&z_0_next]))
            .map(|((((polys, ids), permutations), z_lhs), z_rhs)| {
                z_lhs
                    * polys
                        .iter()
                        .zip(ids)
                        .map(|(poly, id)| poly + beta * id + gamma)
                        .product::<Expression<_>>()
                    - z_rhs
                        * polys
                            .iter()
                            .zip(permutations)
                            .map(|(poly, permutation)| poly + beta * permutation + gamma)
                            .product::<Expression<_>>()
            }), //类似于halo2的以下方法：https://zcash.github.io/halo2/design/proving-system/permutation.html#spanning-a-large-number-of-columns
    ]
    .collect();
    (num_chunks, constraints) //num_chunks就是多项式Z的数量（需要多少个多项式Z）
}

pub(crate) fn permutation_polys<F: PrimeField>(
    num_vars: usize,                //circuit_info.k
    permutation_polys: &[usize],    //哪些多项式涉及permutation
    cycles: &[Vec<(usize, usize)>], //所有的cycle
) -> Vec<MultilinearPolynomial<F>> {
    let poly_index = {
        let mut poly_index = vec![0; permutation_polys.last().map(|poly| 1 + poly).unwrap_or(0)];
        for (idx, poly) in permutation_polys.iter().enumerate() {
            poly_index[*poly] = idx;
        }
        poly_index
    };
    let mut permutations = (0..permutation_polys.len() as u64)
        .map(|idx| {
            steps(F::from(idx << num_vars))
                .take(1 << num_vars)
                .collect_vec()
        })
        .collect_vec(); //假设第一个下标是i，permutations[i]包含了i*2^num_vars ~ i * 2^nums_vars + (2^num_vars-1)
    for cycle in cycles.iter() {
        let (i0, j0) = cycle[0];
        let mut last = permutations[poly_index[i0]][j0];
        for &(i, j) in cycle.iter().cycle().skip(1).take(cycle.len()) {
            mem::swap(&mut permutations[poly_index[i]][j], &mut last);
        }
    }
    permutations
        .into_iter()
        .map(MultilinearPolynomial::new)
        .collect() //最终的输出是一个Vec<MultilinearPolynomial>，nums_vals的值应该是原来的num_vals+log_2(permutation_polys.len())
} //其中permutation_polys.len()反映了涉及置换的多项式的个数，前几位用于表示多项式的编号，后几位表示该多项式的哪一个值（需要把域上的值转化为二进制）

pub(crate) fn cross_system_lookup_constraints<F: PrimeField>(
    circuit_info: &PlonkishCircuitInfo<F>,
    r_for_cross_lookup: &Expression<F>,
    num_builtin_witness_polys: usize,
) -> Vec<Expression<F>> {
    let num_cross_system_polys = circuit_info.cross_system_polys.len();
    let s_offset = circuit_info.num_poly() + num_builtin_witness_polys;
    let z_offset = s_offset + num_cross_system_polys;

    let cross_system_polys = circuit_info
        .cross_system_polys
        .iter()
        .map(|idx| Expression::<F>::Polynomial(Query::new(*idx, Rotation::cur())))
        .collect_vec();
    let s_polys = (s_offset..)
        .map(|idx| Expression::<F>::Polynomial(Query::new(idx, Rotation::cur())))
        .take(num_cross_system_polys)
        .collect_vec();
    let z_polys = (z_offset..)
        .map(|idx| Expression::<F>::Polynomial(Query::new(idx, Rotation::cur())))
        .take(num_cross_system_polys)
        .collect_vec();
    let z_next_polys = (z_offset..)
        .map(|idx| Expression::<F>::Polynomial(Query::new(idx, Rotation::next())))
        .take(num_cross_system_polys)
        .collect_vec();

    let l_0 = &Expression::<F>::lagrange(0);
    let one = &Expression::<F>::one();
    let constraints = chain![
        z_polys.iter().map(|z| l_0 * (z - one)),
        cross_system_polys
            .iter()
            .zip(s_polys.iter())
            .zip(z_polys.iter())
            .zip(z_next_polys.iter())
            .map(|(((cross_system_poly, s_poly), z_poly), z_next_poly)| {
                s_poly * z_next_poly - z_poly * (cross_system_poly + r_for_cross_lookup)
            }),
    ]
    .collect();

    constraints
}

#[cfg(test)]
pub(crate) mod test {
    use crate::{
        backend::hyperplonk::util::{vanilla_plonk_expression, vanilla_plonk_expression_w_cross_lookup, vanilla_plonk_w_lookup_expression},
        util::expression::{Expression, Query, Rotation},
    };
    use halo2_curves::bn256::Fr;
    use std::array;

    #[test]
    fn compose_vanilla_plonk() {
        let num_vars = 3;
        let expression = vanilla_plonk_expression(num_vars);
        assert_eq!(expression, {
            let [pi, q_l, q_r, q_m, q_o, q_c, w_l, w_r, w_o, s_1, s_2, s_3] =
                &array::from_fn(|poly| Query::new(poly, Rotation::cur()))
                    .map(Expression::Polynomial);
            let [z, z_next] = &[
                Query::new(12, Rotation::cur()),
                Query::new(12, Rotation::next()),
            ]
            .map(Expression::Polynomial);
            let [beta, gamma, r_for_cross_lookup,alpha] = &array::from_fn(Expression::<Fr>::Challenge);
            let [id_1, id_2, id_3] = array::from_fn(|idx| {
                Expression::Constant(Fr::from((idx << num_vars) as u64)) + Expression::identity()
            });
            let l_0 = Expression::<Fr>::lagrange(0);
            let one = Expression::one();
            let constraints = {
                vec![
                    q_l * w_l + q_r * w_r + q_m * w_l * w_r + q_o * w_o + q_c + pi,
                    l_0 * (z - one),
                    (z * ((w_l + beta * id_1 + gamma)
                        * (w_r + beta * id_2 + gamma)
                        * (w_o + beta * id_3 + gamma)))
                        - (z_next
                            * ((w_l + beta * s_1 + gamma)
                                * (w_r + beta * s_2 + gamma)
                                * (w_o + beta * s_3 + gamma))),
                ]
            };
            let eq = Expression::eq_xy(0);
            Expression::distribute_powers(&constraints, alpha) * eq
        });
    }

    #[test]
    fn compose_vanilla_plonk_with_cross_lookup() {
        let num_vars = 3;
        let expression =  vanilla_plonk_expression_w_cross_lookup(num_vars);
        assert_eq!(expression, {
            let [pi, q_l, q_r, q_m, q_o, q_c, w_l, w_r, w_o, s] =
                &array::from_fn(|poly| Query::new(poly, Rotation::cur()))
                    .map(Expression::Polynomial);
            let [z, z_next] = &[
                Query::new(10, Rotation::cur()),
                Query::new(10, Rotation::next()),
            ]
            .map(Expression::Polynomial);
            let [beta, gamma, r_for_cross_lookup,alpha] = &array::from_fn(Expression::<Fr>::Challenge);

            let l_0 = Expression::<Fr>::lagrange(0);
            let one = Expression::one();
            let constraints = {
                vec![
                    q_l * w_l + q_r * w_r + q_m * w_l * w_r + q_o * w_o + q_c + pi,
                    l_0 * (z - one),
                    s * z_next - z*( w_r + r_for_cross_lookup) // 假设涉及cross_lookup的是第7个多项式（也就是w_o）
                ]
            };
            let eq = Expression::eq_xy(0);
            Expression::distribute_powers(&constraints, alpha) * eq
        });
    }

    #[test]
    fn compose_vanilla_plonk_w_lookup() {
        let num_vars = 3;
        let expression = vanilla_plonk_w_lookup_expression(num_vars);
        assert_eq!(expression, {
            let [pi, q_l, q_r, q_m, q_o, q_c, q_lookup, t_l, t_r, t_o, w_l, w_r, w_o, s_1, s_2, s_3] =
                &array::from_fn(|poly| Query::new(poly, Rotation::cur()))
                    .map(Expression::Polynomial);
            let [lookup_m, lookup_h] = &[
                Query::new(16, Rotation::cur()),
                Query::new(17, Rotation::cur()),
            ]
            .map(Expression::<Fr>::Polynomial);
            let [perm_z, perm_z_next] = &[
                Query::new(18, Rotation::cur()),
                Query::new(18, Rotation::next()),
            ]
            .map(Expression::Polynomial);
            let [beta, gamma, r_for_cross_lookup,alpha] = &array::from_fn(Expression::<Fr>::Challenge);
            let [id_1, id_2, id_3] = array::from_fn(|idx| {
                Expression::Constant(Fr::from((idx << num_vars) as u64)) + Expression::identity()
            });
            let l_0 = &Expression::<Fr>::lagrange(0);
            let one = &Expression::one();
            let lookup_input =
                &Expression::distribute_powers(&[w_l, w_r, w_o].map(|w| q_lookup * w), beta);
            let lookup_table = &Expression::distribute_powers([t_l, t_r, t_o], beta);
            let constraints = {
                vec![
                    q_l * w_l + q_r * w_r + q_m * w_l * w_r + q_o * w_o + q_c + pi,
                    lookup_h * (lookup_input + gamma) * (lookup_table + gamma)
                        - (lookup_table + gamma)
                        + lookup_m * (lookup_input + gamma),
                    l_0 * (perm_z - one),
                    (perm_z
                        * ((w_l + beta * id_1 + gamma)
                            * (w_r + beta * id_2 + gamma)
                            * (w_o + beta * id_3 + gamma)))
                        - (perm_z_next
                            * ((w_l + beta * s_1 + gamma)
                                * (w_r + beta * s_2 + gamma)
                                * (w_o + beta * s_3 + gamma))),
                ]
            };
            let eq = Expression::eq_xy(0);
            let zero_check_on_every_row = Expression::distribute_powers(&constraints, alpha) * eq;
            let lookup_zero_check = lookup_h;
            Expression::distribute_powers([lookup_zero_check, &zero_check_on_every_row], alpha)
        });
    }
}
