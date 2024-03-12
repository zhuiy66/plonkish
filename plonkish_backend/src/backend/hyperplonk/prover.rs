use crate::{
    backend::hyperplonk::verifier::{pcs_query, point_offset, points},
    pcs::Evaluation,
    piop::sum_check::{
        classic::{ClassicSumCheck, EvaluationsProver},
        SumCheck, VirtualPolynomial,
    },
    poly::multilinear::MultilinearPolynomial,
    util::{
        arithmetic::{div_ceil, steps_by, sum, BatchInvert, PrimeField},
        chain, end_timer,
        expression::{
            rotate::{BinaryField, Rotatable},
            CommonPolynomial, Expression, Rotation,
        },
        parallel::{num_threads, par_map_collect, parallelize, parallelize_iter},
        start_timer,
        transcript::FieldTranscriptWrite,
        Itertools,
    },
    Error,
};
use std::{
    borrow::Borrow,
    collections::{HashMap, HashSet},
    hash::Hash,
};

pub(crate) fn instance_polys<'a, F: PrimeField, R: Rotatable + From<usize>>(
    num_vars: usize,
    instances: impl IntoIterator<Item = impl IntoIterator<Item = &'a F>>,
) -> Vec<MultilinearPolynomial<F>> {
    let usable_indices = R::from(num_vars).usable_indices();//注意，BinaryField的usiable_indices()去掉了全0这个点
    instances
        .into_iter()
        .map(|instances| {
            let mut poly = vec![F::ZERO; 1 << num_vars];
            for (b, instance) in usable_indices.iter().zip(instances.into_iter()) {
                poly[*b] = *instance;
            }
            poly
        })
        .map(MultilinearPolynomial::new)
        .collect()
}

pub(crate) fn lookup_compressed_polys<F: PrimeField, R: Rotatable + From<usize>>(
    lookups: &[Vec<(Expression<F>, Expression<F>)>],
    polys: &[impl Borrow<MultilinearPolynomial<F>>],
    challenges: &[F],
    betas: &[F],
) -> Vec<[MultilinearPolynomial<F>; 2]> {
    if lookups.is_empty() {
        return Default::default();
    }

    let polys = polys.iter().map(Borrow::borrow).collect_vec();
    let num_vars = polys[0].num_vars();
    let expression = lookups
        .iter()
        .flat_map(|lookup| lookup.iter().map(|(input, table)| (input + table)))
        .sum::<Expression<_>>(); //为什么全加起来？大概是让下一段代码能够正确地求出lagranges？
    let lagranges = {
        let rotatable = R::from(num_vars);
        expression
            .used_langrange()
            .into_iter()
            .map(|i| (i, rotatable.nth(i)))//大意是使l_i(X)在乘法子群的第i个元素上为1？
            .collect::<HashSet<_>>()
    };
    lookups
        .iter()
        .map(|lookup| lookup_compressed_poly::<_, R>(lookup, &lagranges, &polys, challenges, betas))
        .collect()
}

pub(super) fn lookup_compressed_poly<F: PrimeField, R: Rotatable + From<usize>>(
    lookup: &[(Expression<F>, Expression<F>)],
    lagranges: &HashSet<(i32, usize)>,
    polys: &[&MultilinearPolynomial<F>],
    challenges: &[F],
    betas: &[F],
) -> [MultilinearPolynomial<F>; 2] {
    let num_vars = polys[0].num_vars();
    let rotatable = R::from(num_vars);
    let compress = |expressions: &[&Expression<F>]| {
        betas
            .iter()
            .copied()
            .zip(expressions.iter().map(|expression| {
                let mut compressed = vec![F::ZERO; 1 << num_vars];//
                parallelize(&mut compressed, |(compressed, start)| {
                    for (b, compressed) in (start..).zip(compressed) {
                        *compressed = expression.evaluate(
                            &|constant| constant,
                            &|common_poly| match common_poly {
                                CommonPolynomial::Identity => F::from(b as u64),
                                CommonPolynomial::Lagrange(i) => {
                                    if lagranges.contains(&(i, b)) {
                                        F::ONE
                                    } else {
                                        F::ZERO
                                    }
                                }
                                CommonPolynomial::EqXY(_) => unreachable!(),
                            },
                            &|query| polys[query.poly()][rotatable.rotate(b, query.rotation())],
                            &|challenge| challenges[challenge],
                            &|value| -value,
                            &|lhs, rhs| lhs + &rhs,
                            &|lhs, rhs| lhs * &rhs,
                            &|value, scalar| value * &scalar,
                        );//witness列已经是把值放在乘法子群上的结果，所以这里不需要row_mapping。
                    }
                });
                MultilinearPolynomial::new(compressed) //对一个Expression生成他在所有点处的值，并构造一个多元线性多项式
            }))
            .sum::<MultilinearPolynomial<_>>() //对求出的所有多项式求RLC
    };

    let (inputs, tables) = lookup
        .iter()
        .map(|(input, table)| (input, table))
        .unzip::<_, _, Vec<_>, Vec<_>>(); //分别得到所有的input列和所有的table列

    let timer = start_timer(|| "compressed_input_poly");
    let compressed_input_poly = compress(&inputs); //得到压缩的input多项式
    end_timer(timer);

    let timer = start_timer(|| "compressed_table_poly");
    let compressed_table_poly = compress(&tables); //得到压缩的table多项式
    end_timer(timer);

    [compressed_input_poly, compressed_table_poly]
}

pub(crate) fn lookup_m_polys<F: PrimeField + Hash>(
    compressed_polys: &[[MultilinearPolynomial<F>; 2]],
) -> Result<Vec<MultilinearPolynomial<F>>, Error> {
    compressed_polys.iter().map(lookup_m_poly).try_collect()
}

pub(super) fn lookup_m_poly<F: PrimeField + Hash>(
    compressed_polys: &[MultilinearPolynomial<F>; 2],
) -> Result<MultilinearPolynomial<F>, Error> {
    let [input, table] = compressed_polys;

    let counts = {
        let indice_map = table.iter().zip(0..).collect::<HashMap<_, usize>>(); // 多项式值向下标的映射

        let chunk_size = div_ceil(input.evals().len(), num_threads());
        let num_chunks = div_ceil(input.evals().len(), chunk_size);
        let mut counts = vec![HashMap::new(); num_chunks];//反映了table中的第i个在input中出现了多少次（如果input中出现了）
        let mut valids = vec![true; num_chunks];
        parallelize_iter(
            counts
                .iter_mut()
                .zip(valids.iter_mut())
                .zip((0..).step_by(chunk_size)),
            |((count, valid), start)| {
                for input in input[start..].iter().take(chunk_size) {
                    if let Some(idx) = indice_map.get(input) {
                        count
                            .entry(*idx)
                            .and_modify(|count| *count += 1)
                            .or_insert(1);
                    } else {
                        *valid = false;
                        break;
                    }
                }
            },
        );
        if valids.iter().any(|valid| !valid) {
            return Err(Error::InvalidSnark("Invalid lookup input".to_string()));
        }
        counts
    };

    let mut m = vec![0; 1 << input.num_vars()];
    for (idx, count) in counts.into_iter().flatten() {
        m[idx] += count;
    }
    let m = par_map_collect(m, |count| match count {
        0 => F::ZERO,
        1 => F::ONE,
        count => F::from(count),
    });
    Ok(MultilinearPolynomial::new(m))//m_poly(i)代表table中的第i个数在input中的出现次数
}

pub(crate) fn lookup_h_polys<F: PrimeField + Hash>(
    compressed_polys: &[[MultilinearPolynomial<F>; 2]],
    m_polys: &[MultilinearPolynomial<F>],
    gamma: &F,
) -> Vec<MultilinearPolynomial<F>> {
    compressed_polys
        .iter()
        .zip(m_polys.iter())
        .map(|(compressed_polys, m_poly)| lookup_h_poly(compressed_polys, m_poly, gamma))
        .collect()
}

pub(super) fn lookup_h_poly<F: PrimeField + Hash>(
    compressed_polys: &[MultilinearPolynomial<F>; 2],
    m_poly: &MultilinearPolynomial<F>,
    gamma: &F,
) -> MultilinearPolynomial<F> {
    let [input, table] = compressed_polys;
    let mut h_input = vec![F::ZERO; 1 << input.num_vars()];
    let mut h_table = vec![F::ZERO; 1 << input.num_vars()];

    parallelize(&mut h_input, |(h_input, start)| {
        for (h_input, input) in h_input.iter_mut().zip(input[start..].iter()) {
            *h_input = *gamma + input;//把input转化成\gamma+input
        }
    });
    parallelize(&mut h_table, |(h_table, start)| {
        for (h_table, table) in h_table.iter_mut().zip(table[start..].iter()) {
            *h_table = *gamma + table;//把table转化成\gamma+table
        }
    });

    let chunk_size = div_ceil(2 * h_input.len(), num_threads());
    parallelize_iter(
        chain![
            h_input.chunks_mut(chunk_size),
            h_table.chunks_mut(chunk_size)
        ],
        |h| {
            h.batch_invert();//求逆元
        },//相当于把h_input和h_table中的所有元素求逆元
    );

    parallelize(&mut h_input, |(h_input, start)| {
        for (h_input, (h_table, m)) in h_input
            .iter_mut()
            .zip(h_table[start..].iter().zip(m_poly[start..].iter()))
        {
            *h_input -= *h_table * m; //h_input(i) = (input(i)+\gamma)^-1 - (table(i)+\gamma)^-1 * m(i)
        }
    });

    if cfg!(feature = "sanity-check") {
        assert_eq!(sum::<F>(&h_input), F::ZERO); // \Sigma((input(i)+\gamma)^-1 - (table(i)+\gamma)^-1 * m(i))=0 
    }

    MultilinearPolynomial::new(h_input) // 这里用的是2022年的cq方案
}

pub(crate) fn permutation_z_polys<F: PrimeField, R: Rotatable + From<usize>>(
    num_chunks: usize,
    permutation_polys: &[(usize, MultilinearPolynomial<F>)],
    polys: &[impl Borrow<MultilinearPolynomial<F>>],
    beta: &F,
    gamma: &F,
) -> Vec<MultilinearPolynomial<F>> {
    if permutation_polys.is_empty() {
        return Vec::new();
    }

    let chunk_size = div_ceil(permutation_polys.len(), num_chunks);
    let polys = polys.iter().map(Borrow::borrow).collect_vec();
    let num_vars = polys[0].num_vars();

    let timer = start_timer(|| "products");
    let products = permutation_polys
        .chunks(chunk_size)
        .enumerate()
        .map(|(chunk_idx, permutation_polys)| {
            let mut product = vec![F::ONE; 1 << num_vars];

            for (poly, permutation_poly) in permutation_polys.iter() {
                parallelize(&mut product, |(product, start)| {
                    for ((product, value), permutation) in product
                        .iter_mut()
                        .zip(polys[*poly][start..].iter())
                        .zip(permutation_poly[start..].iter())
                    {
                        *product *= (*beta * permutation) + gamma + value;
                    }
                });
            }

            parallelize(&mut product, |(product, _)| {
                product.batch_invert();
            });

            for ((poly, _), idx) in permutation_polys.iter().zip(chunk_idx * chunk_size..) {
                let id_offset = idx << num_vars;
                parallelize(&mut product, |(product, start)| {
                    for ((product, value), beta_id) in product
                        .iter_mut()
                        .zip(polys[*poly][start..].iter())
                        .zip(steps_by(F::from((id_offset + start) as u64) * beta, *beta))
                    {
                        *product *= beta_id + gamma + value;
                    }
                });
            }

            product
        })
        .collect_vec();
    end_timer(timer);

    let _timer = start_timer(|| "z_polys");
    let mut z = vec![vec![F::ZERO; 1 << num_vars]; num_chunks];

    let usable_indices = R::from(num_vars).usable_indices();
    let first_idx = usable_indices[0];
    z[0][first_idx] = F::ONE;
    for chunk_idx in 1..num_chunks {
        z[chunk_idx][first_idx] = z[chunk_idx - 1][first_idx] * products[chunk_idx - 1][first_idx];
    }
    for (last_idx, idx) in usable_indices.iter().copied().tuple_windows() {
        z[0][idx] = z[num_chunks - 1][last_idx] * products[num_chunks - 1][last_idx];
        for chunk_idx in 1..num_chunks {
            z[chunk_idx][idx] = z[chunk_idx - 1][idx] * products[chunk_idx - 1][idx];
        }
    }//因为z的检查是涉及next()函数的，所以要按照usable_indices的顺序来赋值；X=0^\mu处的Z值都是0，也能满足约束。

    if cfg!(feature = "sanity-check") {
        let last_idx = *usable_indices.last().unwrap();
        assert_eq!(
            z.last().unwrap()[last_idx] * products.last().unwrap()[last_idx],
            F::ONE
        );
    }

    z.into_iter().map(MultilinearPolynomial::new).collect()//生成所有的Z(X)（用的是类似Halo2中的方法） https://zcash.github.io/halo2/design/proving-system/permutation.html#spanning-a-large-number-of-columns
}

#[allow(clippy::type_complexity)]
pub(super) fn prove_zero_check<F: PrimeField>(
    num_instance_poly: usize,
    expression: &Expression<F>,
    polys: &[&MultilinearPolynomial<F>],
    challenges: Vec<F>,
    y: Vec<F>,
    transcript: &mut impl FieldTranscriptWrite<F>,
) -> Result<(Vec<Vec<F>>, Vec<Evaluation<F>>), Error> {
    prove_sum_check(
        num_instance_poly,
        expression,
        F::ZERO,
        polys,
        challenges,
        y,
        transcript,
    )
}

#[allow(clippy::type_complexity)]
pub(crate) fn prove_sum_check<F: PrimeField>(
    num_instance_poly: usize,
    expression: &Expression<F>,
    sum: F,
    polys: &[&MultilinearPolynomial<F>],
    challenges: Vec<F>,
    y: Vec<F>,
    transcript: &mut impl FieldTranscriptWrite<F>,
) -> Result<(Vec<Vec<F>>, Vec<Evaluation<F>>), Error> {
    let num_vars = polys[0].num_vars();
    let ys = [y]; //y:Vec<F>的长度是1<<num_vars
    let virtual_poly = VirtualPolynomial::new(expression, polys.to_vec(), &challenges, &ys);
    let (_, x, evals) = ClassicSumCheck::<EvaluationsProver<_>, BinaryField>::prove(
        &(),
        num_vars,
        virtual_poly,
        sum,
        transcript,
    )?;//x基本可以认为对应于Hyperplonk中的\alpha_1,\alpha_2,...,\alpha_\mu

    let pcs_query = pcs_query(expression, num_instance_poly);
    let point_offset = point_offset(&pcs_query);

    let timer = start_timer(|| format!("evals-{}", pcs_query.len()));
    let evals = pcs_query
        .iter()
        .flat_map(|query| {
            (point_offset[&query.rotation()]..)
                .zip(if query.rotation() == Rotation::cur() {
                    vec![evals[query]]
                } else {
                    polys[query.poly()].evaluate_for_rotation(&x, query.rotation())
                })
                .map(|(point, eval)| Evaluation::new(query.poly(), point, eval))
        })
        .collect_vec();
    end_timer(timer);

    transcript.write_field_elements(evals.iter().map(Evaluation::value))?;

    Ok((points(&pcs_query, &x), evals))
}
