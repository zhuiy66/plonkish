use rand_core::RngCore;

use crate::{
    poly::commitment::Params,
    transcript::{EncodedChallenge, TranscriptWrite},
};

use super::{Advice, Column};

pub(crate) mod prover;
pub(crate) mod verifier;

#[derive(Debug, Clone)]
pub struct Argument {
    /// A sequence of columns involved in the argument.
    pub(super) columns: Vec<Column<Advice>>, // to support any columns later
}

impl Argument {
    pub(crate) fn new() -> Self {
        Argument { columns: vec![] }
    }

    pub(crate) fn add_column(&mut self, column: Column<Advice>) {
        if !self.columns.contains(&column) {
            self.columns.push(column);
        }
    }

    /// Returns columns that participate on the cross_lookup argument.
    pub fn get_columns(&self) -> Vec<Column<Advice>> {
        self.columns.clone()
    }
}
