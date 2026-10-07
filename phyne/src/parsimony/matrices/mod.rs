use std::{fmt, iter::zip};

use crate::alignment::{Mapping, PairwiseAlignment};
use crate::alphabets::ParsimonySet;
use crate::parsimony::{
    Direction::{self, *},
    ParsimonyScoring, ParsimonySite,
    SiteFlag::{self, *},
};

pub(super) struct ScoreMatrices {
    pub(super) m: Vec<Vec<f64>>,
    pub(super) x: Vec<Vec<f64>>,
    pub(super) y: Vec<Vec<f64>>,
}

impl ScoreMatrices {
    pub(super) fn new(len1: usize, len2: usize) -> ScoreMatrices {
        ScoreMatrices {
            m: vec![vec![0.0; len2]; len1],
            x: vec![vec![0.0; len2]; len1],
            y: vec![vec![0.0; len2]; len1],
        }
    }
}

pub(super) struct TracebackMatrices {
    pub(super) m: Vec<Vec<Direction>>,
    pub(super) x: Vec<Vec<Direction>>,
    pub(super) y: Vec<Vec<Direction>>,
}

impl TracebackMatrices {
    pub(super) fn new(len1: usize, len2: usize) -> TracebackMatrices {
        TracebackMatrices {
            m: vec![vec![Matc; len2]; len1],
            x: vec![vec![GapInY; len2]; len1],
            y: vec![vec![GapInX; len2]; len1],
        }
    }
}

pub(crate) struct ParsimonyAlignmentMatrices<'a> {
    rows: usize,
    cols: usize,
    scoring: &'a dyn ParsimonyScoring,
    x_info: &'a [ParsimonySite],
    x_blen: f64,
    y_info: &'a [ParsimonySite],
    y_blen: f64,
    pub(super) score: ScoreMatrices,
    pub(super) trace: TracebackMatrices,
    pub(super) direction_picker: [&'static [Direction]; 8],
    pub(crate) rng: fn(usize) -> usize,
}

impl fmt::Display for ParsimonyAlignmentMatrices<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "S.M")?;
        for row in &self.score.m {
            writeln!(f, "{row:?}")?;
        }
        writeln!(f, "S.X")?;
        for row in &self.score.x {
            writeln!(f, "{row:?}")?;
        }
        writeln!(f, "S.Y")?;
        for row in &self.score.y {
            writeln!(f, "{row:?}")?;
        }
        writeln!(f, "T.M")?;
        for row in &self.trace.m {
            writeln!(f, "{row:?}")?;
        }
        writeln!(f, "T.X")?;
        for row in &self.trace.x {
            writeln!(f, "{row:?}")?;
        }
        writeln!(f, "T.Y")?;
        for row in &self.trace.y {
            writeln!(f, "{row:?}")?;
        }
        Ok(())
    }
}

impl<'a> ParsimonyAlignmentMatrices<'a> {
    pub(crate) fn new(
        x_info: &'a [ParsimonySite],
        x_blen: f64,
        y_info: &'a [ParsimonySite],
        y_blen: f64,
        scoring: &'a impl ParsimonyScoring,
        rng: fn(usize) -> usize,
    ) -> ParsimonyAlignmentMatrices<'a> {
        let rows = x_info.len() + 1;
        let cols = y_info.len() + 1;
        ParsimonyAlignmentMatrices {
            rows,
            cols,
            scoring,
            x_info,
            x_blen,
            y_info,
            y_blen,
            score: ScoreMatrices::new(rows, cols),
            trace: TracebackMatrices::new(rows, cols),
            direction_picker: [
                /* 000 */ &[][..],
                /* 001 */ &[Matc][..],
                /* 010 */ &[GapInY][..],
                /* 011 */ &[Matc, GapInY][..],
                /* 100 */ &[GapInX][..],
                /* 101 */ &[Matc, GapInX][..],
                /* 110 */ &[GapInY, GapInX][..],
                /* 111 */ &[Matc, GapInX, GapInY][..],
            ],
            rng,
        }
    }

    fn min_score(&self, blen: f64, set: &ParsimonySet, ancestor: &u8) -> f64 {
        set.iter()
            .map(|child| self.scoring.r#match(blen, ancestor, child))
            .fold(f64::INFINITY, f64::min)
    }

    fn score_match_one_branch(&self, blen: f64, a_set: &ParsimonySet, c_set: &ParsimonySet) -> f64 {
        a_set
            .iter()
            .map(|ancestor| self.min_score(blen, c_set, ancestor))
            .fold(f64::INFINITY, f64::min)
    }

    fn select_direction(&self, sm: f64, sx: f64, sy: f64) -> (f64, Direction) {
        let (mut min_val, mut sel_mat) = (sm, 0b001);
        if sx < min_val {
            (min_val, sel_mat) = (sx, 0b010);
        } else if sx == min_val {
            sel_mat |= 0b010;
        }
        if sy < min_val {
            (min_val, sel_mat) = (sy, 0b100);
        } else if sy == min_val {
            sel_mat |= 0b100;
        }
        (
            min_val,
            self.direction_picker[sel_mat][(self.rng)(self.direction_picker[sel_mat].len())],
        )
    }

    pub(crate) fn fill_matrices(&mut self) {
        self.init_x();
        self.init_y();
        for i in 1..self.rows {
            for j in 1..self.cols {
                if self.x_info[i - 1].is_fixed() || self.y_info[j - 1].is_fixed() {
                    let ni = i - self.x_info[i - 1].is_fixed() as usize;
                    let nj = j - self.y_info[j - 1].is_fixed() as usize;
                    self.score.m[i][j] = self.score.m[ni][nj];
                    self.score.x[i][j] = self.score.x[ni][nj];
                    self.score.y[i][j] = self.score.y[ni][nj];
                } else {
                    (self.score.m[i][j], self.trace.m[i][j]) = self.fill_s_m(i - 1, j - 1);
                    (self.score.x[i][j], self.trace.x[i][j]) = self.fill_s_x(i - 1, j);
                    (self.score.y[i][j], self.trace.y[i][j]) = self.fill_s_y(i, j - 1);
                }
            }
        }
    }

    fn init_x(&mut self) {
        for i in 1..self.rows {
            self.score.x[i][0] = self.score.x[i - 1][0]
                + if self.x_info[i - 1].no_gap() {
                    self.score_match_one_branch(
                        self.x_blen,
                        &self.x_info[i - 1].set,
                        &self.x_info[i - 1].set,
                    ) + if self.score.x[i - 1][0] == 0.0 {
                        self.scoring.gap_open(self.y_blen)
                    } else {
                        self.scoring.gap_ext(self.y_blen)
                    }
                } else {
                    0.0
                };
            if !self.x_info[i - 1].is_fixed() {
                self.trace.m[i][0] = GapInY;
                self.trace.x[i][0] = GapInY;
                self.trace.y[i][0] = GapInY;
            }
            self.score.y[i][0] = f64::INFINITY;
            self.score.m[i][0] = f64::INFINITY;
        }
    }

    fn init_y(&mut self) {
        for j in 1..self.cols {
            self.score.y[0][j] = self.score.y[0][j - 1]
                + if self.y_info[j - 1].no_gap() {
                    self.score_match_one_branch(
                        self.y_blen,
                        &self.y_info[j - 1].set,
                        &self.y_info[j - 1].set,
                    ) + if self.score.y[0][j - 1] == 0.0 {
                        self.scoring.gap_open(self.x_blen)
                    } else {
                        self.scoring.gap_ext(self.x_blen)
                    }
                } else {
                    0.0
                };
            if !self.y_info[j - 1].is_fixed() {
                self.trace.m[0][j] = GapInX;
                self.trace.x[0][j] = GapInX;
                self.trace.y[0][j] = GapInX;
            }
            self.score.x[0][j] = f64::INFINITY;
            self.score.m[0][j] = f64::INFINITY;
        }
    }

    fn fill_s_m(&self, i: usize, j: usize) -> (f64, Direction) {
        let x_info = &self.x_info;
        let y_info = &self.y_info;

        let anc_set = if !(&x_info[i].set & &y_info[j].set).is_empty() {
            &x_info[i].set & &y_info[j].set
        } else {
            &x_info[i].set | &y_info[j].set
        };
        let match_score = self.score_match_both_branches(&anc_set, &x_info[i].set, &y_info[j].set);

        let (x_gap_adj, y_gap_adj) = self.score_match_gap_cost_adjustment(i, j);
        self.select_direction(
            self.score.m[i][j] + match_score,
            self.score.x[i][j] + x_gap_adj + match_score,
            self.score.y[i][j] + y_gap_adj + match_score,
        )
    }

    fn score_match_both_branches(
        &self,
        a_set: &ParsimonySet,
        x_set: &ParsimonySet,
        y_set: &ParsimonySet,
    ) -> f64 {
        a_set
            .iter()
            .map(|ancestor| {
                self.min_score(self.x_blen, x_set, ancestor)
                    + self.min_score(self.y_blen, y_set, ancestor)
            })
            .fold(f64::INFINITY, f64::min)
    }

    fn score_match_gap_cost_adjustment(&self, i: usize, j: usize) -> (f64, f64) {
        if !self.x_info[i].is_ext() && !self.y_info[j].is_ext() {
            return (0.0, 0.0);
        }
        let x_gap_adj = if self.y_info[j].is_ext() {
            zip(
                self.trace.x.iter().take(i + 1),
                self.x_info.iter().take(i + 1),
            )
            .rev()
            .find(|(dir, info)| dir[j] != GapInY && !info.is_fixed())
            .filter(|(dir, _)| dir[j] != Matc)
            .map_or(0.0, |_| {
                self.scoring.gap_open(self.y_blen) - self.scoring.gap_ext(self.y_blen)
            })
        } else {
            self.scoring.gap_open(self.y_blen) - self.scoring.gap_ext(self.y_blen)
        };
        let y_gap_adj = if self.x_info[i].is_ext() {
            zip(
                self.trace.y[i].iter().take(j + 1),
                self.y_info.iter().take(j + 1),
            )
            .rev()
            .find(|(dir, info)| **dir != GapInX && !info.is_fixed())
            .filter(|(dir, _)| **dir != Matc)
            .map_or(0.0, |_| {
                self.scoring.gap_open(self.x_blen) - self.scoring.gap_ext(self.x_blen)
            })
        } else {
            self.scoring.gap_open(self.x_blen) - self.scoring.gap_ext(self.x_blen)
        };
        (x_gap_adj, y_gap_adj)
    }

    fn fill_s_x(&self, i: usize, j: usize) -> (f64, Direction) {
        let (sm, sx, sy) = match self.x_info[i].flag {
            GapOpen | GapFixed => (self.score.m[i][j], self.score.x[i][j], self.score.y[i][j]),
            GapExt => (
                self.score.m[i][j] + self.scoring.gap_open(self.x_blen)
                    - self.scoring.gap_ext(self.x_blen),
                self.score.x[i][j],
                self.score.y[i][j] + self.gap_y_cost_adjustment(i, j),
            ),
            NoGap => {
                let match_score = self.score_match_one_branch(
                    self.x_blen,
                    &self.x_info[i].set,
                    &self.x_info[i].set,
                );
                (
                    self.score.m[i][j] + match_score + self.scoring.gap_open(self.y_blen),
                    self.score.x[i][j] + match_score + self.new_gap_y_score(i, j),
                    self.score.y[i][j] + match_score + self.scoring.gap_open(self.y_blen),
                )
            }
        };
        self.select_direction(sm, sx, sy)
    }

    fn gap_y_cost_adjustment(&self, i: usize, j: usize) -> f64 {
        let mut ni = i;
        let mut nj = j;
        while nj > 0
            && ni > 0
            && (self.x_info[ni - 1].is_fixed()
                || self.y_info[nj - 1].is_fixed()
                || self.trace.y[ni][nj] == GapInX)
        {
            ni -= (self.x_info[ni - 1].is_fixed()) as usize;
            nj -= (self.y_info[nj - 1].is_fixed() || self.trace.y[ni][nj] == GapInX) as usize;
        }
        if self.trace.y[ni][nj] != GapInY {
            self.scoring.gap_open(self.x_blen) - self.scoring.gap_ext(self.x_blen)
        } else {
            0.0
        }
    }

    fn new_gap_y_score(&self, i: usize, j: usize) -> f64 {
        zip(
            self.trace.x.iter().take(i + 1).skip(1),
            self.x_info.iter().take(i + 1),
        )
        .rev()
        .find(|(dir, info)| !(dir[j] == GapInY && info.is_possible()))
        .filter(|(dir, info)| !(dir[j] != GapInY && info.is_possible()))
        .map_or(self.scoring.gap_open(self.y_blen), |_| {
            self.scoring.gap_ext(self.y_blen)
        })
    }

    fn fill_s_y(&self, i: usize, j: usize) -> (f64, Direction) {
        let (sm, sx, sy) = match self.y_info[j].flag {
            GapFixed | GapOpen => (self.score.m[i][j], self.score.x[i][j], self.score.y[i][j]),
            GapExt => (
                self.score.m[i][j] + self.scoring.gap_open(self.y_blen)
                    - self.scoring.gap_ext(self.y_blen),
                self.score.x[i][j] + self.gap_x_cost_adjustment(i, j),
                self.score.y[i][j],
            ),
            NoGap => {
                let match_score = self.score_match_one_branch(
                    self.y_blen,
                    &self.y_info[j].set,
                    &self.y_info[j].set,
                );
                (
                    self.score.m[i][j] + match_score + self.scoring.gap_open(self.x_blen),
                    self.score.x[i][j] + match_score + self.scoring.gap_open(self.x_blen),
                    self.score.y[i][j] + match_score + self.new_x_gap_score(i, j),
                )
            }
        };
        self.select_direction(sm, sx, sy)
    }

    fn gap_x_cost_adjustment(&self, i: usize, j: usize) -> f64 {
        let mut ni = i;
        let mut nj = j;
        while ni > 0
            && nj > 0
            && (self.y_info[nj - 1].is_fixed()
                || self.x_info[ni - 1].is_fixed()
                || self.trace.x[ni][nj] == GapInY)
        {
            ni -= (self.x_info[ni - 1].is_fixed() || self.trace.x[ni][nj] == GapInY) as usize;
            nj -= (self.y_info[nj - 1].is_fixed()) as usize;
        }
        if self.trace.x[ni][nj] != GapInX {
            self.scoring.gap_open(self.y_blen) - self.scoring.gap_ext(self.y_blen)
        } else {
            0.0
        }
    }

    fn new_x_gap_score(&self, i: usize, j: usize) -> f64 {
        zip(
            self.trace.y[i].iter().take(j + 1).skip(1),
            self.y_info.iter().take(j + 1),
        )
        .rev()
        .find(|(dir, info)| !(**dir == GapInX && info.is_possible()))
        .filter(|(dir, info)| !(**dir != GapInY && info.is_possible()))
        .map_or(self.scoring.gap_open(self.x_blen), |_| {
            self.scoring.gap_ext(self.x_blen)
        })
    }

    pub(crate) fn traceback(&self) -> (Vec<ParsimonySite>, PairwiseAlignment, f64) {
        let mut i = self.rows - 1;
        let mut j = self.cols - 1;
        let (pars_score, mut action) =
            self.select_direction(self.score.m[i][j], self.score.x[i][j], self.score.y[i][j]);
        let max_alignment_length = self.rows + self.cols - 2;
        let mut node_info = Vec::<ParsimonySite>::with_capacity(max_alignment_length);
        let mut alignment = PairwiseAlignment::new(
            Mapping::with_capacity(max_alignment_length),
            Mapping::with_capacity(max_alignment_length),
        );
        while i > 0 || j > 0 {
            if (i > 0 && self.x_info[i - 1].is_fixed()) || (j > 0 && self.y_info[j - 1].is_fixed())
            {
                if i > 0 && self.x_info[i - 1].is_fixed() {
                    i -= 1;
                    alignment.map_x.push(Some(i));
                    alignment.map_y.push(None);
                    node_info.push(ParsimonySite::new(ParsimonySet::gap(), GapFixed));
                }
                if j > 0 && self.y_info[j - 1].is_fixed() {
                    j -= 1;
                    alignment.map_x.push(None);
                    alignment.map_y.push(Some(j));
                    node_info.push(ParsimonySite::new(ParsimonySet::gap(), GapFixed));
                }
            } else {
                let (map_x, map_y, set, flag) = match action {
                    Matc => {
                        action = self.trace.m[i][j];
                        i -= 1;
                        j -= 1;
                        let mut set = &self.x_info[i].set & &self.y_info[j].set;
                        if set.is_empty() {
                            set = &self.x_info[i].set | &self.y_info[j].set;
                        }
                        (Some(i), Some(j), set, NoGap)
                    }
                    GapInY => {
                        action = self.trace.x[i][j];
                        i -= 1;
                        let (set, flag) = match self.x_info[i].flag {
                            GapOpen | GapExt => (ParsimonySet::gap(), GapFixed),
                            NoGap => (self.x_info[i].set.clone(), self.gap_x_open_or_ext(i, j)),
                            GapFixed => unreachable!(),
                        };
                        (Some(i), None, set, flag)
                    }
                    GapInX => {
                        action = self.trace.y[i][j];
                        j -= 1;
                        let (set, flag) = match self.y_info[j].flag {
                            GapOpen | GapExt => (ParsimonySet::gap(), GapFixed),
                            NoGap => (self.y_info[j].set.clone(), self.gap_y_open_or_ext(i, j)),
                            GapFixed => unreachable!(),
                        };
                        (None, Some(j), set, flag)
                    }
                };
                node_info.push(ParsimonySite::new(set, flag));
                alignment.map_x.push(map_x);
                alignment.map_y.push(map_y);
            }
        }
        node_info.reverse();
        alignment.map_x.reverse();
        alignment.map_y.reverse();
        (node_info, alignment, pars_score)
    }

    fn gap_x_open_or_ext(&self, i: usize, j: usize) -> SiteFlag {
        if self.trace.x[i + 1][j] != GapInY
            || i == 0
            || (self.x_info[i - 1].is_possible() && self.trace.x[i][j] != GapInX)
        {
            GapOpen
        } else {
            GapExt
        }
    }

    fn gap_y_open_or_ext(&self, i: usize, j: usize) -> SiteFlag {
        if self.trace.y[i][j + 1] != GapInX
            || j == 0
            || (self.y_info[j - 1].is_possible() && self.trace.y[i][j] != GapInY)
        {
            GapOpen
        } else {
            GapExt
        }
    }
}

#[cfg(test)]
mod tests;
