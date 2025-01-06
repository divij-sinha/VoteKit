from functools import lru_cache
from typing import Optional

import numpy as np
import pandas as pd
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

from .ballot import Ballot
from .pref_profile import PreferenceProfile


@np.vectorize
@lru_cache(maxsize=None)
def score_ballots_to_array(x):
    r = pd.Series(x.scores)
    return r


@np.vectorize
@lru_cache(maxsize=None)
def rank_ballots_to_array(x):
    r = np.array(x.ranking)
    return r


@lru_cache
def array_to_ballots_hashable(x):
    return Ballot(x)


@np.vectorize
def array_to_ballots(x):
    return array_to_ballots_hashable(tuple([k for k in x if k is not None]))


@dataclass(frozen=True, config=ConfigDict(arbitrary_types_allowed=True))
class SenstivityBaseClass:
    """
    Base class for sensitivity tests. Takes in a PreferenceProfile object and returns new PreferenceProfile objects, subject to the sensitivity test performed.

    Args:
        pref_profile (PreferenceProfile): PreferenceProfile object.
        n_iters (int): Number of iterations for the sensitivity test.
        volume (float): Volume of ballots to drop
    """

    pref_profile: PreferenceProfile
    n_iters: int = 10_000
    volume: float = 0.95
    # def __init__(
    #     self,
    #     pref_profile: PreferenceProfile,
    #     n_iters: int = 10_000,
    #     volume: float = 0.95,
    #     # n_ranking: int = None,
    #     # labels: list | np.ndarray = None,
    #     # voting_method: str = "pickone",
    #     # max_rank: int = 5,
    # ) -> None:
    #     self.pref_profile = pref_profile


@dataclass(frozen=True, config=ConfigDict(arbitrary_types_allowed=True))
class DropoutSensitivity(SenstivityBaseClass):
    def next_pref_profile(self):
        """
        Return n_iters of pref_profiles, holding on to volume of the voters and dropping the rest.
        """
        raw_ballots = np.array(self.pref_profile.ballots)
        self.pref_profile.total_ballot_wt
        self.rng = np.random.default_rng()
        for _ in range(self.n_iters):
            sample_indices = self.rng.choice(
                len(raw_ballots),
                size=int(len(raw_ballots) * (1 - self.volume)),
                replace=False,
            )
            cur_ballot_list = raw_ballots[sample_indices].tolist()
            cur_pp = PreferenceProfile(cur_ballot_list, candidates=self.pref_profile.candidates)
            yield cur_pp


@dataclass(frozen=True, config=ConfigDict(arbitrary_types_allowed=True))
class JitterSensitivity(SenstivityBaseClass):
    """
    Jitter the scores or rankings of the selected volume of the ballots in the PreferenceProfile object and return n_iter jittered PreferenceProfile objects.

    Args:
        max_score (int): Maximum score for the scores. Default is None. Must be entered if scores are present.
        min_score (int): Minimum score for the scores. Default is 0. Must be entered if scores are present.
        score_move (float | int): The amount by which to move the scores. Default is 1. Must be entered if scores are present.
        max_rank (int): Maximum rank for the rankings. Default is None. Must be entered if rankings are present.
    """

    max_score: Optional[int] = None
    min_score: Optional[int] = 0
    score_move: Optional[float | int] = 1
    max_rank: Optional[int] = None

    def _run_rank_iter(self, _):
        cur_ballots_array = self.ballots_array.copy()
        sample_indices = self.rng.choice(
            len(self.ballots), size=int(self.sample_n / 2), replace=True
        )
        c_df = pd.DataFrame([sample_indices]).T.value_counts().reset_index(name="count")
        while c_df.shape[0] > 0:
            sample_indices = c_df.iloc[:, 0].to_numpy()
            col_1 = self.rng.choice(
                np.min([self.max_rank, self.df.shape[1]]), size=sample_indices.shape
            )
            col_2 = self.rng.choice([-1, 1], size=col_1.shape) + col_1
            col_2[col_2 < 0] = col_1[col_2 < 0]
            col_2[col_2 >= np.min([self.max_rank, self.df.shape[1]])] = col_1[
                col_2 >= np.min([self.max_rank, self.df.shape[1]])
            ]
            t = cur_ballots_array[sample_indices, col_1]
            cur_ballots_array[sample_indices, col_1] = cur_ballots_array[sample_indices, col_2]
            cur_ballots_array[sample_indices, col_2] = t
            c_df = c_df.loc[c_df["count"] > 1]
            c_df.loc[:, "count"] = c_df["count"] - 1
        jittered_ballots = array_to_ballots(cur_ballots_array).tolist()
        irv_profile = PreferenceProfile(ballots=jittered_ballots)
        return irv_profile

    def _run_score_iter(self):
        curiter_df = np.copy(self.df)
        rows = self.rng.choice(self.df.shape[0], size=self.sample_n)
        cols = self.rng.choice(self.df.shape[1], size=self.sample_n)
        rc_df = pd.DataFrame([rows, cols]).T.value_counts().reset_index(name="count")
        while rc_df.shape[0] > 0:
            rows = rc_df.iloc[:, 0].to_numpy()
            cols = rc_df.iloc[:, 1].to_numpy()
            # print(rows.shape, cols.shape)
            to_jitter = curiter_df[rows, cols]
            proposed_moves = self.score_move * (np.random.choice([1, -1], size=to_jitter.shape))
            jittered = to_jitter + proposed_moves
            curiter_df[rows, cols] = jittered
            rc_df = rc_df.loc[rc_df["count"] > 1]
            rc_df.loc[:, "count"] = rc_df["count"] - 1
        curiter_df[curiter_df < self.min_score] = self.min_score
        curiter_df[curiter_df > self.max_score] = self.max_score
        return PreferenceProfile(
            curiter_df.apply(lambda x: Ballot(scores=x.to_dict()), axis=1).to_list()
        )

    def next_pref_profile(self):
        """
        Return n_iters of pref_profiles, jittering volume of the voters and holding the rest.
        """
        raw_ballots = np.array(self.pref_profile.ballots)
        self.rng = np.random.default_rng()
        if raw_ballots[0].ranking is not None:
            if self.max_rank is None:
                raise ValueError("max_rank must be entered for ranking ballots.")
            self.sample_n = (
                len(raw_ballots)
                * np.sum(
                    np.arange(
                        len(self.pref_profile.candidates)
                        - np.min([self.max_rank, len(self.pref_profile.candidates)])
                        + 1,
                        len(self.pref_profile.candidates) + 1,
                    )
                )
                * self.volume
            )
            self.ballots_array = np.stack(score_ballots_to_array(self.ballots))
            for _ in np.arange(self.n_iters):
                yield self._run_rank_iter(_)
        elif raw_ballots[0].scores is not None:
            if self.max_score is None or self.score_move is None or self.min_score is None:
                raise ValueError(
                    "min_score, max_score, score_move must be entered for score ballots."
                )
            self.sample_n = (
                self.pref_profile.total_ballot_wt * self.volume * (self.max_score - self.min_score)
            )
            self.df = pd.DataFrame(score_ballots_to_array(raw_ballots.ballots).tolist())
            for _ in range(self.n_iters):
                yield self._run_score_iter()
        else:
            raise ValueError("Ballots must have ranking or scores to jitter.")
