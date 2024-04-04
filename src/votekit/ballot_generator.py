from abc import abstractmethod
from functools import reduce
import itertools as it
from fractions import Fraction
import math
import numpy as np
from pathlib import Path
import pickle
import random
import warnings
from typing import Optional, Union, Tuple
import apportionment.methods as apportion  # type: ignore

from .ballot import Ballot
from .pref_profile import PreferenceProfile
from .pref_interval import combine_preference_intervals, PreferenceInterval


def sample_cohesion_ballot_types(
    slate_to_non_zero_candidates: dict,
    num_ballots: int,
    cohesion_parameters_for_bloc: dict,
):
    """
    Used to generate bloc orderings given cohesion parameters.

    Args:
        slate_to_non_zero_candidates (dict): A mapping of slates to their list of non_zero
                                            candidates.
        num_ballots (int): the number of ballots to generate.
        cohesion_parameters_for_bloc (dict): A mapping of blocs to cohesion parameters.
                                Note, this is equivalent to one value in the cohesion_parameters
                                dictionary.


    Returns:
      A list of lists of length `num_ballots`, where each sublist contains the bloc names in order
      they appear on that ballot.
    """
    candidates = list(it.chain(*list(slate_to_non_zero_candidates.values())))
    ballots = [[-1]] * num_ballots
    # precompute coin flips
    coin_flips = list(np.random.uniform(size=len(candidates) * num_ballots))

    def which_bin(dist_bins, flip):
        for i, bin in enumerate(dist_bins):
            if bin < flip <= dist_bins[i + 1]:
                return i

    blocs_og, values_og = [list(x) for x in zip(*cohesion_parameters_for_bloc.items())]

    for j in range(num_ballots):
        blocs, values = blocs_og.copy(), values_og.copy()
        # Pre-calculate distribution_bins
        distribution_bins = [0] + [sum(values[: i + 1]) for i in range(len(blocs))]
        ballot_type = [-1] * len(candidates)

        for i, flip in enumerate(
            coin_flips[j * len(candidates) : (j + 1) * len(candidates)]
        ):
            bloc_index = which_bin(distribution_bins, flip)
            bloc_type = blocs[bloc_index]
            ballot_type[i] = bloc_type

            # Check if that exhausts a slate of candidates
            if ballot_type.count(bloc_type) == len(
                slate_to_non_zero_candidates[bloc_type]
            ):
                del blocs[bloc_index]
                del values[bloc_index]
                total_value_sum = sum(values)
                values = [v / total_value_sum for v in values]
                distribution_bins = [0] + [
                    sum(values[: i + 1]) for i in range(len(blocs))
                ]

        ballots[j] = ballot_type

    return ballots


class BallotGenerator:
    """
    Base class for ballot generation models that use the candidate simplex
    (e.g. Plackett-Luce, Bradley-Terry, etc.).

    **Attributes**

    `candidates`
    :   list of candidates in the election.

    `cohesion_parameters`
    : dictionary of dictionaries mapping of bloc to cohesion parameters.
        (ex. {bloc_1: {bloc_1: .7, bloc_2: .2, bloc_3:.1}})

    `pref_intervals_by_bloc`
    :   dictionary of dictionaries mapping of bloc to preference intervals.
        (ex. {bloc_1: {bloc_1 : PI, bloc_2: PI}}).

    `bloc_voter_prop`
    :   dictionary mapping of bloc to voter proportions (ex. {bloc: voter proportion}).


    ???+ note
        * Voter proportion for blocs must sum to 1.
        * Preference interval for candidates must sum to 1.
        * Must have same blocs in `pref_intervals_by_bloc` and `bloc_voter_prop`.

    **Methods**
    """

    def __init__(
        self,
        **kwargs,
    ):
        if "candidates" not in kwargs and "slate_to_candidates" not in kwargs:
            raise ValueError(
                "At least one of candidates or slate_to_candidates must be provided."
            )

        if "candidates" in kwargs:
            self.candidates = kwargs["candidates"]

        if "slate_to_candidates" in kwargs:
            self.slate_to_candidates = kwargs["slate_to_candidates"]
            self.candidates = [
                c for c_list in self.slate_to_candidates.values() for c in c_list
            ]

        nec_parameters = [
            "pref_intervals_by_bloc",
            "cohesion_parameters",
            "bloc_voter_prop",
        ]

        if any(x in kwargs for x in nec_parameters):
            if not all(x in kwargs for x in nec_parameters):
                raise ValueError(
                    f"If one of {nec_parameters} is provided, all must be provided."
                )

            bloc_voter_prop = kwargs["bloc_voter_prop"]
            pref_intervals_by_bloc = kwargs["pref_intervals_by_bloc"]
            cohesion_parameters = kwargs["cohesion_parameters"]

            if round(sum(bloc_voter_prop.values()), 8) != 1.0:
                raise ValueError("Voter proportion for blocs must sum to 1")

            if bloc_voter_prop.keys() != pref_intervals_by_bloc.keys():
                raise ValueError(
                    "Blocs are not the same between bloc_voter_prop and pref_intervals_by_bloc."
                )

            if bloc_voter_prop.keys() != cohesion_parameters.keys():
                raise ValueError(
                    "Blocs are not the same between bloc_voter_prop and cohesion_parameters."
                )

            if pref_intervals_by_bloc.keys() != cohesion_parameters.keys():
                raise ValueError(
                    "Blocs are not the same between pref_intervals_by_bloc and cohesion_parameters."
                )

            for bloc, cohesion_parameter_dict in cohesion_parameters.items():
                if round(sum(cohesion_parameter_dict.values()), 8) != 1.0:
                    raise ValueError(
                        f"Cohesion parameters for bloc {bloc} must sum to 1."
                    )

            self.pref_intervals_by_bloc = pref_intervals_by_bloc
            self.bloc_voter_prop = bloc_voter_prop
            self.blocs = list(self.bloc_voter_prop.keys())
            self.cohesion_parameters = cohesion_parameters

    @classmethod
    def from_params(
        cls,
        slate_to_candidates: dict,
        bloc_voter_prop: dict,
        cohesion_parameters: dict,
        alphas: dict,
        **data,
    ):
        """
        Initializes a BallotGenerator by constructing a preference interval
        from parameters; the prior parameters (if inputted) will be overwritten.

        Args:
            slate_to_candidates (dict): A mapping of blocs to candidates
                (ex. {bloc: [candidate]})
            bloc_voter_prop (dict): A mapping of the percentage of total voters
                 per bloc (ex. {bloc: 0.7})
            cohesion_parameters (dict): Cohension factors for each bloc (ex. {bloc_1: {bloc_1: .9,
                                                                                        bloc_2:.1})
            alphas (dict): Alpha for the Dirichlet distribution of each bloc
                            (ex. {bloc: {bloc: 1, opposing_bloc: 1/2}}).

        Raises:
            ValueError: If the voter proportion for blocs don't sum to 1.
            ValueError: Blocs are not the same.

        Returns:
            (BallotGenerator): Initialized ballot generator.

        ???+ note
            * Voter proportion for blocs must sum to 1.
            * Each cohesion parameter must be in the interval [0,1].
            * Dirichlet parameters are in the interval $(0,\infty)$.
        """
        if round(sum(bloc_voter_prop.values()), 8) != 1.0:
            raise ValueError("Voter proportion for blocs must sum to 1")

        if slate_to_candidates.keys() != bloc_voter_prop.keys():
            raise ValueError("Blocs are not the same")

        pref_intervals_by_bloc = {}
        for current_bloc in bloc_voter_prop:
            intervals = {}
            for b in bloc_voter_prop:
                interval = PreferenceInterval.from_dirichlet(
                    candidates=slate_to_candidates[b], alpha=alphas[current_bloc][b]
                )
                intervals[b] = interval

            pref_intervals_by_bloc[current_bloc] = intervals

        if "candidates" not in data:
            cands = [cand for cands in slate_to_candidates.values() for cand in cands]
            data["candidates"] = cands

        data["pref_intervals_by_bloc"] = pref_intervals_by_bloc
        data["bloc_voter_prop"] = bloc_voter_prop
        data["cohesion_parameters"] = cohesion_parameters

        if cls in [
            AlternatingCrossover,
            slate_PlackettLuce,
            slate_BradleyTerry,
            CambridgeSampler,
        ]:
            generator = cls(
                slate_to_candidates=slate_to_candidates,
                **data,
            )

        else:
            generator = cls(**data)

        return generator

    @abstractmethod
    def generate_profile(
        self, number_of_ballots: int, by_bloc: bool = False
    ) -> Union[PreferenceProfile, Tuple, dict]:
        """
        Generates a `PreferenceProfile`.

        Args:
            number_of_ballots (int): Number of ballots to generate.
            by_bloc (bool): True if you want a tuple (pp_by_bloc, pp), which is a dictionary of
                            PreferenceProfiles with keys = blocs and the aggregated profile.
                    False if you want the aggregated profile. Defaults to False.

        Returns:
            (PreferenceProfile): A generated `PreferenceProfile`.
        """
        pass

    @staticmethod
    def _round_num(num: float) -> int:
        """
        Rounds up or down a float randomly.

        Args:
            num (float): Number to round.

        Returns:
            int: A whole number.
        """
        rand = np.random.random()
        return math.ceil(num) if rand > 0.5 else math.floor(num)

    @staticmethod
    def ballot_pool_to_profile(ballot_pool, candidates) -> PreferenceProfile:
        """
        Given a list of ballots and candidates, convert them into a `PreferenceProfile.`

        Args:
            ballot_pool (list of tuple): A list of ballots, where each ballot is a tuple
                    of candidates indicating their ranking from top to bottom.
            candidates (list): A list of candidates.

        Returns:
            (PreferenceProfile): A PreferenceProfile representing the ballots in the election.
        """
        ranking_counts: dict[tuple, int] = {}
        ballot_list: list[Ballot] = []

        for ranking in ballot_pool:
            tuple_rank = tuple(ranking)
            ranking_counts[tuple_rank] = (
                ranking_counts[tuple_rank] + 1 if tuple_rank in ranking_counts else 1
            )

        for ranking, count in ranking_counts.items():
            rank = tuple([frozenset([cand]) for cand in ranking])
            b = Ballot(ranking=rank, weight=Fraction(count))
            ballot_list.append(b)

        return PreferenceProfile(ballots=ballot_list, candidates=candidates)


class BallotSimplex(BallotGenerator):
    """
    Base class for ballot generation models that use the ballot simplex
    (e.g. ImpartialCulture, ImpartialAnonymousCulture).

    **Attributes**

    `alpha`
    :   (float) alpha parameter for ballot simplex. Defaults to None.

    `point`
    :   dictionary representing a point in the ballot simplex with candidate as
        keys and electoral support as values. Defaults to None.

    ???+ note

        Point or alpha arguments must be included to initialize.

    **Methods**
    """

    def __init__(
        self, alpha: Optional[float] = None, point: Optional[dict] = None, **data
    ):
        if alpha is None and point is None:
            raise AttributeError("point or alpha must be initialized")
        self.alpha = alpha
        if alpha == float("inf"):
            self.alpha = 1e20
        if alpha == 0:
            self.alpha = 1e-10
        self.point = point
        super().__init__(**data)

    @classmethod
    def from_point(cls, point: dict, **data):
        """
        Initializes a Ballot Simplex model from a point in the Dirichlet distribution.

        Args:
            point (dict): A mapping of candidate to candidate support.

        Raises:
            ValueError: If the candidate support does not sum to 1.

        Returns:
            (BallotSimplex): Initialized from point.
        """
        if sum(point.values()) != 1.0:
            raise ValueError(
                f"probability distribution from point ({point.values()}) does not sum to 1"
            )
        return cls(point=point, **data)

    @classmethod
    def from_alpha(cls, alpha: float, **data):
        """
        Initializes a Ballot Simplex model from an alpha value for the Dirichlet
        distribution.

        Args:
            alpha (float): An alpha parameter for the Dirichlet distribution.

        Returns:
            (BallotSimplex): Initialized from alpha.
        """

        return cls(alpha=alpha, **data)

    def generate_profile(
        self, number_of_ballots, by_bloc: bool = False
    ) -> Union[PreferenceProfile, dict]:
        """
        Generates a PreferenceProfile from the ballot simplex.
        """

        perm_set = it.permutations(self.candidates, len(self.candidates))

        perm_rankings = [list(value) for value in perm_set]

        if self.alpha is not None:
            draw_probabilities = list(
                np.random.default_rng().dirichlet([self.alpha] * len(perm_rankings))
            )

        elif self.point:
            # calculates probabilities for each ranking
            # using probability distribution for candidate support
            draw_probabilities = [
                reduce(
                    lambda prod, cand: prod * self.point[cand] if self.point else 0,
                    ranking,
                    1.0,
                )
                for ranking in perm_rankings
            ]
            draw_probabilities = [
                prob / sum(draw_probabilities) for prob in draw_probabilities
            ]

        indices = np.random.choice(
            a=len(perm_rankings), size=number_of_ballots, p=draw_probabilities
        )
        ballot_pool = [perm_rankings[indices[i]] for i in range(number_of_ballots)]

        return self.ballot_pool_to_profile(ballot_pool, self.candidates)


class ImpartialCulture(BallotSimplex):
    """
    Impartial Culture model with an alpha value of 1e10 (should be infinity theoretically).
    This model is uniform on all linear rankings.


    **Attributes**

    `candidates`
    : (list) a list of candidates

    `alpha`
    :   (float) alpha parameter for ballot simplex. Defaults to None.

    `point`
    :   dictionary representing a point in the ballot simplex with candidate as
        keys and electoral support as values. Defaults to None.



    **Methods**

    See `BallotSimplex` object.

    ???+ note

        Point or alpha arguments must be included to initialize. For details see
        `BallotSimplex` and `BallotGenerator` object.
    """

    def __init__(self, **data):
        super().__init__(alpha=float("inf"), **data)


class ImpartialAnonymousCulture(BallotSimplex):
    """
    Impartial Anonymous Culture model with an alpha value of 1. This model choose uniformly
        from among all distributions on full linear rankings, and then draws according to the
        chosen distribution.

    **Attributes**

    `candidates`
    : (list) a list of candidates

    `alpha`
    :   (float) alpha parameter for ballot simplex. Defaults to None.

    `point`
    :   dictionary representing a point in the ballot simplex with candidate as
        keys and electoral support as values. Defaults to None.

    **Methods**

    See `BallotSimplex` base class.

    ???+ note

        Point or alpha arguments must be included to initialize. For details see
        `BallotSimplex` and `BallotGenerator` object.
    """

    def __init__(self, **data):
        super().__init__(alpha=1, **data)


class short_name_PlackettLuce(BallotGenerator):
    """
    Class for generating short name Plackett Luce ballots. This model samples without
    replacement from a preference interval. Equivalent to name-PlackettLuce if
    ballot_length = number of  candidates. Can be initialized with an interval or can be
    constructed with the  Dirichlet distribution using the `from_params` method in the
    `BallotGenerator` class.

    **Attributes**

    `candidates`
    : a list of candidates.

    `pref_intervals_by_bloc`
    :   dictionary of dictionaries mapping of bloc to preference intervals.
        (ex. {bloc_1: {bloc_1 : PI, bloc_2: PI}}).

    `cohesion_parameters`
    : dictionary of dictionaries of cohesion parameters (ex. {bloc_1: {bloc_1:.7, bloc_2: .3}})

    `bloc_voter_prop`
    :   dictionary mapping of bloc to voter proportions (ex. {bloc: proportion}).

    `ballot_length`
    : number of votes allowed per ballot

    **Methods**

    See `BallotGenerator` base class
    """

    def __init__(self, ballot_length: int, **data):
        # Call the parent class's __init__ method to handle common parameters
        super().__init__(**data)
        self.ballot_length = ballot_length

        # if dictionary of pref intervals
        if isinstance(
            list(self.pref_intervals_by_bloc.values())[0], PreferenceInterval
        ):
            self.pref_interval_by_bloc = self.pref_intervals_by_bloc

        # if nested dictionary of pref intervals, combine by cohesion
        else:
            self.pref_interval_by_bloc = {
                bloc: combine_preference_intervals(
                    [self.pref_intervals_by_bloc[bloc][b] for b in self.blocs],
                    [self.cohesion_parameters[bloc][b] for b in self.blocs],
                )
                for bloc in self.blocs
            }

    def generate_profile(
        self, number_of_ballots: int, by_bloc: bool = False
    ) -> Union[PreferenceProfile, Tuple]:
        """
        Args:
        `number_of_ballots`: The number of ballots to generate.

        `by_bloc`: True if you want to return a dictionary of PreferenceProfiles by bloc.
                    False if you want the full, aggregated PreferenceProfile.
        """
        # the number of ballots per bloc is determined by Huntington-Hill apportionment
        bloc_props = list(self.bloc_voter_prop.values())
        ballots_per_block = dict(
            zip(
                self.blocs,
                apportion.compute("huntington", bloc_props, number_of_ballots),
            )
        )

        # dictionary to store preference profiles by bloc
        pp_by_bloc = {b: PreferenceProfile() for b in self.blocs}

        for bloc in self.blocs:
            # number of voters in this bloc
            num_ballots = ballots_per_block[bloc]
            ballot_pool = [Ballot()] * num_ballots
            non_zero_cands = list(self.pref_interval_by_bloc[bloc].non_zero_cands)
            pref_interval_values = [
                self.pref_interval_by_bloc[bloc].interval[c] for c in non_zero_cands
            ]
            zero_cands = list(self.pref_interval_by_bloc[bloc].zero_cands)

            # if there aren't enough non-zero supported candidates,
            # include 0 support as ties
            number_to_sample = self.ballot_length
            number_tied = None

            if len(non_zero_cands) < number_to_sample:
                number_tied = number_to_sample - len(non_zero_cands)
                number_to_sample = len(non_zero_cands)

            for i in range(num_ballots):
                # generates ranking based on probability distribution of non candidate support
                # samples ballot_length candidates
                non_zero_ranking = list(
                    np.random.choice(
                        non_zero_cands,
                        number_to_sample,
                        p=pref_interval_values,
                        replace=False,
                    )
                )

                ranking = [frozenset({cand}) for cand in non_zero_ranking]

                if number_tied:
                    tied_candidates = list(
                        np.random.choice(
                            zero_cands,
                            number_tied,
                            replace=False,
                        )
                    )
                    ranking.append(frozenset(tied_candidates))

                ballot_pool[i] = Ballot(ranking=tuple(ranking), weight=Fraction(1, 1))

            # create PP for this bloc
            pp = PreferenceProfile(ballots=ballot_pool)
            pp = pp.condense_ballots()
            pp_by_bloc[bloc] = pp

        # combine the profiles
        pp = PreferenceProfile(ballots=[])
        for profile in pp_by_bloc.values():
            pp += profile

        if by_bloc:
            return (pp_by_bloc, pp)

        # else return the combined profiles
        else:
            return pp


class name_PlackettLuce(short_name_PlackettLuce):
    """
    Class for generating full ballots with name-PlackettLuce. This model samples without
    replacement from a preference interval. Can be initialized with an interval or can be
    constructed with the Dirichlet distribution using the `from_params` method in the
    `BallotGenerator` class.

    **Attributes**

    `candidates`
    : a list of candidates.

    `pref_intervals_by_bloc`
    :   dictionary of dictionaries mapping of bloc to preference intervals.
        (ex. {bloc_1: {bloc_1 : PI, bloc_2: PI}}).

    `cohesion_parameters`
    : dictionary of dictionaries of cohesion parameters (ex. {bloc_1: {bloc_1:.7, bloc_2: .3}})

    `bloc_voter_prop`
    :   dictionary mapping of bloc to voter proportions (ex. {bloc: proportion}).


    **Methods**

    See `BallotGenerator` base class
    """

    def __init__(self, cohesion_parameters: dict, **data):
        if "candidates" in data:
            ballot_length = len(data["candidates"])
        elif "slate_to_candidates" in data:
            ballot_length = sum(
                len(c_list) for c_list in data["slate_to_candidates"].values()
            )
        else:
            raise ValueError("One of candidates or slate_to_candidates must be passed.")

        # Call the parent class's __init__ method to handle common parameters
        super().__init__(
            ballot_length=ballot_length, cohesion_parameters=cohesion_parameters, **data
        )


class name_BradleyTerry(BallotGenerator):
    """
    Class for generating ballots using a name-BradleyTerry model. The probability of sampling
    the ranking $X>Y>Z$ is proportional to $P(X>Y)*P(X>Z)*P(Y>Z)$.
    These individual probabilities are based on the preference interval: $P(X>Y) = x/(x+y)$.
    Can be initialized with an interval or can be constructed with the Dirichlet distribution using
    the `from_params` method in the `BallotGenerator` class.

    **Attributes**

    `candidates`
    : a list of candidates.

    `pref_intervals_by_bloc`
    : dictionary of dictionaries mapping of bloc to preference intervals or dictionary of PIs.
        (ex. {bloc_1: {bloc_1 : PI, bloc_2: PI}}).

    `bloc_voter_prop`
    :   dictionary mapping of slate to voter proportions (ex. {race: voter proportion}).

    `cohesion_parameters`
    : dictionary of dictionaries of cohesion parameters (ex. {bloc_1: {bloc_1:.7, bloc_2: .3}})

    **Methods**

    See `BallotGenerator` base class.
    """

    def __init__(self, cohesion_parameters: dict, **data):
        # Call the parent class's __init__ method to handle common parameters
        super().__init__(cohesion_parameters=cohesion_parameters, **data)

        # if dictionary of pref intervals
        if isinstance(
            list(self.pref_intervals_by_bloc.values())[0], PreferenceInterval
        ):
            self.pref_interval_by_bloc = self.pref_intervals_by_bloc

        # if nested dictionary of pref intervals, combine by cohesion
        else:
            self.pref_interval_by_bloc = {
                bloc: combine_preference_intervals(
                    [self.pref_intervals_by_bloc[bloc][b] for b in self.blocs],
                    [self.cohesion_parameters[bloc][b] for b in self.blocs],
                )
                for bloc in self.blocs
            }

        if len(self.candidates) < 12:
            # precompute pdfs for sampling
            self.pdfs_by_bloc = {
                bloc: self._BT_pdf(self.pref_interval_by_bloc[bloc].interval)
                for bloc in self.blocs
            }
        else:
            warnings.warn(
                "For 12 or more candidates, exact sampling is computationally infeasible. \
                    Please only use the built in generate_profile_MCMC method."
            )

    def _calc_prob(self, permutations: list[tuple], cand_support_dict: dict) -> dict:
        """
        given a list of (possibly incomplete) rankings and the preference interval, \
        calculates the probability of observing each ranking

        Args:
            permutations (list[tuple]): a list of permuted rankings
            cand_support_dict (dict): a mapping from candidate to their \
            support (preference interval)

        Returns:
            dict: a mapping of the rankings to their probability
        """
        ranking_to_prob = {}
        for ranking in permutations:
            prob = 1
            for i in range(len(ranking)):
                cand_i = ranking[i]
                greater_cand_support = cand_support_dict[cand_i]
                for j in range(i + 1, len(ranking)):
                    cand_j = ranking[j]
                    cand_support = cand_support_dict[cand_j]
                    prob *= greater_cand_support / (greater_cand_support + cand_support)
            ranking_to_prob[ranking] = prob
        return ranking_to_prob

    def _make_pow(self, lst):
        """
        Helper method for _BT_pdf.
        Takes is a list representing the preference lengths of each candidate
        in a permutation.
        Computes the numerator of BT probability.
        """
        ret = 1
        m = len(lst)
        for i, val in enumerate(lst):
            if i < m - 1:
                ret *= val ** (m - i - 1)
        return ret

    def _BT_pdf(self, dct):
        """
        Construct the BT pdf as a dictionary (ballot, probability) given a preference
        interval as a dictionary (candidate, preference).
        """

        # gives PI lengths for each candidate in permutation
        def pull_perm(lst):
            nonlocal dct
            return [dct[i] for i in lst]

        new_dct = {
            perm: self._make_pow(pull_perm(perm))
            for perm in it.permutations(dct.keys(), len(dct))
        }
        summ = sum(new_dct.values())
        return {key: value / summ for key, value in new_dct.items()}

    def generate_profile(
        self, number_of_ballots, by_bloc: bool = False
    ) -> Union[PreferenceProfile, Tuple]:
        # the number of ballots per bloc is determined by Huntington-Hill apportionment

        bloc_props = list(self.bloc_voter_prop.values())
        ballots_per_block = dict(
            zip(
                self.blocs,
                apportion.compute("huntington", bloc_props, number_of_ballots),
            )
        )

        pp_by_bloc = {b: PreferenceProfile() for b in self.blocs}

        for bloc in self.blocs:
            num_ballots = ballots_per_block[bloc]

            # Directly initialize the list using good memory trick
            ballot_pool = [Ballot()] * num_ballots
            zero_cands = self.pref_interval_by_bloc[bloc].zero_cands
            pdf_dict = self.pdfs_by_bloc[bloc]

            # Directly use the keys and values from the dictionary for sampling
            rankings, probs = zip(*pdf_dict.items())

            # The return of this will be a numpy array, so we don't need to make it into a list
            sampled_indices = np.array(
                np.random.choice(
                    a=len(rankings),
                    size=num_ballots,
                    p=probs,
                ),
                ndmin=1,
            )

            for j, index in enumerate(sampled_indices):
                ranking = [frozenset({cand}) for cand in rankings[index]]

                # Add any zero candidates as ties only if they exist
                if zero_cands:
                    ranking.append(frozenset(zero_cands))

                ballot_pool[j] = Ballot(ranking=tuple(ranking), weight=Fraction(1, 1))

            pp = PreferenceProfile(ballots=ballot_pool)
            pp = pp.condense_ballots()
            pp_by_bloc[bloc] = pp

        # combine the profiles
        pp = PreferenceProfile(ballots=[])
        for profile in pp_by_bloc.values():
            pp += profile

        if by_bloc:
            return (pp_by_bloc, pp)

        # else return the combined profiles
        else:
            return pp

    def _BT_mcmc(
        self, num_ballots, pref_interval, seed_ballot, zero_cands={}, verbose=False
    ):
        """
        Sample from BT distribution for a given preference interval using MCMC.

        num_ballots (int): the number of ballots to sample
        pref_interval (dict): the preference interval to determine BT distribution
        sub_sample_length (int): how many attempts at swaps to make before saving ballot
        seed_ballot: Ballot, the seed ballot for the Markov chain
        verbose: bool, if True, print the acceptance ratio of the chain
        """

        # check that seed ballot has no ties
        for s in seed_ballot.ranking:
            if len(s) > 1:
                raise ValueError("Seed ballot contains ties")

        ballots = [-1] * num_ballots
        accept = 0
        current_ranking = list(seed_ballot.ranking)
        num_candidates = len(current_ranking)

        # presample swap indices
        swap_indices = [
            (j1, j1 + 1)
            for j1 in random.choices(range(num_candidates - 1), k=num_ballots)
        ]

        # generate MCMC sample
        for i in range(num_ballots):
            # choose adjacent pair to propose a swap
            j1, j2 = swap_indices[i]
            acceptance_prob = min(
                1,
                pref_interval[next(iter(current_ranking[j2]))]
                / pref_interval[next(iter(current_ranking[j1]))],
            )

            # if you accept, make the swap
            if random.random() < acceptance_prob:
                current_ranking[j1], current_ranking[j2] = (
                    current_ranking[j2],
                    current_ranking[j1],
                )
                accept += 1

            if len(zero_cands) > 0:
                ballots[i] = Ballot(ranking=current_ranking + [zero_cands])
            else:
                ballots[i] = Ballot(ranking=current_ranking)

        if verbose:
            print(
                f"Acceptance ratio as number accepted / total steps: {accept/num_ballots:.2}"
            )

        if -1 in ballots:
            raise ValueError("Some element of ballots list is not a ballot.")

        pp = PreferenceProfile(ballots=ballots)
        pp = pp.condense_ballots()
        return pp

    def generate_profile_MCMC(
        self, number_of_ballots: int, verbose=False, by_bloc: bool = False
    ) -> Union[PreferenceProfile, Tuple]:
        """
        Sample from the BT distribution using Markov Chain Monte Carlo. `number_of_ballots` should
        be sufficiently large to allow for convergence of the chain.

        Args:
            number_of_ballots (int): Number of ballots to generate.
            verbose (bool, optional): If True, print the acceptance ratio of the chain. Default
                                        is False.
            by_bloc (bool, optional): True if you want a tuple (pp_by_bloc, pp), which is a
                                    dictionary of  PreferenceProfiles with keys = blocs and the
                                    aggregated profile. False if you want the aggregated profile.
                                    Defaults to False.

        Returns:
            Generated ballots as a PreferenceProfile or tuple (dict, PreferenceProfile).
        """

        # the number of ballots per bloc is determined by Huntington-Hill apportionment
        bloc_props = list(self.bloc_voter_prop.values())
        ballots_per_block = dict(
            zip(
                self.blocs,
                apportion.compute("huntington", bloc_props, number_of_ballots),
            )
        )

        pp_by_bloc = {b: PreferenceProfile() for b in self.blocs}

        for bloc in self.blocs:
            num_ballots = ballots_per_block[bloc]
            pref_interval = self.pref_interval_by_bloc[bloc]
            pref_interval_dict = pref_interval.interval
            non_zero_cands = pref_interval.non_zero_cands
            zero_cands = pref_interval.zero_cands

            seed_ballot = Ballot(
                ranking=tuple([frozenset({c}) for c in non_zero_cands])
            )
            pp = self._BT_mcmc(
                num_ballots,
                pref_interval_dict,
                seed_ballot,
                zero_cands=zero_cands,
                verbose=verbose,
            )

            pp_by_bloc[bloc] = pp

        # combine the profiles
        pp = PreferenceProfile(ballots=[])
        for profile in pp_by_bloc.values():
            pp += profile

        if by_bloc:
            return (pp_by_bloc, pp)

        # else return the combined profiles
        else:
            return pp


class AlternatingCrossover(BallotGenerator):
    """
    Class for Alternating Crossover style of generating ballots.
    AC assumes that voters either rank all of their own blocs candidates above the other bloc,
    or the voters "crossover" and rank a candidate from the other bloc first, then alternate
    between candidates from their own bloc and the opposing.
    Should only be used when there are two blocs.

    Can be initialized with an interval or can be
    constructed with the Dirichlet distribution using the `from_params` method in the
    `BallotGenerator` class.

    **Attributes**

    `pref_intervals_by_bloc`
    :   dictionary of dictionaries mapping of bloc to preference intervals.
        (ex. {bloc_1: {bloc_1 : PI, bloc_2: PI}}).

    `bloc_voter_prop`
    :   dictionary mapping of slate to voter proportions (ex. {bloc: voter proportion}).

    `slate_to_candidates`
    :   dictionary mapping of slate to candidates (ex. {bloc: [candidate1, candidate2]}).

    `cohesion_parameters`
    : dictionary of dictionaries of cohesion parameters (ex. {bloc_1: {bloc_1:.7, bloc_2: .3}})

    **Methods**

    See `BallotGenerator` base class.
    """

    def __init__(
        self,
        cohesion_parameters: dict,
        **data,
    ):
        # Call the parent class's __init__ method to handle common parameters
        super().__init__(cohesion_parameters=cohesion_parameters, **data)

    def generate_profile(
        self, number_of_ballots: int, by_bloc: bool = False
    ) -> Union[PreferenceProfile, Tuple]:
        # compute the number of bloc and crossover voters in each bloc using Huntington Hill
        cohesion_parameters = {
            b: self.cohesion_parameters[b][b] for b in self.cohesion_parameters
        }

        voter_types = [(b, type) for b in self.blocs for type in ["bloc", "cross"]]

        voter_props = [
            cohesion_parameters[b] * self.bloc_voter_prop[b]
            if t == "bloc"
            else (1 - cohesion_parameters[b]) * self.bloc_voter_prop[b]
            for b, t in voter_types
        ]

        ballots_per_type = dict(
            zip(
                voter_types,
                apportion.compute("huntington", voter_props, number_of_ballots),
            )
        )

        pp_by_bloc = {b: PreferenceProfile() for b in self.blocs}

        for i, bloc in enumerate(self.blocs):
            ballot_pool = []
            num_bloc_ballots = ballots_per_type[(bloc, "bloc")]
            num_cross_ballots = ballots_per_type[(bloc, "cross")]

            pref_interval_dict = self.pref_intervals_by_bloc[bloc]

            opposing_slate = self.blocs[(i + 1) % 2]

            opposing_cands = list(pref_interval_dict[opposing_slate].interval.keys())
            bloc_cands = list(pref_interval_dict[bloc].interval.keys())

            pref_for_opposing = list(
                pref_interval_dict[opposing_slate].interval.values()
            )
            pref_for_bloc = list(pref_interval_dict[bloc].interval.values())

            for i in range(num_cross_ballots + num_bloc_ballots):
                bloc_cands = list(
                    np.random.choice(
                        bloc_cands,
                        p=pref_for_bloc,
                        size=len(bloc_cands),
                        replace=False,
                    )
                )
                opposing_cands = list(
                    np.random.choice(
                        opposing_cands,
                        p=pref_for_opposing,
                        size=len(opposing_cands),
                        replace=False,
                    )
                )

                if i < num_cross_ballots:
                    # alternate the bloc and opposing bloc candidates to create crossover ballots
                    ranking = [
                        frozenset({cand})
                        for pair in zip(opposing_cands, bloc_cands)
                        for cand in pair
                    ]
                else:
                    ranking = [frozenset({c}) for c in bloc_cands] + [
                        frozenset({c}) for c in opposing_cands
                    ]

                ballot = Ballot(ranking=tuple(ranking), weight=Fraction(1, 1))
                ballot_pool.append(ballot)

            pp = PreferenceProfile(ballots=ballot_pool)
            pp = pp.condense_ballots()
            pp_by_bloc[bloc] = pp

        # combine the profiles
        pp = PreferenceProfile(ballots=[])
        for profile in pp_by_bloc.values():
            pp += profile

        if by_bloc:
            return (pp_by_bloc, pp)

        # else return the combined profiles
        else:
            return pp


class OneDimSpatial(BallotGenerator):
    """
    1-D spatial model for ballot generation. Assumes the candidates are normally distributed on
    the real line. Then voters are also normally distributed, and vote based on Euclidean distance
    to the candidates.

    **Attributes**
    `candidates`
        : a list of candidates.

    See `BallotGenerator` base class.

    **Methods**

    See `BallotGenerator` base class.
    """

    def generate_profile(
        self, number_of_ballots: int, by_bloc: bool = False
    ) -> Union[PreferenceProfile, Tuple]:
        candidate_position_dict = {c: np.random.normal(0, 1) for c in self.candidates}
        voter_positions = np.random.normal(0, 1, number_of_ballots)

        ballot_pool = []

        for vp in voter_positions:
            distance_dict = {
                c: abs(v - vp) for c, v, in candidate_position_dict.items()
            }
            candidate_order = sorted(distance_dict, key=distance_dict.__getitem__)
            ballot_pool.append(candidate_order)

        return self.ballot_pool_to_profile(ballot_pool, self.candidates)


class CambridgeSampler(BallotGenerator):
    """
    Class for generating ballots based on historical RCV elections occurring
    in Cambridge. Alternative election data can be used if specified. Assumes that there are two
    blocs, a majority and a minority bloc, and determines this based on the bloc_voter_prop attr.

    Based on cohesion parameters, decides if a voter casts their top choice within their bloc
    or in the opposing bloc. Then uses historical data; given their first choice, choose a
    ballot type from the historical distribution.


    **Attributes**

    `slate_to_candidates`
    :   dictionary mapping of slate to candidates (ex. {bloc: [candidate]}).

    `bloc_voter_prop`
    :   dictionary mapping of bloc to voter proportions (ex. {bloc: voter proportion}).

    `cohesion_parameters`
    : dictionary of dictionaries of cohesion parameters (ex. {bloc_1: {bloc_1:.7, bloc_2: .3}})

    `pref_intervals_by_bloc`
    :   dictionary of dictionaries mapping of bloc to preference intervals.
        (ex. {bloc_1: {bloc_1 : PI, bloc_2: PI}}).

    `historical_majority`
    : name of majority bloc in historical data, defaults to W for Cambridge.

    `historical_minority`
    : name of minority bloc in historical data, defaults to C for Cambridge.

    `path`
    :   file path to an election data file to sample from. Defaults to Cambridge elections.

    **Methods**

    See `BallotGenerator` base class.
    """

    def __init__(
        self,
        cohesion_parameters: dict,
        path: Optional[Path] = None,
        historical_majority: Optional[str] = "W",
        historical_minority: Optional[str] = "C",
        **data,
    ):
        # Call the parent class's __init__ method to handle common parameters
        super().__init__(cohesion_parameters=cohesion_parameters, **data)

        self.historical_majority = historical_majority
        self.historical_minority = historical_minority

        if len(self.slate_to_candidates.keys()) > 2:
            raise UserWarning(
                f"This model currently only supports at two blocs, but you \
                              passed {len(self.slate_to_candidates.keys())}"
            )

        self.majority_bloc = [
            bloc for bloc, prop in self.bloc_voter_prop.items() if prop >= 0.5
        ][0]

        self.minority_bloc = [
            bloc for bloc in self.bloc_voter_prop.keys() if bloc != self.majority_bloc
        ][0]

        self.bloc_to_historical = {
            self.majority_bloc: self.historical_majority,
            self.minority_bloc: self.historical_minority,
        }

        # # changing names to match historical data, if statement handles generating from_params
        # # only want to run this now if generating from init
        # if len(self.cohesion_parameters) > 0:
        #     self._rename_blocs()

        if path:
            self.path = path
        else:
            BASE_DIR = Path(__file__).resolve().parent
            DATA_DIR = BASE_DIR / "data/"
            self.path = Path(DATA_DIR, "Cambridge_09to17_ballot_types.p")

    def generate_profile(
        self, number_of_ballots: int, by_bloc: bool = False
    ) -> Union[PreferenceProfile, Tuple]:
        with open(self.path, "rb") as pickle_file:
            ballot_frequencies = pickle.load(pickle_file)

        cohesion_parameters = {b: self.cohesion_parameters[b][b] for b in self.blocs}

        # compute the number of bloc and crossover voters in each bloc using Huntington Hill
        voter_types = [
            (b, t) for b in list(self.bloc_voter_prop.keys()) for t in ["bloc", "cross"]
        ]

        voter_props = [
            cohesion_parameters[b] * self.bloc_voter_prop[b]
            if t == "bloc"
            else (1 - cohesion_parameters[b]) * self.bloc_voter_prop[b]
            for b, t in voter_types
        ]

        ballots_per_type = dict(
            zip(
                voter_types,
                apportion.compute("huntington", voter_props, number_of_ballots),
            )
        )

        pp_by_bloc = {b: PreferenceProfile() for b in self.blocs}

        for i, bloc in enumerate(self.blocs):
            bloc_voters = ballots_per_type[(bloc, "bloc")]
            cross_voters = ballots_per_type[(bloc, "cross")]
            ballot_pool = [Ballot()] * (bloc_voters + cross_voters)

            # store the opposition bloc
            opp_bloc = self.blocs[(i + 1) % 2]

            # find total number of ballots that start with bloc and opp_bloc
            bloc_first_count = sum(
                [
                    freq
                    for ballot, freq in ballot_frequencies.items()
                    if ballot[0] == self.bloc_to_historical[bloc]
                ]
            )

            opp_bloc_first_count = sum(
                [
                    freq
                    for ballot, freq in ballot_frequencies.items()
                    if ballot[0] == self.bloc_to_historical[opp_bloc]
                ]
            )

            # Compute the pref interval for this bloc
            pref_interval_dict = combine_preference_intervals(
                list(self.pref_intervals_by_bloc[bloc].values()),
                [cohesion_parameters[bloc], 1 - cohesion_parameters[bloc]],
            )

            # compute the relative probabilities of each ballot
            # sorted by ones where the ballot lists the bloc first
            # and those that list the opp first
            prob_ballot_given_bloc_first = {
                ballot: freq / bloc_first_count
                for ballot, freq in ballot_frequencies.items()
                if ballot[0] == self.bloc_to_historical[bloc]
            }

            prob_ballot_given_opp_first = {
                ballot: freq / opp_bloc_first_count
                for ballot, freq in ballot_frequencies.items()
                if ballot[0] == self.bloc_to_historical[opp_bloc]
            }

            bloc_voter_ordering = random.choices(
                list(prob_ballot_given_bloc_first.keys()),
                weights=list(prob_ballot_given_bloc_first.values()),
                k=bloc_voters,
            )
            cross_voter_ordering = random.choices(
                list(prob_ballot_given_opp_first.keys()),
                weights=list(prob_ballot_given_opp_first.values()),
                k=cross_voters,
            )

            # Generate ballots
            for i in range(bloc_voters + cross_voters):
                # Based on first choice, randomly choose
                # ballots weighted by Cambridge frequency
                if i < bloc_voters:
                    bloc_ordering = bloc_voter_ordering[i]
                else:
                    bloc_ordering = cross_voter_ordering[i - bloc_voters]

                # Now turn bloc ordering into candidate ordering
                pl_ordering = list(
                    np.random.choice(
                        list(pref_interval_dict.interval.keys()),
                        len(pref_interval_dict.interval),
                        p=list(pref_interval_dict.interval.values()),
                        replace=False,
                    )
                )
                ordered_bloc_slate = [
                    c for c in pl_ordering if c in self.slate_to_candidates[bloc]
                ]
                ordered_opp_slate = [
                    c for c in pl_ordering if c in self.slate_to_candidates[opp_bloc]
                ]

                # Fill in the bloc slots as determined
                # With the candidate ordering generated with PL
                full_ballot = []
                for b in bloc_ordering:
                    if b == self.bloc_to_historical[bloc]:
                        if ordered_bloc_slate:
                            full_ballot.append(ordered_bloc_slate.pop(0))
                    else:
                        if ordered_opp_slate:
                            full_ballot.append(ordered_opp_slate.pop(0))

                ranking = tuple([frozenset({cand}) for cand in full_ballot])
                ballot_pool[i] = Ballot(ranking=ranking, weight=Fraction(1, 1))

            pp = PreferenceProfile(ballots=ballot_pool)
            pp = pp.condense_ballots()
            pp_by_bloc[bloc] = pp

        # combine the profiles
        pp = PreferenceProfile(ballots=[])
        for profile in pp_by_bloc.values():
            pp += profile

        if by_bloc:
            return (pp_by_bloc, pp)

        # else return the combined profiles
        else:
            return pp


class name_Cumulative(BallotGenerator):
    """
    Class for generating cumulative ballots. This model samples with
    replacement from a combined preference interval and counts candidates with multiplicity.
    Can be initialized with an interval or can be constructed with the Dirichlet distribution
    using the `from_params` method in the `BallotGenerator` class.

    **Attributes**

    `candidates`
    : a list of candidates.

    `pref_intervals_by_bloc`
    :   dictionary of dictionaries mapping of bloc to preference intervals.
        (ex. {bloc_1: {bloc_1 : PI, bloc_2: PI}}).

    `cohesion_parameters`
    : dictionary of dictionaries of cohesion parameters (ex. {bloc_1: {bloc_1:.7, bloc_2: .3}})

    `bloc_voter_prop`
    :   dictionary mapping of bloc to voter proportions (ex. {bloc: proportion}).

    `num_votes`
    : the number of votes allowed per ballot.

    **Methods**

    See `BallotGenerator` base class
    """

    def __init__(self, cohesion_parameters: dict, num_votes: int, **data):
        # Call the parent class's __init__ method to handle common parameters
        super().__init__(cohesion_parameters=cohesion_parameters, **data)
        self.num_votes = num_votes

        # if dictionary of pref intervals is passed
        if isinstance(
            list(self.pref_intervals_by_bloc.values())[0], PreferenceInterval
        ):
            self.pref_interval_by_bloc = self.pref_intervals_by_bloc

        # if nested dictionary of pref intervals, combine by cohesion
        else:
            self.pref_interval_by_bloc = {
                bloc: combine_preference_intervals(
                    [self.pref_intervals_by_bloc[bloc][b] for b in self.blocs],
                    [self.cohesion_parameters[bloc][b] for b in self.blocs],
                )
                for bloc in self.blocs
            }

    def generate_profile(
        self, number_of_ballots: int, by_bloc: bool = False
    ) -> Union[PreferenceProfile, Tuple]:
        """
        Args:
        `number_of_ballots`: The number of ballots to generate.

        `by_bloc`: True if you want to return a dictionary of PreferenceProfiles by bloc.
                    False if you want the full, aggregated PreferenceProfile.
        """
        # the number of ballots per bloc is determined by Huntington-Hill apportionment
        bloc_props = list(self.bloc_voter_prop.values())
        ballots_per_block = dict(
            zip(
                self.blocs,
                apportion.compute("huntington", bloc_props, number_of_ballots),
            )
        )

        pp_by_bloc = {b: PreferenceProfile() for b in self.blocs}

        for bloc in self.bloc_voter_prop.keys():
            ballot_pool = []
            # number of voters in this bloc
            num_ballots = ballots_per_block[bloc]
            pref_interval = self.pref_interval_by_bloc[bloc]

            # finds candidates with non-zero preference
            non_zero_cands = list(pref_interval.non_zero_cands)
            # creates the interval of probabilities for candidates supported by this block
            cand_support_vec = [pref_interval.interval[cand] for cand in non_zero_cands]

            for _ in range(num_ballots):
                # generates ranking based on probability distribution of non zero candidate support
                list_ranking = list(
                    np.random.choice(
                        non_zero_cands,
                        self.num_votes,
                        p=cand_support_vec,
                        replace=True,
                    )
                )

                ranking = tuple([frozenset({cand}) for cand in list_ranking])

                ballot_pool.append(Ballot(ranking=ranking, weight=Fraction(1, 1)))

            pp = PreferenceProfile(ballots=ballot_pool)
            pp = pp.condense_ballots()
            pp_by_bloc[bloc] = pp

        # combine the profiles
        pp = PreferenceProfile(ballots=[])
        for profile in pp_by_bloc.values():
            pp += profile

        if by_bloc:
            return (pp_by_bloc, pp)

        # else return the combined profiles
        else:
            return pp


class slate_PlackettLuce(BallotGenerator):
    """
    Class for generating ballots using a slate-PlackettLuce model.
    This model first samples a ballot type by flipping a cohesion parameter weighted coin.
    It then fills out the ballot type via sampling with out replacement from the interval.

    Can be initialized with an interval or can be
    constructed with the Dirichlet distribution using the `from_params` method in the
    `BallotGenerator` class.

    **Attributes**

    `slate_to_candidates`
    :   dictionary mapping of slate to candidates (ex. {bloc: [candidate]}).

    `pref_intervals_by_bloc`
    :   dictionary of dictionaries mapping of bloc to preference intervals.
        (ex. {bloc_1: {bloc_1 : PI, bloc_2: PI}}).

    `bloc_voter_prop`
    :   dictionary mapping of bloc to voter proportions (ex. {bloc: proportion}).

    `cohesion_parameters`
    : dictionary of dictionaries of cohesion parameters (ex. {bloc_1: {bloc_1:.7, bloc_2: .3}})

    **Methods**

    See `BallotGenerator` base class
    """

    def __init__(self, cohesion_parameters: dict, **data):
        # Call the parent class's __init__ method to handle common parameters
        super().__init__(cohesion_parameters=cohesion_parameters, **data)

    def generate_profile(
        self, number_of_ballots: int, by_bloc: bool = False
    ) -> Union[PreferenceProfile, Tuple]:
        """
        Args:
        `number_of_ballots`: The number of ballots to generate.

        `by_bloc`: True if you want to return a dictionary of PreferenceProfiles by bloc.
                    False if you want the full, aggregated PreferenceProfile.
        """
        bloc_props = list(self.bloc_voter_prop.values())
        ballots_per_block = dict(
            zip(
                self.blocs,
                apportion.compute("huntington", bloc_props, number_of_ballots),
            )
        )

        pref_profile_by_bloc = {}

        for i, bloc in enumerate(self.blocs):
            # number of voters in this bloc
            num_ballots = ballots_per_block[bloc]
            ballot_pool = [Ballot()] * num_ballots
            pref_intervals = self.pref_intervals_by_bloc[bloc]
            zero_cands = set(
                it.chain(*[pi.zero_cands for pi in pref_intervals.values()])
            )

            slate_to_non_zero_candidates = {
                s: [c for c in c_list if c not in zero_cands]
                for s, c_list in self.slate_to_candidates.items()
            }

            ballot_types = sample_cohesion_ballot_types(
                slate_to_non_zero_candidates=slate_to_non_zero_candidates,
                num_ballots=num_ballots,
                cohesion_parameters_for_bloc=self.cohesion_parameters[bloc],
            )

            for j, bt in enumerate(ballot_types):
                cand_ordering_by_bloc = {}

                for b in self.blocs:
                    # create a pref interval dict of only this blocs candidates
                    bloc_cand_pref_interval = pref_intervals[b].interval
                    cands = pref_intervals[b].non_zero_cands

                    # if there are no non-zero candidates, skip this bloc
                    if len(cands) == 0:
                        continue

                    distribution = [bloc_cand_pref_interval[c] for c in cands]

                    # sample
                    cand_ordering = np.random.choice(
                        a=list(cands), size=len(cands), p=distribution, replace=False
                    )
                    cand_ordering_by_bloc[b] = list(cand_ordering)

                ranking = [frozenset({-1})] * len(bt)
                for i, b in enumerate(bt):
                    # append the current first candidate, then remove them from the ordering
                    ranking[i] = frozenset({cand_ordering_by_bloc[b][0]})
                    cand_ordering_by_bloc[b].pop(0)

                if len(zero_cands) > 0:
                    ranking.append(frozenset(zero_cands))
                ballot_pool[j] = Ballot(ranking=tuple(ranking), weight=Fraction(1, 1))

            pp = PreferenceProfile(ballots=ballot_pool)
            pp = pp.condense_ballots()
            pref_profile_by_bloc[bloc] = pp

        # combine the profiles
        pp = PreferenceProfile(ballots=[])
        for profile in pref_profile_by_bloc.values():
            pp += profile

        if by_bloc:
            return (pref_profile_by_bloc, pp)

        # else return the combined profiles
        else:
            return pp


class slate_BradleyTerry(BallotGenerator):
    """
    Class for generating ballots using a slate-BradleyTerry model. It
    presamples ballot types by checking all pairwise comparisons, then fills out candidate
    ordering by sampling without replacement from preference intervals.

    Only works with 2 blocs at the moment.

    Can be initialized with an interval or can be
    constructed with the Dirichlet distribution using the `from_params` method in the
    `BallotGenerator` class.

    **Attributes**

    `slate_to_candidates`
    :   dictionary mapping of slate to candidates (ex. {bloc: [candidate]}).

    `pref_intervals_by_bloc`
    :   dictionary of dictionaries mapping of bloc to preference intervals.
        (ex. {bloc_1: {bloc_1 : PI, bloc_2: PI}}).

    `bloc_voter_prop`
    :   dictionary mapping of bloc to voter proportions (ex. {bloc: proportion}).

    `cohesion_parameters`
    : dictionary of dictionaries of cohesion parameters (ex. {bloc_1: {bloc_1:.7, bloc_2: .3}})

    **Methods**

    See `BallotGenerator` base class
    """

    def __init__(self, cohesion_parameters: dict, **data):
        # Call the parent class's __init__ method to handle common parameters
        super().__init__(cohesion_parameters=cohesion_parameters, **data)

        if len(self.slate_to_candidates.keys()) > 2:
            raise UserWarning(
                f"This model currently only supports at most two blocs, but you \
                              passed {len(self.slate_to_candidates.keys())}"
            )

        self.ballot_type_pdf = {
            b: self._compute_ballot_type_dist(b, self.blocs[(i + 1) % 2])
            for i, b in enumerate(self.blocs)
        }

    def _compute_ballot_type_dist(self, bloc, opp_bloc):
        """
        Return a dictionary with keys ballot types and values equal to probability of sampling.
        """
        blocs_to_sample = [
            b for b in self.blocs for _ in range(len(self.slate_to_candidates[b]))
        ]
        total_comparisons = np.prod(
            [len(l_of_c) for l_of_c in self.slate_to_candidates.values()]
        )
        cohesion = self.cohesion_parameters[bloc][bloc]

        def prob_of_type(b_type):
            success = sum(
                b_type[i + 1 :].count(opp_bloc)
                for i, b in enumerate(b_type)
                if b == bloc
            )
            return pow(cohesion, success) * pow(
                1 - cohesion, total_comparisons - success
            )

        pdf = {
            b: prob_of_type(b)
            for b in set(it.permutations(blocs_to_sample, len(blocs_to_sample)))
        }

        summ = sum(pdf.values())
        return {b: v / summ for b, v in pdf.items()}

    def _sample_ballot_types_deterministic(
        self, bloc: str, opp_bloc: str, num_ballots: int
    ):
        """
        Used to generate bloc orderings for deliberative.

        Returns a list of lists, where each sublist contains the bloc names in order they appear
        on the ballot.
        """
        # pdf = self._compute_ballot_type_dist(bloc=bloc, opp_bloc=opp_bloc)
        pdf = self.ballot_type_pdf[bloc]
        b_types = list(pdf.keys())
        probs = list(pdf.values())

        sampled_indices = np.random.choice(len(b_types), size=num_ballots, p=probs)

        return [b_types[i] for i in sampled_indices]

    def _sample_ballot_types_MCMC(
        self, bloc: str, num_ballots: int, verbose: bool = False
    ):
        """
        Generate ballot types using MCMC that has desired stationary distribution.
        """

        seed_ballot_type = [
            b for b in self.blocs for _ in range(len(self.slate_to_candidates[b]))
        ]

        ballots = [[-1]] * num_ballots
        accept = 0
        current_ranking = seed_ballot_type

        cohesion = self.cohesion_parameters[bloc][bloc]

        # presample swap indices
        swap_indices = [
            (j1, j1 + 1)
            for j1 in np.random.choice(len(seed_ballot_type) - 1, size=num_ballots)
        ]

        odds = (1 - cohesion) / cohesion
        # generate MCMC sample
        for i in range(num_ballots):
            # choose adjacent pair to propose a swap
            j1, j2 = swap_indices[i]

            # if swap reduces number of voters bloc above opposing bloc
            if (
                current_ranking[j1] != current_ranking[j2]
                and current_ranking[j1] == bloc
            ):
                acceptance_prob = odds

            # if swap increases number of voters bloc above opposing or swaps two of same bloc
            else:
                acceptance_prob = 1

            # if you accept, make the swap
            if random.random() < acceptance_prob:
                current_ranking[j1], current_ranking[j2] = (
                    current_ranking[j2],
                    current_ranking[j1],
                )
                accept += 1

            ballots[i] = current_ranking.copy()

        if verbose:
            print(
                f"Acceptance ratio as number accepted / total steps: {accept/num_ballots:.2}"
            )

        if -1 in ballots:
            raise ValueError("Some element of ballots list is not a ballot.")

        return ballots

    def generate_profile(
        self, number_of_ballots: int, by_bloc: bool = False, deterministic: bool = True
    ) -> Union[PreferenceProfile, Tuple]:
        """
        Args:
        `number_of_ballots`: The number of ballots to generate.

        `by_bloc`: True if you want to return a dictionary of PreferenceProfiles by bloc.
                    False if you want the full, aggregated PreferenceProfile.

        `deterministic`: True if you want to use the computed pdf for the slate-BT model,
                        False if you want to use MCMC approximation. Defaults to True.
        """
        # the number of ballots per bloc is determined by Huntington-Hill apportionment
        bloc_props = list(self.bloc_voter_prop.values())
        ballots_per_block = dict(
            zip(
                self.blocs,
                apportion.compute("huntington", bloc_props, number_of_ballots),
            )
        )

        pref_profile_by_bloc = {}

        for i, bloc in enumerate(self.blocs):
            # number of voters in this bloc
            num_ballots = ballots_per_block[bloc]
            ballot_pool = [Ballot()] * num_ballots
            pref_intervals = self.pref_intervals_by_bloc[bloc]
            zero_cands = set(
                it.chain(*[pi.zero_cands for pi in pref_intervals.values()])
            )

            if deterministic:
                ballot_types = self._sample_ballot_types_deterministic(
                    bloc=bloc, opp_bloc=self.blocs[(i + 1) % 2], num_ballots=num_ballots
                )
            else:
                ballot_types = self._sample_ballot_types_MCMC(
                    bloc=bloc, num_ballots=num_ballots
                )

            for j, bt in enumerate(ballot_types):
                cand_ordering_by_bloc = {}

                for b in self.blocs:
                    # create a pref interval dict of only this blocs candidates
                    bloc_cand_pref_interval = pref_intervals[b].interval
                    cands = pref_intervals[b].non_zero_cands

                    # if there are no non-zero candidates, skip this bloc
                    if len(cands) == 0:
                        continue

                    distribution = [bloc_cand_pref_interval[c] for c in cands]

                    # sample
                    cand_ordering = np.random.choice(
                        a=list(cands), size=len(cands), p=distribution, replace=False
                    )

                    cand_ordering_by_bloc[b] = list(cand_ordering)

                ranking = [frozenset({-1})] * len(bt)
                for i, b in enumerate(bt):
                    # append the current first candidate, then remove them from the ordering
                    ranking[i] = frozenset({cand_ordering_by_bloc[b][0]})
                    cand_ordering_by_bloc[b].pop(0)

                if len(zero_cands) > 0:
                    ranking.append(frozenset(zero_cands))
                ballot_pool[j] = Ballot(ranking=tuple(ranking), weight=Fraction(1, 1))

            pp = PreferenceProfile(ballots=ballot_pool)
            pp = pp.condense_ballots()
            pref_profile_by_bloc[bloc] = pp

        # combine the profiles
        pp = PreferenceProfile(ballots=[])
        for profile in pref_profile_by_bloc.values():
            pp += profile

        if by_bloc:
            return (pref_profile_by_bloc, pp)

        # else return the combined profiles
        else:
            return pp
