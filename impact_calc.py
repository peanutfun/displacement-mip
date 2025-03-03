"""Module for impact calculation and calibration on multiple exposures"""

from dataclasses import dataclass, InitVar
from numbers import Number
from typing import Optional
from pathlib import Path
import re
import logging

import numpy as np
import xarray as xr
import pandas as pd
from scipy import sparse
from scipy.optimize import NonlinearConstraint
from sklearn.metrics import mean_squared_log_error

from climada.engine import ImpactCalc, Impact
from climada.entity import Exposures, ImpactFuncSet, ImpactFunc
from climada.hazard import Hazard
from climada.util.calibrate import BayesianOptimizer, Input
from climada.util.coordinates import country_to_iso
from climada.util.config import CONFIG

from climada_petals.hazard.rf_glofas import hazard_series_from_dataset

YEAR_RANGE_DEFAULT = np.arange(2008, 2023 + 1)
DISPLACEMENT_DATA_PATH = (
    Path(__file__).parent / "data/IDMC_GIDD_Disasters_Internal_Displacement_Data.xlsx"
)

LOGGER = logging.getLogger("climada.engine.impact_calc")


class PersistingImpactCalc(ImpactCalc):
    """Persist data useful for multiple impact calculations on the same objects"""

    def __init__(
        self,
        exposures,
        impfset,
        hazard,
        assign_centroids=True,
        ignore_cover=False,
        ignore_deductible=False,
    ):
        """Initialze object

        Parameters
        ----------
        assign_centroids : bool, optional
            indicates whether centroids are assigned to the self.exposures object.
            Centroids assignment is an expensive operation; set this to ``False`` to save
            computation time if the hazards' centroids are already assigned to the exposures
            object.
            Default: True
        ignore_cover : bool, optional
            if set to True, the column 'cover' of the exposures GeoDataFrame, if present, is
            ignored and the impact it not capped by the values in this column.
            Default: False
        ignore_deductible : bool, opotional
            if set to True, the column 'deductible' of the exposures GeoDataFrame, if present, is
            ignored and the impact it not reduced through values in this column.
            Default: False
        """
        self._exposures = exposures
        self._impfset = impfset
        self._hazard = hazard
        self.ignore_cover = ignore_cover
        self.ignore_deductible = ignore_deductible
        self._orig_exp_idx = np.arange(self._exposures.gdf.shape[0])

        # Check if centroids are assigned
        self.check_hazard_exposure_compatibility()
        if assign_centroids or (
            self.hazard.centr_exp_col not in self.exposures.gdf.columns
        ):
            self.exposures.assign_centroids(hazard, overwrite=True)

        self._prepare()

    def _prepare(
        self,
    ):
        self._impf_col = self.exposures.get_impf_column(self.hazard.haz_type)
        self._exp_gdf = self.minimal_exp_gdf(
            self._impf_col,
            assign_centroids=False,
            ignore_cover=self.ignore_cover,
            ignore_deductible=self.ignore_deductible,
        )
        self._exposure_idx_chunks = {
            impf_id: self._chunk_exp_idx(
                self.hazard.size,
                (self._exp_gdf[self._impf_col].to_numpy() == impf_id).nonzero()[0],
            )
            for impf_id in self._exp_gdf[self._impf_col].dropna().unique()
        }
        self._centroids_idx_chunks = {
            impf_id: [
                np.unique(
                    self._exp_gdf[self.hazard.centr_exp_col].to_numpy()[idx],
                    return_inverse=True,
                )
                for idx in idx_chunks
            ]
            for impf_id, idx_chunks in self._exposure_idx_chunks.items()
        }
        # for key, val in self._centroids_idx_chunks.items():
        #     print(key, val)

    @property
    def exposures(self):
        return self._exposures

    @exposures.setter
    def exposures(self, new):
        self._exposures = new
        self._prepare()
        self.check_hazard_exposure_compatibility()

    @property
    def hazard(self):
        return self._hazard

    @hazard.setter
    def hazard(self, new):
        self._hazard = new
        self._prepare()
        self.check_hazard_exposure_compatibility()

    @property
    def impfset(self):
        return self._impfset

    @impfset.setter
    def impfset(self, new):
        self._impfset = new
        self.check_hazard_exposure_compatibility()

    def _chunk_exp_idx(self, haz_size, idx_exp_impf):
        """
        Chunk computations in sizes that roughly fit into memory
        """
        max_size = CONFIG.max_matrix_size.int()
        if haz_size > max_size:
            raise ValueError(
                f"Hazard size '{haz_size}' exceeds maximum matrix size '{max_size}'. "
                "Increase max_matrix_size configuration parameter accordingly."
            )
        n_chunks = np.ceil(haz_size * len(idx_exp_impf) / max_size)
        return np.array_split(idx_exp_impf, n_chunks)

    def check_hazard_exposure_compatibility(self):
        """"""
        # check for compability of exposures and hazard type
        if all(
            name not in self.exposures.gdf.columns
            for name in [
                "if_",
                f"if_{self.hazard.haz_type}",
                "impf_",
                f"impf_{self.hazard.haz_type}",
            ]
        ):
            raise AttributeError(
                "Impact calculation not possible. No impact functions found "
                f"for hazard type {self.hazard.haz_type} in exposures."
            )

        # check for compability of impact function and hazard type
        # if not self.impfset.get_func(haz_type=self.hazard.haz_type):
        #     raise AttributeError(
        #         "Impact calculation not possible. No impact functions found "
        #         f"for hazard type {self.hazard.haz_type} in impf_set."
        #     )

    def impact(self, save_mat=True):
        """Compute the impact of a hazard on exposures.

        Parameters
        ----------
        save_mat : bool, optional
            if true, save the total impact matrix (events x exposures)
            Default: True

        Examples
        --------
            >>> haz = Hazard.from_mat(HAZ_DEMO_MAT)  # Set hazard
            >>> impfset = ImpactFuncSet.from_excel(ENT_TEMPLATE_XLS)
            >>> exp = Exposures(pd.read_excel(ENT_TEMPLATE_XLS))
            >>> impcalc = ImpactCal(exp, impfset, haz)
            >>> imp = impcalc.impact(insured=True)
            >>> imp.aai_agg

        See also
        --------
        apply_deductible_to_mat : apply deductible to impact matrix
        apply_cover_to_mat : apply cover to impact matrix
        """
        if self._exp_gdf.size == 0:
            return self._return_empty(save_mat)

        LOGGER.info(
            "Calculating impact for %s assets (>0) and %s events.",
            self._exp_gdf.size,
            self.n_events,
        )
        imp_mat_gen = self._imp_mat_gen()
        insured = ("cover" in self._exp_gdf and self._exp_gdf.cover.max() >= 0) or (
            "deductible" in self._exp_gdf and self._exp_gdf.deductible.max() > 0
        )
        if insured:
            LOGGER.info(
                "cover and/or deductible columns detected,"
                " going to calculate insured impact"
            )
            # TODO: make a better impact matrix generator for insured impacts when
            # the impact matrix is already present
            imp_mat_gen = self.insured_mat_gen(imp_mat_gen, self._exp_gdf, impf_col)

        return self._return_impact(imp_mat_gen, save_mat)

    def _imp_mat_gen(self):
        """
        Generator of impact sub-matrices and correspoding exposures indices
        """
        for impf_id, exp_idx_chunks in self._exposure_idx_chunks.items():
            impf = self.impfset.get_func(haz_type=self.hazard.haz_type, fun_id=impf_id)
            # centroids_idx_chunks, centroids_idx_reverse_chunks = (
            #     self._centroids_idx_chunks[impf_id]
            # )
            for exp_idx, (centr_idx, centr_idx_rev) in zip(
                exp_idx_chunks, self._centroids_idx_chunks[impf_id]
            ):
                # print(exp_idx.shape)
                # print(centr_idx.shape)
                # print(centr_idx_rev.shape)
                exp_values = self._exp_gdf["value"].to_numpy()[exp_idx]
                yield (
                    self.impact_matrix(exp_values, centr_idx, centr_idx_rev, impf),
                    exp_idx,
                )

    def impact_matrix(self, exp_values, centroids_idx, centroids_idx_reverse, impf):
        """
        Compute the impact matrix for given exposure values,
        assigned centroids, a hazard, and one impact function.
        """
        mdr = self.get_mdr(centroids_idx, centroids_idx_reverse, impf)
        # exp_values_csr = sparse.csr_matrix([exp_values])
        # exp_values_csr = np.matrix([exp_values], copy=False)
        exp_values_csr = [exp_values]
        # print(exp_values_csr.shape, np.matrix([exp_values], copy=False).shape)
        fract = self.hazard._get_fraction(
            centroids_idx
        )  # pylint: disable=protected-access
        if fract is None:
            val = mdr.multiply(exp_values_csr)
        else:
            val = fract[:, centroids_idx_reverse].multiply(mdr).multiply(exp_values_csr)
        # val = val.tocsr(copy=True)
        # print(val.shape, val.nnz)
        val = val.tocsr()
        val.eliminate_zeros()
        return val

    def get_mdr(self, centroids_idx, centroids_idx_reverse, impf):
        """
        Return Mean Damage Ratio (mdr) for chosen centroids (cent_idx)
        for given impact function.
        """
        mdr = self.hazard.intensity[:, centroids_idx]
        if impf.calc_mdr(0) == 0:
            mdr.data = impf.calc_mdr(mdr.data)
        else:
            LOGGER.warning(
                "Impact function id=%d has mdr(0) != 0."
                "The mean damage ratio must thus be computed for all values of"
                "hazard intensity including 0 which can be very time consuming.",
                impf.id,
            )
            mdr = sparse.csr_matrix(impf.calc_mdr(mdr.toarray()))
        return mdr[:, centroids_idx_reverse]


@dataclass
class MultiExpImpactCalc:
    """Calculate an impact with multiple exposures"""

    exposures: dict[int, Exposures]
    impfset: ImpactFuncSet
    hazard: Hazard

    check: InitVar[bool] = False

    def __post_init__(self, check):
        """Check exposures mapping"""
        if check:
            if sorted(self.hazard.event_id.flat) != sorted(self.exposures.keys()):
                raise RuntimeError(
                    "Invalid mapping between hazard event_id and exposures"
                )
            # TODO: Test that exposures have same coordinates!

    # def impact(self, save_mat=True, assign_centroids=False):
    #     """Compute the impact"""
    #     impacts = [
    #         ImpactCalc(
    #             exposures=self.exposures[event_id],
    #             impfset=self.impfset,
    #             hazard=self.hazard.select(event_id=[event_id]),
    #         ).impact(save_mat=save_mat, assign_centroids=assign_centroids)
    #         for event_id in self.hazard.event_id.flat
    #     ]
    #     for impact in impacts:
    #         for attr in ("crs", "tot_value", "unit", "frequency_unit"):
    #             setattr(impact, attr, 0)
    #     return Impact.concat(impacts, reset_event_ids=False)

    def impact(self, save_mat=False, assign_centroids=False):
        """Compute the impact"""
        at_event = [
            ImpactCalc(
                exposures=self.exposures[event_id],
                impfset=self.impfset,
                hazard=self.hazard.select(event_id=[event_id]),
            )
            .impact(save_mat=save_mat, assign_centroids=assign_centroids)
            .at_event
            for event_id in self.hazard.event_id.flat
        ]
        at_event = np.concatenate(at_event)
        exp = self.exposures[self.hazard.event_id[0]]
        return Impact.from_eih(
            exposures=exp,
            hazard=self.hazard,
            at_event=at_event,
            eai_exp=np.zeros(len(exp.gdf)),
            aai_agg=0.0,
        )


@dataclass
class PersistingMultiExpBayesianOptimizer(BayesianOptimizer):
    """"""

    def __post_init__(self, random_state, allow_duplicate_points, bayes_opt_kwds):
        super().__post_init__(random_state, allow_duplicate_points, bayes_opt_kwds)
        self._impact_calc = {
            event_id: PersistingImpactCalc(
                exposures=exp,
                impfset=None,
                hazard=self.input.hazard.select(event_id=[event_id]),
                assign_centroids=False,
            )
            for event_id, exp in self.input.exposure.items()
        }

    def _opt_func(self, *args, **kwargs) -> Number:
        """Optimization with 'MultiExpImpactCalc'"""
        # Create the impact function set from a new parameter estimate
        params = self._kwargs_to_impact_func_creator(*args, **kwargs)
        impf_set = self.input.impact_func_creator(**params)
        for impact_calc in self._impact_calc.values():
            impact_calc.impfset = impf_set

        impact = [
            impact_calc.impact(save_mat=True)
            for impact_calc in self._impact_calc.values()
        ]
        for imp in impact:
            for attr in ("crs", "tot_value", "unit", "frequency_unit"):
                setattr(imp, attr, 0)
        impact = Impact.concat(impact, reset_event_ids=False)

        # Transform to DataFrame, align, and compute target function
        data_aligned, impact_df_aligned = self.input.impact_to_aligned_df(
            impact, fillna=0
        )
        return self._target_func(data_aligned, impact_df_aligned)


@dataclass
class MultiExpBayesianOptimizer(BayesianOptimizer):
    """A bayesian optimizer capable of handling multiple exposures"""

    # NOTE: Must have default value because it will be placed after parameters with
    #       default value. Issue might be circumvented in Python 3.10 with 'kw_only'.
    # exposures: Optional[dict[int, Exposures]] = None

    # def __post_init__(self, random_state, allow_duplicate_points, bayes_opt_kwds):
    #     """Check if exposures is given"""
    #     if self.exposures is None:
    #         raise ValueError("Must define a value for 'exposures'")
    #     return super().__post_init__(
    #         random_state, allow_duplicate_points, bayes_opt_kwds
    #     )

    def _opt_func(self, *args, **kwargs) -> Number:
        """Optimization with 'MultiExpImpactCalc'"""
        # Create the impact function set from a new parameter estimate
        params = self._kwargs_to_impact_func_creator(*args, **kwargs)
        impf_set = self.input.impact_func_creator(**params)

        # Compute the impact
        impact = MultiExpImpactCalc(
            exposures=self.input.exposure,
            impfset=impf_set,
            hazard=self.input.hazard,
            check=False,
        ).impact(**self.input.impact_calc_kwds)

        # Transform to DataFrame, align, and compute target function
        data_aligned, impact_df_aligned = self.input.impact_to_aligned_df(
            impact, fillna=0
        )
        return self._target_func(data_aligned, impact_df_aligned)


def load_worldpop(country: str, year: int) -> Exposures:
    """Load the WorldPop data"""
    country = country.lower()
    iso = country_to_iso(country, representation="alpha3").lower()
    if year not in np.arange(2008, 2020):
        year = 2020
    filepath = (
        Path(__file__).parent
        / f"data/worldpop/{country}/{iso}_ppp_{year}_1km_Aggregated_UNadj.tif"
    )

    # assert filepath.is_file()
    return Exposures.from_raster(filepath)


def load_worldpop_map(
    country: str, year_range: np.ndarray = None
) -> dict[int, Exposures]:
    """Load all WorldPop data for a certain range"""
    if year_range is None:
        year_range = YEAR_RANGE_DEFAULT

    return {year: load_worldpop(country, year) for year in year_range.flat}


def load_hazard(country: str, intensity: str) -> Hazard:
    """Load the flood footprints into a single hazard set"""
    country = country.lower()

    def add_year(ds: xr.Dataset):
        match = re.search(r"{0}-(\d{{4}}).nc".format(country), ds.encoding["source"])
        return ds.assign_coords(time=[int(match.group(1))])

    with xr.open_mfdataset(
        str(Path(__file__).parent / f"data/generated/{country}/hazard/*.nc"),
        chunks={},
        combine="nested",
        concat_dim="time",
        preprocess=add_year,
    ) as ds:
        hazard = hazard_series_from_dataset(ds, intensity=intensity, event_dim="time")
    hazard.event_id = YEAR_RANGE_DEFAULT
    hazard.frequency = np.ones_like(hazard.event_id) / hazard.size

    return hazard


def load_yearly_displacement_data(
    country: str, ignore_zero_impact: bool = False
) -> pd.DataFrame:
    """Load the displacement data for a specific country"""
    data = pd.read_excel(DISPLACEMENT_DATA_PATH)
    country = country_to_iso(country, representation="alpha3")
    data_country = data.loc[
        data["ISO3"] == country,
        ["Date of Event (start)", "Disaster Internal Displacements"],
    ]
    data_country = (
        data_country.set_index("Date of Event (start)")[
            "Disaster Internal Displacements"
        ]
        .resample("1y")
        .sum()
    )
    if ignore_zero_impact:
        data_country[data_country < 1] = np.nan
    data_country.index = pd.Index(data_country.index.year, name="year")
    data_country.name = country_to_iso(country, representation="numeric")
    return data_country.to_frame()


def sigmoid_impf(intensity, threshold, half_point, exponent, upper_limit, haz_type, id):
    """
    Create a sigmoid impact function based on given parameters.

    Parameters
    ----------
    intensity : numpy.ndarray
        Array of intensity values for the impact function.
    threshold : float
        Threshold value for the sigmoid function.
    half_point : float
        Half-point value for the sigmoid function.
    exponent : float
        Exponent value for the sigmoid function.
    upper_limit : float
        Upper limit value for the sigmoid function.
    haz_type : str
        Type of hazard for the impact function.
    id : int
        Identifier for the impact function.

    Returns
    -------
    ImpactFuncSet
        Set of impact functions containing the sigmoid impact function.
    """
    mdd = np.zeros_like(intensity)
    if threshold < half_point:
        mdd = (intensity - threshold) / (half_point - threshold)
    mdd[mdd < 0] = 0
    mdd = mdd**exponent / (1 + mdd**exponent)
    mdd = mdd * upper_limit

    return ImpactFuncSet(
        [
            ImpactFunc(
                haz_type=haz_type,
                id=id,
                intensity=intensity,
                mdd=mdd,
                paa=np.ones_like(intensity),
            )
        ]
    )


def calibration_input(country: str, intensity: str, function: str):
    """Load everything to run a calibration"""
    # Load stuff
    print("Loading hazard")
    hazard = load_hazard(country=country, intensity=intensity)
    print("Loading exposure")
    exposures_map = load_worldpop_map(country=country)
    print("Loading data")
    data_displacement = load_yearly_displacement_data(
        country=country, ignore_zero_impact=True
    )

    # Assign exposure
    print("Assigning centroids")
    exp = next(iter(exposures_map.values()))
    exp.assign_centroids(hazard)
    for exposure in exposures_map.values():
        exposure.gdf["centr_RF"] = exp.gdf["centr_RF"]
        exposure.gdf["impf_RF"] = 1
    region_id = country_to_iso(country, representation="numeric")

    # Set up impact function creator
    if function == "step":
        bounds = {"threshold": (0.01, 3), "ratio": (0.01, 1)}
        constraints = None
        impact_func_creator = lambda threshold, ratio: ImpactFuncSet(
            [
                ImpactFunc.from_step_impf(
                    (0, threshold, 100), mdd=(0, ratio), haz_type="RF"
                )
            ]
        )
    elif function == "sigmoid":
        # Calculate intensity
        range_increase_factor = 0.2
        step = 0.01  # 1 cm
        haz_min = np.min(hazard.intensity.data)
        haz_max = np.max(hazard.intensity.data)
        haz_range = haz_max - haz_min
        haz_min = max(0.0, haz_min - haz_range * range_increase_factor)
        haz_max = haz_max + haz_range * range_increase_factor
        num_steps = int((haz_max - haz_min) / step)
        hazard_intensity = np.linspace(haz_min, haz_max, num_steps)

        # Create impact function

        # Threshold, Half-point
        # bounds = {"threshold": (0.01, 2), "half_point": (0.01, 10)}
        constraints = NonlinearConstraint(
            lambda threshold, half_point, **_: threshold - half_point, lb=-np.inf, ub=0
        )

        # Half-point, upper limit
        bounds = {
            "threshold": (0.01, 2),
            "half_point": (0.01, 5),
            "upper_limit": (0.001, 1),
        }
        constraints = None
        impact_func_creator = lambda threshold, half_point, upper_limit: sigmoid_impf(
            intensity=hazard_intensity,
            threshold=threshold,
            half_point=half_point,
            upper_limit=upper_limit,
            exponent=3,
            haz_type="RF",
            id=1,
        )
    else:
        raise NotImplementedError(f"Function: {function}")

    # Set up input
    print("Creating input")
    input_calibration = Input(
        hazard=hazard,
        exposure=exposures_map,
        data=data_displacement,
        impact_func_creator=impact_func_creator,
        impact_to_dataframe=lambda imp: pd.DataFrame.from_records(
            {region_id: imp.at_event}, index=imp.event_id
        ),
        cost_func=mean_squared_log_error,
        bounds=bounds,
        constraints=constraints,
        impact_calc_kwds={"save_mat": False, "assign_centroids": False},
        assign_centroids=False,
    )
    return input_calibration
