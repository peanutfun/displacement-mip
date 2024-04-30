from itertools import product
import matplotlib.colors as mcolors
from multiprocessing.pool import Pool
import argparse

from climada.util import log_level
from climada.util.calibrate import BayesianOptimizerController

from impact_calc import calibration_input, PersistingMultiExpBayesianOptimizer


def calibrate(ctry, prot):
    intensity = "flood_depth" if prot == "no_protection" else "flood_depth_flopros"

    inp = calibration_input(ctry, intensity, "sigmoid")
    opt = PersistingMultiExpBayesianOptimizer(inp)
    controller = BayesianOptimizerController.from_input(inp, sampling_base=5)

    with log_level("ERROR", "climada.util.calibrate"):
        with log_level("ERROR", "climada.engine.impact"):
            res = opt.run(controller)
            res.to_hdf5(
                f"data/generated/{ctry}/calibration/sigmoid-pspace-{prot}.h5",
                mode="w",
            )
            ax = res.plot_p_space(
                x="half_point",
                y="upper_limit",
                norm=mcolors.LogNorm(),
            )
            ax.set_title(f"{ctry} - {prot}")
            ax.get_figure().savefig(
                f"data/generated/{ctry}/calibration/sigmoid-pspace-{prot}.pdf"
            )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_proc", "-n", default=1, type=int, help="Number of parallel processes"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    countries = ("ethiopia", "somalia", "sudan")
    protection = ("no_protection", "flopros")

    with Pool(processes=args.num_proc) as pool:
        pool.starmap(calibrate, product(countries, protection))


if __name__ == "__main__":
    main()
