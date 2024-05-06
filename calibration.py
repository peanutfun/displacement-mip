from itertools import product, repeat
import matplotlib.colors as mcolors
from multiprocessing.pool import Pool
import argparse
from pathlib import Path

from climada.util import log_level
from climada.util.calibrate import BayesianOptimizerController

from impact_calc import calibration_input, PersistingMultiExpBayesianOptimizer


def calibrate(ctry, prot, output_dir):
    intensity = "flood_depth" if prot == "no_protection" else "flood_depth_flopros"

    inp = calibration_input(ctry, intensity, "sigmoid")
    opt = PersistingMultiExpBayesianOptimizer(inp)
    controller = BayesianOptimizerController.from_input(inp, sampling_base=4)
    
    output_path = Path(f"data/generated/{ctry}") / output_dir
    output_path.mkdir(parents=True, exist_ok=True)

    with log_level("ERROR", "climada.util.calibrate"):
        with log_level("ERROR", "climada.engine.impact"):
            res = opt.run(controller)
            res.to_hdf5(
                output_path / f"pspace-{prot}.h5",
                mode="w",
            )
            axes = res.plot_p_space(norm=mcolors.LogNorm())
            for idx, ax in enumerate(axes):
                ax.set_title(f"{ctry} - {prot}")
                ax.get_figure().savefig(
                    foutput_path / f"pspace-{prot}-{idx}.pdf"
                )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=str, help="Name of the output directory (in country tree)")
    parser.add_argument(
        "--num_proc", "-n", default=1, type=int, help="Number of parallel processes"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    countries = ("ethiopia", "somalia", "sudan")
    protection = ("no_protection", "flopros")
    
    params = list(product(countries, protection))
    params = [tpl + (args.output_dir, ) for tpl in params]

    with Pool(processes=args.num_proc) as pool:
        pool.starmap(calibrate, params)


if __name__ == "__main__":
    main()
