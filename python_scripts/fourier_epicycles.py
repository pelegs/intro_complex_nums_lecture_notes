import argparse

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
from scipy.integrate import quad_vec
from tqdm import tqdm


def set_args():
    parser = argparse.ArgumentParser(
        prog="Simple Fourier epicycle animation",
        description="Calculates and animates Fourier epicycles for a given path",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--num_circles",
        "-N",
        type=int,
        help="Number of Fourier circles to calculate",
        default=10,
    )
    parser.add_argument(
        "--num_frames",
        "-t",
        type=int,
        help="Number of time steps (and thus, frames) for animation (time is always in [0, 2pi))",
        default=250,
    )
    parser.add_argument(
        "--path-interval",
        "-k",
        type=int,
        help="Take only every k-th point from path",
        default=1,
    )
    parser.add_argument(
        "--filename",
        "-f",
        type=str,
        help="Filename of points data (npy format)",
        default="paths/rectangle.npy",
    )
    args = parser.parse_args()
    return args


def load_path(args):
    try:
        pts_spatial_complex = np.load(args.filename)[:: args.path_interval]
    except OSError as e:
        raise f"Can't load {args.filename}. Stated error: {e}"
    except Exception as e:
        raise f"Error: {e}"
    return pts_spatial_complex


def set_path(path_vertices):
    # center path (move average to origin)
    path_vertices -= np.mean(path_vertices)
    # calc max distance from path to origin, so that we could set the axes
    # such that it shows all of it with some space to spare
    max_distance = np.max(np.abs(path_vertices))
    return path_vertices, max_distance


def set_time(args, path_vertices):
    time_list = np.linspace(0, 2 * np.pi, path_vertices.shape[0])
    time_series = np.linspace(0, 2 * np.pi, args.num_frames)
    return time_list, time_series


def set_freqs(args):
    freqs = np.arange(-args.num_circles, args.num_circles + 1, 1)
    return freqs


def get_coeffs(args, freqs, time_list, path_vertices):
    # initialize coefficient array
    coeffs = np.zeros(2 * args.num_circles + 1, dtype=np.complex128)

    # create progress bar
    pbar = tqdm(freqs, desc="Calculating Fourier coefficients")

    # actual Fourier calculation (should be with np.fft.fft...
    # ...but it doesn't work ¯\_(ツ)_/¯)
    for i, k in enumerate(pbar):
        coeffs[i] = (
            1
            / (2 * np.pi)
            * quad_vec(
                lambda t: np.interp(
                    t,
                    time_list,
                    path_vertices,
                )
                * np.exp(-k * t * 1j),
                0,
                2 * np.pi,
                limit=100,
                full_output=1,
            )[0]
        )

    # sort coeffs by their norms (i.e. |z|)
    coeff_norms = np.abs(coeffs)
    coeffs_sorted_indices = np.flip(np.argsort(coeff_norms))
    coeffs_sorted = coeffs[coeffs_sorted_indices]
    return coeffs_sorted, coeffs_sorted_indices


def update_animation(step):
    plot_lines.set_data(np.real(circle_centers[step]), np.imag(circle_centers[step]))
    for circle, circle_patch in zip(circle_centers[step], circle_patches):
        circle_patch.center = [np.real(circle), np.imag(circle)]
    total_path.set_data(
        [np.real(circle_centers[:step, -1])], [np.imag(circle_centers[:step, -1])]
    )
    return [plot_lines, circle_patches, total_path]


def main():
    global plot_lines, circle_patches, total_path, circle_centers

    args = set_args()
    path_vertices = load_path(args)
    path_vertices, max_distance = set_path(path_vertices)
    time_list, time_series = set_time(args, path_vertices)
    freqs = set_freqs(args)
    coeffs, coeffs_sorted_indices = get_coeffs(args, freqs, time_list, path_vertices)
    freqs = freqs[coeffs_sorted_indices]

    circles = np.zeros(
        shape=(args.num_frames, 2 * args.num_circles + 1), dtype=np.complex128
    )
    for step, time in enumerate(time_series):
        circles[step] = coeffs * np.exp(freqs * 1j * time)
    circle_centers = np.cumsum(circles, axis=1)
    circle_centers = np.c_[np.zeros(args.num_frames), circle_centers]

    # Graphics
    fig, ax = plt.subplots()
    ax.set_title("Epicycles?")
    ax.set_xlabel("Real")
    ax.set_ylabel("Imaginary")
    ax.set_xlim(-1.5 * max_distance, 1.5 * max_distance)
    ax.set_ylim(-1.5 * max_distance, 1.5 * max_distance)
    ax.set_aspect("equal", "box")
    ax.grid()

    # ax.plot(np.real(pts_interpolated), np.imag(pts_interpolated))
    (plot_lines,) = ax.plot(
        np.real(circle_centers[0]), np.imag(circle_centers[0]), c="black"
    )
    (total_path,) = ax.plot(
        [np.real(circle_centers[0, -1])], [np.imag(circle_centers[0, -1])], c="red"
    )
    circle_patches = []
    for circle, coeff in zip(circle_centers[0], coeffs):
        circle_patch = Circle(
            xy=(np.real(circle), np.imag(circle)),
            radius=np.abs(coeff),
            fill=False,
            color="purple",
        )
        circle_patches.append(circle_patch)
        ax.add_patch(circle_patch)

    animation = FuncAnimation(
        fig=fig, func=update_animation, frames=args.num_frames, interval=0
    )
    plt.show()


if __name__ == "__main__":
    main()
