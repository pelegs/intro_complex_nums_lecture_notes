import numpy as np

if __name__ == "__main__":
    # test stuff
    points = np.array(
        [
            1 + 0j,
            0 + 1j,
            -1 + 0j,
            0 - 1j,
        ]
    )
    cs = np.fft.ifft(points, n=40)
    print(cs)
