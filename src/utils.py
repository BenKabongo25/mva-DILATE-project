# Machine Learning for Times Series
# DILATE Project
#
# Ben Kabongo & Martin Brosset
# M2 MVA


import numpy as np


def generate_synthetic_dataset(n_samples: int, n_dim: int, input_length: int, output_length: int, 
                              scale: float=1, n_breakpoints: int=1):
    """
    Parameters
        n_samples: number of samples
        n_dim: number of dimension
        input_length: input length
        output_length: output length
        scale: scale
        n_breakpoints: number of breakpoints to introduce in each signal
    Returns
        input: input signals
        output: signals to predicts
    """
    signal_length = input_length + output_length
    signals = []

    for _ in range(n_samples):
        signal = scale * np.random.random(size=(signal_length, n_dim))
        if n_breakpoints > 0:
            a = np.random.randint(1, signal_length//2, size=n_breakpoints)
            b = np.random.randint(signal_length//2, signal_length, size=n_breakpoints)
            i = np.random.random(size=n_dim)
            j = np.random.random(size=n_dim)
            for (ak, bk) in zip(a, b):
                signal[ak:bk] += j - i
        signals.append(signal)
    
    signals = np.stack(signals)
    input_signals = signals[:, :input_length]
    output_signals = signals[:, input_length:]
    return input_signals, output_signals


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    plt.style.use("bmh")

    n_samples = 2
    input_length = 50
    output_length = 50

    n_dim = 1

    input_signals, output_signals = generate_synthetic_dataset(n_samples, n_dim, input_length, output_length, scale=0.01, n_breakpoints=2)

    plt.figure(figsize=(5 * n_samples, 4))
    for i in range(n_samples):
        plt.subplot(1, n_samples, i + 1)
        plt.plot(np.arange(0, input_length + 1), np.vstack([input_signals[i], output_signals[i][0]]), label="input")
        plt.plot(np.arange(input_length, input_length + output_length), output_signals[i], label="output")
        plt.title(f"Signal {i + 1}")
        plt.legend()
    plt.show()