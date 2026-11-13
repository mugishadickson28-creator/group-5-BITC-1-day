"""A from-scratch XOR network illustrating differential, integral, and vector calculus."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]

X: FloatArray = np.array(
    [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]],
    dtype=np.float64,
)
Y: FloatArray = np.array([[0.0], [1.0], [1.0], [0.0]], dtype=np.float64)


def sigmoid(z: FloatArray) -> FloatArray:
    """Logistic activation, whose derivative is sigmoid(z) * (1 - sigmoid(z))."""
    z = np.clip(z, -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(-z))


def initialize_parameters(seed: int) -> dict[str, FloatArray]:
    """Return reproducible weights for a 2 -> 4 -> 1 network."""
    rng = np.random.default_rng(seed)
    return {
        "W1": rng.normal(0.0, 0.8, size=(2, 4)),
        "b1": np.zeros((1, 4), dtype=np.float64),
        "W2": rng.normal(0.0, 0.8, size=(4, 1)),
        "b2": np.zeros((1, 1), dtype=np.float64),
    }


def forward(
    x: FloatArray, parameters: dict[str, FloatArray]
) -> tuple[FloatArray, tuple[FloatArray, FloatArray, FloatArray]]:
    """Compute predictions and cache activations needed by the chain rule."""
    z1 = x @ parameters["W1"] + parameters["b1"]
    a1 = np.tanh(z1)
    z2 = a1 @ parameters["W2"] + parameters["b2"]
    a2 = sigmoid(z2)
    return a2, (z1, a1, z2)


def binary_cross_entropy(y: FloatArray, prediction: FloatArray) -> float:
    """Mean binary cross-entropy, the empirical average over the four examples."""
    epsilon = 1e-12
    prediction = np.clip(prediction, epsilon, 1.0 - epsilon)
    return float(-np.mean(y * np.log(prediction) + (1.0 - y) * np.log(1.0 - prediction)))


def gradients(
    x: FloatArray,
    y: FloatArray,
    parameters: dict[str, FloatArray],
) -> dict[str, FloatArray]:
    """Backpropagate the mean loss, using matrix calculus for all parameters."""
    prediction, (_, a1, _) = forward(x, parameters)
    sample_count = x.shape[0]

    # BCE gives dL/dA2 = (A2 - Y) / (A2 * (1 - A2) * m), while
    # sigmoid gives dA2/dZ2 = A2 * (1 - A2); the chain-rule product simplifies.
    dz2 = (prediction - y) / sample_count
    dw2 = a1.T @ dz2
    db2 = np.sum(dz2, axis=0, keepdims=True)

    # Continue the chain: dL/dA1 = dL/dZ2 W2^T; tanh'(Z1) = 1 - A1^2.
    da1 = dz2 @ parameters["W2"].T
    dz1 = da1 * (1.0 - a1**2)
    dw1 = x.T @ dz1
    db1 = np.sum(dz1, axis=0, keepdims=True)
    return {"W1": dw1, "b1": db1, "W2": dw2, "b2": db2}


def train(
    initial_parameters: dict[str, FloatArray],
    *,
    learning_rate: float,
    epochs: int,
    momentum: float = 0.0,
) -> tuple[dict[str, FloatArray], list[float]]:
    """Full-batch gradient descent; momentum accumulates a decaying gradient sum."""
    parameters = {name: value.copy() for name, value in initial_parameters.items()}
    velocity = {name: np.zeros_like(value) for name, value in parameters.items()}
    history: list[float] = []

    for _ in range(epochs):
        prediction, _ = forward(X, parameters)
        history.append(binary_cross_entropy(Y, prediction))
        current_gradients = gradients(X, Y, parameters)

        for name in parameters:
            # v_t = beta*v_(t-1) + grad_t is a discrete, exponentially weighted
            # accumulation (an integral-like memory); theta moves by -eta*v_t.
            velocity[name] = momentum * velocity[name] + current_gradients[name]
            parameters[name] -= learning_rate * velocity[name]

    prediction, _ = forward(X, parameters)
    history.append(binary_cross_entropy(Y, prediction))
    return parameters, history


def make_figure(
    standard_history: list[float],
    momentum_history: list[float],
    parameters: dict[str, FloatArray],
    output_path: Path,
) -> None:
    """Plot loss comparison and the learned XOR decision boundary side by side."""
    figure, (loss_axis, boundary_axis) = plt.subplots(1, 2, figsize=(12, 5))

    loss_axis.plot(standard_history, label="Standard gradient descent", linewidth=2)
    loss_axis.plot(momentum_history, label="Gradient descent + momentum", linewidth=2)
    loss_axis.set(title="Learning speed", xlabel="Epoch", ylabel="Mean binary cross-entropy")
    loss_axis.set_yscale("log")
    loss_axis.grid(True, alpha=0.25)
    loss_axis.legend()

    grid_x, grid_y = np.meshgrid(
        np.linspace(-0.2, 1.2, 240), np.linspace(-0.2, 1.2, 240)
    )
    grid_points = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    probabilities, _ = forward(grid_points, parameters)
    probability_grid = probabilities.reshape(grid_x.shape)
    boundary_axis.contourf(
        grid_x,
        grid_y,
        probability_grid,
        levels=[0.0, 0.5, 1.0],
        colors=["#d9eaf7", "#fde2d5"],
        alpha=0.9,
    )
    boundary_axis.contour(
        grid_x, grid_y, probability_grid, levels=[0.5], colors="#253858", linewidths=2
    )
    labels = Y.ravel().astype(int)
    boundary_axis.scatter(
        X[:, 0],
        X[:, 1],
        c=np.where(labels == 1, "#1565c0", "#d84315"),
        edgecolors="black",
        s=110,
        zorder=3,
    )
    boundary_axis.set(
        title="Momentum model: XOR decision boundary",
        xlabel="Input x₁",
        ylabel="Input x₂",
        xlim=(-0.2, 1.2),
        ylim=(-0.2, 1.2),
        aspect="equal",
    )
    boundary_axis.set_xticks([0, 1])
    boundary_axis.set_yticks([0, 1])
    boundary_axis.grid(True, alpha=0.2)

    figure.suptitle("XOR learned with calculus (NumPy from scratch)", fontsize=14)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches="tight")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7, help="reproducible initialization seed")
    parser.add_argument("--epochs", type=int, default=12_000, help="full-batch training steps")
    parser.add_argument("--no-show", action="store_true", help="save the figure without opening a window")
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive")

    initial = initialize_parameters(args.seed)
    standard_parameters, standard_history = train(
        initial, learning_rate=0.8, epochs=args.epochs
    )
    momentum_parameters, momentum_history = train(
        initial, learning_rate=0.8, epochs=args.epochs, momentum=0.9
    )

    standard_predictions, _ = forward(X, standard_parameters)
    momentum_predictions, _ = forward(X, momentum_parameters)
    standard_accuracy = float(np.mean((standard_predictions >= 0.5) == Y))
    momentum_accuracy = float(np.mean((momentum_predictions >= 0.5) == Y))
    output_path = Path(__file__).resolve().with_name("xor_calculus_results.png")
    make_figure(standard_history, momentum_history, momentum_parameters, output_path)

    print("=" * 66)
    print(" XOR NEURAL NETWORK | CALCULUS IN ACTION")
    print("=" * 66)
    print("Architecture : 2 inputs -> 4 tanh neurons -> 1 sigmoid output")
    print(f"Seed / steps : {args.seed} / {args.epochs:,} full-batch updates")
    print("Calculus     : chain rule + matrix gradients + averaged loss")
    print("-" * 66)
    print(f"{'Optimizer':<29}{'Final loss':>16}{'Accuracy':>17}")
    print("-" * 66)
    print(f"{'Standard gradient descent':<29}{standard_history[-1]:>16.6f}{standard_accuracy:>16.0%}")
    print(f"{'Gradient descent + momentum':<29}{momentum_history[-1]:>16.6f}{momentum_accuracy:>16.0%}")
    print("-" * 66)
    print("Momentum predictions (00, 01, 10, 11):")
    print("  " + "  ".join(f"{value:.3f}" for value in momentum_predictions.ravel()))
    print(f"Figure saved to: {output_path}")
    print("=" * 66)

    if not args.no_show:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()
