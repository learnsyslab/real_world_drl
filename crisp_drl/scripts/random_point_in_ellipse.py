import numpy as np
import random
import matplotlib.pyplot as plt


# Method 1: rho and phi sampled uniformly
def random_point_method1(width, height):
    rho = random.random()
    phi = random.random() * 2 * np.pi
    x = np.sqrt(rho) * np.cos(phi)
    y = np.sqrt(rho) * np.sin(phi)
    x = x * width / 2.0
    y = y * height / 2.0
    return np.array([x, y])


# Method 2: theta-based sampling
def generate_theta(a, b):
    """Returns theta in [-pi/2, 3pi/2]"""
    u = random.random() / 4.0
    theta = np.arctan(b / a * np.tan(2 * np.pi * u))

    v = random.random()
    if v < 0.25:
        return theta
    elif v < 0.5:
        return np.pi - theta
    elif v < 0.75:
        return np.pi + theta
    else:
        return -theta


def radius(a, b, theta):
    return a * b / np.sqrt((b * np.cos(theta)) ** 2 + (a * np.sin(theta)) ** 2)


def random_point_method2(a, b):
    random_theta = generate_theta(a, b)
    max_radius = radius(a, b, random_theta)
    random_radius = max_radius * np.sqrt(random.random())

    return np.array(
        [random_radius * np.cos(random_theta), random_radius * np.sin(random_theta)]
    )


def compare_distributions(n_samples=10000, width=4.0, height=2.0):
    """Compare the two sampling methods."""
    a = width / 2.0
    b = height / 2.0

    # Generate samples
    samples1 = np.array([random_point_method1(width, height) for _ in range(n_samples)])
    samples2 = np.array([random_point_method2(a, b) for _ in range(n_samples)])

    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Scatter plots
    axes[0, 0].scatter(samples1[:, 0], samples1[:, 1], alpha=0.3, s=1)
    axes[0, 0].set_title("Method 1: Scatter Plot")
    axes[0, 0].set_xlim(-a * 1.1, a * 1.1)
    axes[0, 0].set_ylim(-b * 1.1, b * 1.1)
    axes[0, 0].set_aspect("equal")
    axes[0, 0].set_xlabel("x")
    axes[0, 0].set_ylabel("y")

    axes[1, 0].scatter(samples2[:, 0], samples2[:, 1], alpha=0.3, s=1)
    axes[1, 0].set_title("Method 2: Scatter Plot")
    axes[1, 0].set_xlim(-a * 1.1, a * 1.1)
    axes[1, 0].set_ylim(-b * 1.1, b * 1.1)
    axes[1, 0].set_aspect("equal")
    axes[1, 0].set_xlabel("x")
    axes[1, 0].set_ylabel("y")

    # X histograms
    bins = 50
    axes[0, 1].hist(samples1[:, 0], bins=bins, density=True, alpha=0.7)
    axes[0, 1].set_title("Method 1: X Distribution")
    axes[0, 1].set_xlabel("x")
    axes[0, 1].set_ylabel("Density")

    axes[1, 1].hist(samples2[:, 0], bins=bins, density=True, alpha=0.7)
    axes[1, 1].set_title("Method 2: X Distribution")
    axes[1, 1].set_xlabel("x")
    axes[1, 1].set_ylabel("Density")

    # Y histograms
    axes[0, 2].hist(samples1[:, 1], bins=bins, density=True, alpha=0.7)
    axes[0, 2].set_title("Method 1: Y Distribution")
    axes[0, 2].set_xlabel("y")
    axes[0, 2].set_ylabel("Density")

    axes[1, 2].hist(samples2[:, 1], bins=bins, density=True, alpha=0.7)
    axes[1, 2].set_title("Method 2: Y Distribution")
    axes[1, 2].set_xlabel("y")
    axes[1, 2].set_ylabel("Density")

    plt.tight_layout()
    plt.savefig("ellipse_comparison.png", dpi=150)
    plt.show()

    # 2D histogram comparison
    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))

    h1 = axes2[0].hist2d(samples1[:, 0], samples1[:, 1], bins=30, cmap="Blues")
    axes2[0].set_title("Method 1: 2D Histogram")
    axes2[0].set_xlabel("x")
    axes2[0].set_ylabel("y")
    axes2[0].set_aspect("equal")
    plt.colorbar(h1[3], ax=axes2[0])

    h2 = axes2[1].hist2d(samples2[:, 0], samples2[:, 1], bins=30, cmap="Blues")
    axes2[1].set_title("Method 2: 2D Histogram")
    axes2[1].set_xlabel("x")
    axes2[1].set_ylabel("y")
    axes2[1].set_aspect("equal")
    plt.colorbar(h2[3], ax=axes2[1])

    plt.tight_layout()
    plt.savefig("ellipse_2d_histogram.png", dpi=150)
    plt.show()

    # Print statistics
    print("=" * 60)
    print("Statistics Comparison")
    print("=" * 60)
    print(
        f"Method 1 - X: mean={samples1[:, 0].mean():.4f}, std={samples1[:, 0].std():.4f}"
    )
    print(
        f"Method 2 - X: mean={samples2[:, 0].mean():.4f}, std={samples2[:, 0].std():.4f}"
    )
    print(
        f"Method 1 - Y: mean={samples1[:, 1].mean():.4f}, std={samples1[:, 1].std():.4f}"
    )
    print(
        f"Method 2 - Y: mean={samples2[:, 1].mean():.4f}, std={samples2[:, 1].std():.4f}"
    )

    # Check if all points are inside the ellipse
    inside1 = np.sum((samples1[:, 0] / a) ** 2 + (samples1[:, 1] / b) ** 2 <= 1)
    inside2 = np.sum((samples2[:, 0] / a) ** 2 + (samples2[:, 1] / b) ** 2 <= 1)
    print(f"\nPoints inside ellipse:")
    print(f"Method 1: {inside1}/{n_samples} ({100 * inside1 / n_samples:.2f}%)")
    print(f"Method 2: {inside2}/{n_samples} ({100 * inside2 / n_samples:.2f}%)")


if __name__ == "__main__":
    compare_distributions(n_samples=50000, width=4.0, height=2.0)
