import matplotlib.pyplot as plt
import numpy as np


def generate_sample_distrib_chat(
    latencies: np.ndarray,
    title: str = "Time Distribution",
    x_label: str = "Latency (ms)",
    y_label: str = "Number of Operations",
):
    # Generate a more realistic distribution of latencies for a web server handling millions of requests
    np.random.seed(42)  # For reproducibility

    # Calculate percentiles
    _percentiles = np.percentile(latencies, [50, 90, 95, 99, 99.9])

    # Plot the histogram of latencies with percentile markers
    plt.figure(figsize=(10, 6))
    plt.hist(all_latencies, bins=1000, color="skyblue", alpha=0.7, label="Response Times", edgecolor="lightgray")
    plt.axvline(_percentiles[0], color="green", linestyle="dashed", linewidth=1, label="50th Percentile")
    plt.axvline(_percentiles[1], color="blue", linestyle="dashed", linewidth=1, label="90th Percentile")
    plt.axvline(_percentiles[2], color="orange", linestyle="dashed", linewidth=1, label="95th Percentile")
    plt.axvline(_percentiles[3], color="red", linestyle="dashed", linewidth=1, label="99th Percentile")
    plt.axvline(_percentiles[4], color="purple", linestyle="dashed", linewidth=1, label="99.9th Percentile")

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.xlim(0, 2000)  # Limit x-axis for better visualization
    plt.legend()
    plt.yscale("log")  # Log scale for y-axis to handle wide range of values
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    # Simulate latencies: Mostly fast responses, with a long tail of slower responses
    # Using a combination of normal distribution for the majority of responses and a log-normal for the long tail
    normal_latencies = np.random.normal(loc=120, scale=20, size=950000)  # Fast responses
    long_tail_latencies = np.random.lognormal(mean=5, sigma=0.4, size=50000)  # Simulating the long tail

    # Combine the distributions
    all_latencies = np.concatenate((normal_latencies, long_tail_latencies))

    generate_sample_distrib_chat(
        all_latencies, title="Web Server Response Time Distribution", y_label="Number of Requests"
    )
