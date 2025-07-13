# JeffersonStats: Advanced Statistical Analysis MCP Server

![Version](https://img.shields.io/badge/version-0.1.0-blue)
![Python](https://img.shields.io/badge/python-3.11+-brightgreen)
![License](https://img.shields.io/badge/license-MIT-green)

## Overview

JeffersonStats is a powerful, high-performance statistical analysis server built on the FastMCP framework. It provides a comprehensive suite of statistical tools accessible via a clean, intuitive API. Whether you're performing basic descriptive statistics or advanced statistical tests, JeffersonStats delivers accurate results with minimal configuration.

## Features

JeffersonStats offers a rich set of statistical capabilities:

### Basic Statistics
- Mean, median, mode, and range calculations
- Standard deviation and variance
- Quartiles and interquartile range (IQR)
- Percentile and quantile calculations

### Advanced Statistics
- Skewness and kurtosis analysis
- Correlation coefficients (Pearson, Spearman, Kendall's tau)
- Covariance calculations
- Z-score transformations

### Hypothesis Testing
- T-tests (one-sample, independent, paired)
- ANOVA (Analysis of Variance)
- Chi-square tests
- Mann-Whitney U test
- Wilcoxon signed-rank test
- Normality tests (Shapiro-Wilk)
- Binomial tests

### Data Analysis
- Linear regression
- Confidence intervals (standard and bootstrap)
- Outlier detection
- Moving averages
- Frequency tables
- Comprehensive descriptive statistics summaries

## Why Choose JeffersonStats?

- **High Performance**: Built on optimized NumPy and SciPy libraries for fast computation
- **Easy Integration**: Simple HTTP API that works with any programming language or platform
- **Comprehensive**: Over 30 statistical tools in a single package
- **Reliable**: Based on industry-standard statistical implementations
- **Containerized**: Easy deployment with Docker
- **Scalable**: Designed to handle large datasets efficiently

## Installation

### Using Python

```bash
# Clone the repository
git clone https://github.com/yourusername/JeffersonStats.git
cd JeffersonStats

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the server
python mcpserver.py
```

### Using Docker

```bash
# Clone the repository
git clone https://github.com/yourusername/JeffersonStats.git
cd JeffersonStats

# Build the Docker image
docker build -t jeffersonstats .

# Run the container
docker run -p 8080:8080 jeffersonstats
```

The server will be available at http://localhost:8080.

## Usage

JeffersonStats exposes its statistical tools through a MCP server using streamble-http transport. Here are some examples:

## MCP Clients supported

- CherryStudio
- VSCode
- Cursor
- WindSurf
- BlackGoose

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built with [FastMCP](https://github.com/fastmcp/fastmcp)
- Statistical calculations powered by [NumPy](https://numpy.org/) and [SciPy](https://scipy.org/)
