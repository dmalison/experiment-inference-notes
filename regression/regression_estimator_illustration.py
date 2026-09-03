"""Compare unadjusted, CUPED, and fully interacted regression ATE estimators."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy import stats


N_UNITS = 1_000
TREATMENT_PROBABILITY = 0.5
LEVEL_MEAN_VECTOR = (0.0, 1.0, 0.0)
LEVEL_SCALE_VECTOR = (1.0, 1.0, 1.0)
LEVEL_CORRELATION_MATRIX = (
    (1.0, 1.0, 0.5),
    (1.0, 1.0, 0.5),
    (0.5, 0.5, 1.0),
)
INTERACTION_MEAN_VECTOR = (0.0, 0.0, 0.0)
INTERACTION_SCALE_VECTOR = (1.0, 3.0, 1.0)
INTERACTION_CORRELATION_MATRIX = (
    (1.0, 1.0, 1.0 / 2.0),
    (1.0, 1.0, 1.0 / 2.0),
    (1.0 / 2.0, 1.0 / 2.0, 1.0),
)
N_DRAWS = 50_000
CHUNK_SIZE = 500
POPULATION_SEED = 8
ASSIGNMENT_SEED = 0
CI_LEVEL = 0.95

CONTROL_COLOR = "#4c78a8"
TREATMENT_COLOR = "#e45756"


@dataclass(frozen=True)
class Scenario:
    title: str
    y0: np.ndarray
    y1: np.ndarray
    output_name: str

    @property
    def ate(self) -> float:
        return float(self.y1.mean() - self.y0.mean())


@dataclass(frozen=True)
class SimulationDraws:
    estimates: dict[str, np.ndarray]
    interval_widths: dict[str, np.ndarray]


ESTIMATOR_NAMES = (
    "Treatment-group means",
    "CUPED",
    "Regression",
)


def draw_population(
    rng: np.random.Generator,
    mean_vector: tuple[float, float, float],
    scale_vector: tuple[float, float, float],
    correlation_matrix: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a finite population with exact target means and covariance."""
    scales = np.asarray(scale_vector)
    correlation = np.asarray(correlation_matrix)
    mean = np.asarray(mean_vector)
    initial_draws = rng.standard_normal((N_UNITS, 3))
    centered_draws = initial_draws - initial_draws.mean(axis=0)
    sample_covariance = centered_draws.T @ centered_draws / N_UNITS
    eigenvalues, eigenvectors = np.linalg.eigh(sample_covariance)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        inverse_square_root = (
            eigenvectors
            @ np.diag(np.power(eigenvalues, -0.5))
            @ eigenvectors.T
        )
        standardized_draws = centered_draws @ inverse_square_root
        try:
            correlation_factor = np.linalg.cholesky(correlation)
        except np.linalg.LinAlgError:
            eigenvalues, eigenvectors = np.linalg.eigh(correlation)
            if eigenvalues.min() < -1e-12:
                raise ValueError(
                    "correlation matrix must be positive semidefinite"
                )
            correlation_factor = eigenvectors @ np.diag(
                np.sqrt(np.maximum(eigenvalues, 0.0))
            )
        population = standardized_draws @ correlation_factor.T
    population = population * scales + mean

    target_covariance = correlation * np.outer(scales, scales)
    assert np.allclose(population.mean(axis=0), mean, atol=1e-12)
    assert np.allclose(
        np.cov(population, rowvar=False, bias=True),
        target_covariance,
        atol=1e-12,
    )
    return population[:, 0], population[:, 1], population[:, 2]


def confidence_interval_width(
    control_variance: np.ndarray,
    treatment_variance: np.ndarray,
    control_count: np.ndarray,
    treatment_count: np.ndarray,
) -> np.ndarray:
    """Compute the chapter's conservative ATE confidence-interval width."""
    control_scale = np.sqrt(
        treatment_count / control_count * control_variance
    )
    treatment_scale = np.sqrt(
        control_count / treatment_count * treatment_variance
    )
    standard_error = (control_scale + treatment_scale) / np.sqrt(N_UNITS)
    critical_value = stats.norm.ppf((1.0 + CI_LEVEL) / 2.0)
    return 2.0 * critical_value * standard_error


def simulate_estimators(
    scenario: Scenario,
    covariate: np.ndarray,
) -> SimulationDraws:
    """Simulate the three ATE estimators over repeated Bernoulli assignments."""
    rng = np.random.default_rng(ASSIGNMENT_SEED)
    covariate = covariate - covariate.mean()
    moment_matrix = np.column_stack(
        (
            covariate,
            np.square(covariate),
            scenario.y0,
            np.square(scenario.y0),
            covariate * scenario.y0,
            scenario.y1,
            np.square(scenario.y1),
            covariate * scenario.y1,
        )
    )
    population_totals = moment_matrix.sum(axis=0)
    covariate_sum_of_squares = population_totals[1]

    estimate_batches = {name: [] for name in ESTIMATOR_NAMES}
    width_batches = {name: [] for name in ESTIMATOR_NAMES}

    completed = 0
    while completed < N_DRAWS:
        batch_size = min(CHUNK_SIZE, N_DRAWS - completed)
        treatment = rng.binomial(
            1,
            TREATMENT_PROBABILITY,
            size=(batch_size, N_UNITS),
        ).astype(float)
        treatment_count = treatment.sum(axis=1)
        control_count = N_UNITS - treatment_count
        valid = (treatment_count > 1.0) & (control_count > 1.0)
        treatment = treatment[valid]
        treatment_count = treatment_count[valid]
        control_count = control_count[valid]

        treatment_sums = np.einsum(
            "bi,ij->bj", treatment, moment_matrix, optimize=False
        )

        treatment_covariate_sum = treatment_sums[:, 0]
        treatment_covariate_square_sum = treatment_sums[:, 1]
        treatment_outcome_sum = treatment_sums[:, 5]
        treatment_outcome_square_sum = treatment_sums[:, 6]
        treatment_covariate_outcome_sum = treatment_sums[:, 7]

        control_covariate_sum = (
            population_totals[0] - treatment_covariate_sum
        )
        control_covariate_square_sum = (
            population_totals[1] - treatment_covariate_square_sum
        )
        control_outcome_sum = population_totals[2] - treatment_sums[:, 2]
        control_outcome_square_sum = (
            population_totals[3] - treatment_sums[:, 3]
        )
        control_covariate_outcome_sum = (
            population_totals[4] - treatment_sums[:, 4]
        )

        treatment_mean = treatment_outcome_sum / treatment_count
        control_mean = control_outcome_sum / control_count
        treatment_covariate_mean = (
            treatment_covariate_sum / treatment_count
        )
        control_covariate_mean = control_covariate_sum / control_count

        treatment_covariate_ss = np.maximum(
            treatment_covariate_square_sum
            - np.square(treatment_covariate_sum) / treatment_count,
            0.0,
        )
        control_covariate_ss = np.maximum(
            control_covariate_square_sum
            - np.square(control_covariate_sum) / control_count,
            0.0,
        )
        treatment_outcome_ss = np.maximum(
            treatment_outcome_square_sum
            - np.square(treatment_outcome_sum) / treatment_count,
            0.0,
        )
        control_outcome_ss = np.maximum(
            control_outcome_square_sum
            - np.square(control_outcome_sum) / control_count,
            0.0,
        )
        treatment_cross_product = (
            treatment_covariate_outcome_sum
            - treatment_covariate_sum * treatment_outcome_sum
            / treatment_count
        )
        control_cross_product = (
            control_covariate_outcome_sum
            - control_covariate_sum * control_outcome_sum / control_count
        )

        treatment_slope = (
            treatment_cross_product / treatment_covariate_ss
        )
        control_slope = control_cross_product / control_covariate_ss
        regression_estimate = (
            treatment_mean
            - treatment_covariate_mean * treatment_slope
            - control_mean
            + control_covariate_mean * control_slope
        )
        regression_treatment_residual_ss = np.maximum(
            treatment_outcome_ss
            - np.square(treatment_cross_product) / treatment_covariate_ss,
            0.0,
        )
        regression_control_residual_ss = np.maximum(
            control_outcome_ss
            - np.square(control_cross_product) / control_covariate_ss,
            0.0,
        )

        cuped_slope = (
            control_covariate_outcome_sum
            + treatment_covariate_outcome_sum
        ) / covariate_sum_of_squares
        unadjusted_estimate = treatment_mean - control_mean
        cuped_estimate = unadjusted_estimate - (
            treatment_covariate_mean - control_covariate_mean
        ) * cuped_slope
        cuped_treatment_residual_ss = np.maximum(
            treatment_outcome_ss
            - 2.0 * cuped_slope * treatment_cross_product
            + np.square(cuped_slope) * treatment_covariate_ss,
            0.0,
        )
        cuped_control_residual_ss = np.maximum(
            control_outcome_ss
            - 2.0 * cuped_slope * control_cross_product
            + np.square(cuped_slope) * control_covariate_ss,
            0.0,
        )

        method_values = {
            "Treatment-group means": (
                unadjusted_estimate,
                control_outcome_ss / control_count,
                treatment_outcome_ss / treatment_count,
            ),
            "CUPED": (
                cuped_estimate,
                cuped_control_residual_ss / control_count,
                cuped_treatment_residual_ss / treatment_count,
            ),
            "Regression": (
                regression_estimate,
                regression_control_residual_ss / control_count,
                regression_treatment_residual_ss / treatment_count,
            ),
        }
        for name, (
            estimates,
            control_variance,
            treatment_variance,
        ) in method_values.items():
            estimate_batches[name].append(estimates)
            width_batches[name].append(
                confidence_interval_width(
                    control_variance,
                    treatment_variance,
                    control_count,
                    treatment_count,
                )
            )
        completed += treatment_count.size

    return SimulationDraws(
        estimates={
            name: np.concatenate(batches)[:N_DRAWS]
            for name, batches in estimate_batches.items()
        },
        interval_widths={
            name: np.concatenate(batches)[:N_DRAWS]
            for name, batches in width_batches.items()
        },
    )


def draw_figure(
    scenario: Scenario,
    covariate: np.ndarray,
    example_treatment: np.ndarray,
) -> plt.Figure:
    """Draw potential outcomes and population and sample projections."""
    covariate = covariate - covariate.mean()
    figure, dgp_axis = plt.subplots(
        figsize=(6.2, 4.8),
        dpi=150,
    )

    shown = np.argsort(covariate)[::5]
    dgp_axis.scatter(
        covariate[shown],
        scenario.y0[shown],
        s=13,
        color=CONTROL_COLOR,
        alpha=0.32,
        linewidths=0.0,
    )
    dgp_axis.scatter(
        covariate[shown],
        scenario.y1[shown],
        s=13,
        color=TREATMENT_COLOR,
        alpha=0.32,
        linewidths=0.0,
    )
    selected_outcomes = np.where(
        example_treatment,
        scenario.y1,
        scenario.y0,
    )
    dgp_axis.scatter(
        covariate[shown],
        selected_outcomes[shown],
        s=25,
        facecolors="none",
        edgecolors="#222222",
        linewidths=0.7,
        zorder=4,
    )
    covariate_grid = np.linspace(
        np.quantile(covariate, 0.005),
        np.quantile(covariate, 0.995),
        200,
    )
    covariate_ss = np.dot(covariate, covariate)
    control_slope = np.dot(covariate, scenario.y0) / covariate_ss
    treatment_slope = np.dot(covariate, scenario.y1) / covariate_ss
    dgp_axis.plot(
        covariate_grid,
        scenario.y0.mean() + control_slope * covariate_grid,
        color=CONTROL_COLOR,
        linewidth=2.2,
    )
    dgp_axis.plot(
        covariate_grid,
        scenario.y1.mean() + treatment_slope * covariate_grid,
        color=TREATMENT_COLOR,
        linewidth=2.2,
    )
    estimated_projections = []
    for assigned, outcome in (
        (~example_treatment, scenario.y0),
        (example_treatment, scenario.y1),
    ):
        assigned_covariate = covariate[assigned]
        assigned_outcome = outcome[assigned]
        centered_assigned_covariate = (
            assigned_covariate - assigned_covariate.mean()
        )
        estimated_slope = np.dot(
            centered_assigned_covariate,
            assigned_outcome - assigned_outcome.mean(),
        ) / np.dot(
            centered_assigned_covariate,
            centered_assigned_covariate,
        )
        estimated_intercept = (
            assigned_outcome.mean()
            - assigned_covariate.mean() * estimated_slope
        )
        estimated_projections.append(
            (estimated_intercept, estimated_slope)
        )
    for (estimated_intercept, estimated_slope), color in zip(
        estimated_projections,
        (CONTROL_COLOR, TREATMENT_COLOR),
    ):
        dgp_axis.plot(
            covariate_grid,
            estimated_intercept + estimated_slope * covariate_grid,
            color=color,
            linewidth=1.8,
            linestyle="--",
            zorder=3,
        )
    dgp_axis.set_xlabel(r"centered covariate $w_i-\overline{w}_n$")
    dgp_axis.set_ylabel("potential outcome")
    lower_limit, upper_limit = dgp_axis.get_ylim()
    dgp_axis.set_ylim(
        lower_limit,
        upper_limit + 0.3 * (upper_limit - lower_limit),
    )
    legend_handles = [
        Line2D(
            [],
            [],
            color=CONTROL_COLOR,
            linestyle="None",
            marker="o",
            markersize=4.5,
            markeredgewidth=0.0,
            label=r"$y_{i,0}$",
        ),
        Line2D(
            [],
            [],
            color=CONTROL_COLOR,
            linewidth=2.2,
            label=(
                r"$\overline{y}_0+\beta_{0,n}"
                r"(w_i-\overline{w}_n)$"
            ),
        ),
        Line2D(
            [],
            [],
            color=CONTROL_COLOR,
            linewidth=1.8,
            linestyle="--",
            label=(
                r"$\widehat{y}_{0,n}^{\mathrm{reg}}"
                r"+\widehat{\beta}_{0,n}(w_i-\overline{w}_n)$"
            ),
        ),
        Line2D(
            [],
            [],
            color=TREATMENT_COLOR,
            linestyle="None",
            marker="o",
            markersize=4.5,
            markeredgewidth=0.0,
            label=r"$y_{i,1}$",
        ),
        Line2D(
            [],
            [],
            color=TREATMENT_COLOR,
            linewidth=2.2,
            label=(
                r"$\overline{y}_1+\beta_{1,n}"
                r"(w_i-\overline{w}_n)$"
            ),
        ),
        Line2D(
            [],
            [],
            color=TREATMENT_COLOR,
            linewidth=1.8,
            linestyle="--",
            label=(
                r"$\widehat{y}_{1,n}^{\mathrm{reg}}"
                r"+\widehat{\beta}_{1,n}(w_i-\overline{w}_n)$"
            ),
        ),
        Line2D(
            [],
            [],
            color="#222222",
            linestyle="None",
            marker="o",
            markerfacecolor="none",
            markersize=5.5,
            markeredgewidth=0.7,
            label=r"$Y_i$",
        ),
        Line2D([], [], linestyle="None", label=""),
        Line2D([], [], linestyle="None", label=""),
    ]
    legend = dgp_axis.legend(
        handles=legend_handles,
        frameon=True,
        framealpha=1.0,
        edgecolor="#999999",
        facecolor="white",
        fontsize=8.0,
        loc="upper left",
        ncols=3,
        columnspacing=1.4,
        handletextpad=0.6,
        borderpad=0.7,
        borderaxespad=0.6,
    )
    legend.get_frame().set_linewidth(0.7)
    dgp_axis.spines["top"].set_visible(False)
    dgp_axis.spines["right"].set_visible(False)
    dgp_axis.tick_params(labelsize=9)

    figure.tight_layout()
    return figure


def validate_results(
    level_draws: SimulationDraws,
    interaction_draws: SimulationDraws,
    level_ate: float,
    interaction_ate: float,
) -> None:
    """Check simulation invariants and the comparisons shown in the figures."""
    for name, estimates in level_draws.estimates.items():
        assert abs(estimates.mean() - level_ate) < 0.003, name
    for name, estimates in interaction_draws.estimates.items():
        assert abs(estimates.mean() - interaction_ate) < 0.003, name

    level_standard_deviations = {
        name: estimates.std()
        for name, estimates in level_draws.estimates.items()
    }
    level_widths = {
        name: widths.mean()
        for name, widths in level_draws.interval_widths.items()
    }
    assert np.isclose(
        level_standard_deviations["CUPED"],
        level_standard_deviations["Regression"],
        rtol=0.03,
    )
    assert np.isclose(
        level_widths["CUPED"],
        level_widths["Regression"],
        rtol=0.03,
    )
    assert (
        level_standard_deviations["Regression"]
        < level_standard_deviations["Treatment-group means"]
    )
    assert (
        level_widths["Regression"]
        < level_widths["Treatment-group means"]
    )

    interaction_standard_deviations = {
        name: estimates.std()
        for name, estimates in interaction_draws.estimates.items()
    }
    interaction_widths = {
        name: widths.mean()
        for name, widths in interaction_draws.interval_widths.items()
    }
    assert np.isclose(
        interaction_standard_deviations["CUPED"],
        interaction_standard_deviations["Regression"],
        rtol=0.03,
    )
    assert (
        interaction_standard_deviations["Regression"]
        < interaction_standard_deviations["Treatment-group means"]
    )
    assert (
        interaction_standard_deviations["CUPED"]
        < interaction_standard_deviations["Treatment-group means"]
    )
    assert (
        interaction_widths["Regression"]
        < interaction_widths["CUPED"]
        < interaction_widths["Treatment-group means"]
    )


def print_summary(scenario: Scenario, draws: SimulationDraws) -> None:
    print(f"{scenario.title}: ATE = {scenario.ate:.4f}")
    for name in ESTIMATOR_NAMES:
        estimates = draws.estimates[name]
        interval_widths = draws.interval_widths[name]
        coverage = np.mean(
            np.abs(estimates - scenario.ate) <= interval_widths / 2.0
        )
        print(
            f"  {name}: coverage = {coverage:.4f}, "
            f"mean CI width = {interval_widths.mean():.4f}"
        )


def main() -> None:
    rng = np.random.default_rng(POPULATION_SEED)
    assignment_rng = np.random.default_rng(ASSIGNMENT_SEED)
    example_treatment = assignment_rng.binomial(
        1,
        TREATMENT_PROBABILITY,
        size=N_UNITS,
    ).astype(bool)
    assert 1 < example_treatment.sum() < N_UNITS - 1
    level_y0, level_y1, level_covariate = draw_population(
        rng,
        mean_vector=LEVEL_MEAN_VECTOR,
        scale_vector=LEVEL_SCALE_VECTOR,
        correlation_matrix=LEVEL_CORRELATION_MATRIX,
    )
    assert np.allclose(level_y1 - level_y0, 1.0, atol=1e-12)
    level_scenario = Scenario(
        title="Level effect shift",
        y0=level_y0,
        y1=level_y1,
        output_name="regression_level_effect_illustration.png",
    )
    interaction_y0, interaction_y1, interaction_covariate = draw_population(
        rng,
        mean_vector=INTERACTION_MEAN_VECTOR,
        scale_vector=INTERACTION_SCALE_VECTOR,
        correlation_matrix=INTERACTION_CORRELATION_MATRIX,
    )
    assert np.allclose(interaction_y1, 3.0 * interaction_y0, atol=1e-12)
    interaction_scenario = Scenario(
        title="Interaction effect",
        y0=interaction_y0,
        y1=interaction_y1,
        output_name="regression_interaction_effect_illustration.png",
    )

    level_draws = simulate_estimators(level_scenario, level_covariate)
    interaction_draws = simulate_estimators(
        interaction_scenario,
        interaction_covariate,
    )
    validate_results(
        level_draws,
        interaction_draws,
        level_scenario.ate,
        interaction_scenario.ate,
    )

    output_directory = Path(__file__).parent
    for scenario, covariate, draws in (
        (level_scenario, level_covariate, level_draws),
        (interaction_scenario, interaction_covariate, interaction_draws),
    ):
        figure = draw_figure(scenario, covariate, example_treatment)
        output = output_directory / scenario.output_name
        figure.savefig(output, dpi=150, bbox_inches="tight")
        plt.close(figure)
        print(f"wrote {output}")
        print_summary(scenario, draws)


if __name__ == "__main__":
    main()