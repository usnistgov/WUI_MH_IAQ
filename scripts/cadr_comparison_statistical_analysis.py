"""
Statistical analysis script for WUI CADR comparison (Figure 4).

This script performs z-test comparisons between portable air cleaner CADRs
measured in different configurations:
1. Laboratory standard (ASTM WK81750): 390 ± 30 m³/h
2. 1 Air Cleaner (Sealed Room): 347.4 ± 108.6 m³/h
3. 4 Air Cleaners (House) per cleaner: 458.2 ± 25.0 m³/h

The z-test approach follows the Type B uncertainty methodology described
in John Lu's email (September 17, 2025) and the Lowhorn et al. (2011) paper:
- var(mu_a - mu_b) = var(mu_a) + var(mu_b) = sd(mu_a)^2 + sd(mu_b)^2
- z = (mu_a - mu_b) / sqrt(var(mu_a - mu_b))
- p-value = 2 * (1 - Pnorm(|z|)) for two-tailed test
- p-value = 1 - Pnorm(z) for one-tailed test (greater than)

Requires:
    - numpy
    - scipy

Output:
    - Console output with statistical test results
"""

import numpy as np
from scipy.stats import norm

# ============================================================================
# STATISTICAL ANALYSIS CONFIGURATION
# ============================================================================
STATISTICAL_CONFIG = {
    "alpha": 0.05,  # Significance level for hypothesis tests
    "confidence_level": 0.95,  # Confidence level (1 - alpha)
}


# ============================================================================
# DATA FROM FIGURE 4
# ============================================================================
# CADR values in m³/h with uncertainties

CADR_DATA = {
    "Laboratory_Standard": {
        "label": "Laboratory Standard (ASTM WK81750)",
        "mean": 390.0,  # m³/h
        "uncertainty": 30.0,  # m³/h (95% CI)
    },
    "Sealed_Room": {
        "label": "1 Air Cleaner (Sealed Room)",
        "mean": 347.4,  # m³/h
        "uncertainty": 108.6,  # m³/h (95% CI)
    },
    "Four_Air_Cleaners": {
        "label": "4 Air Cleaners (House) per cleaner",
        "mean": 458.2,  # m³/h
        "uncertainty": 25.0,  # m³/h (95% CI)
    },
}


# ============================================================================
# STATISTICAL ANALYSIS FUNCTIONS
# ============================================================================


def get_significance_indicator(p_value):
    """
    Return significance indicator based on p-value.
    
    Parameters:
    -----------
    p_value : float
        P-value from statistical test
        
    Returns:
    --------
    str : Significance indicator (*** / ** / * / ns)
    """
    if p_value < 0.001:
        return "***"
    elif p_value < 0.01:
        return "**"
    elif p_value < 0.05:
        return "*"
    else:
        return "ns"


def perform_z_test_comparison(mu_a, sd_a, mu_b, sd_b, label_a, label_b, test_type="two-tailed"):
    """
    Perform z-test comparison between two groups using Type B uncertainty approach.
    
    Based on John Lu's email from September 17, 2025:
    - var(mu_a - mu_b) = var(mu_a) + var(mu_b) = sd(mu_a)^2 + sd(mu_b)^2
    - z = (mu_a - mu_b) / sqrt(var(mu_a - mu_b))
    - p-value = 2 * (1 - Pnorm(|z|)) for two-tailed test
    - p-value = 1 - Pnorm(z) for one-tailed test (greater than)
    
    Parameters:
    -----------
    mu_a : float
        Mean value for group A
    sd_a : float
        Standard deviation (uncertainty) for group A
    mu_b : float
        Mean value for group B
    sd_b : float
        Standard deviation (uncertainty) for group B
    label_a : str
        Label for group A
    label_b : str
        Label for group B
    test_type : str
        "two-tailed" or "one-tailed-greater" (A > B)
        
    Returns:
    --------
    dict : Dictionary with test results
    """
    # Calculate combined variance
    var_diff = sd_a**2 + sd_b**2
    
    # Calculate z-statistic
    z_stat = (mu_a - mu_b) / np.sqrt(var_diff)
    
    # Calculate p-value based on test type
    if test_type == "two-tailed":
        p_value = 2 * (1 - norm.cdf(abs(z_stat)))
    elif test_type == "one-tailed-greater":
        # Testing if mu_a > mu_b
        p_value = 1 - norm.cdf(z_stat)
    else:
        raise ValueError("test_type must be 'two-tailed' or 'one-tailed-greater'")
    
    # Determine significance
    alpha = STATISTICAL_CONFIG["alpha"]
    is_significant = p_value < alpha
    
    # Create significance indicator
    sig_indicator = get_significance_indicator(p_value)
    
    return {
        "label_a": label_a,
        "label_b": label_b,
        "mu_a": mu_a,
        "sd_a": sd_a,
        "mu_b": mu_b,
        "sd_b": sd_b,
        "difference": mu_a - mu_b,
        "var_diff": var_diff,
        "z_stat": z_stat,
        "p_value": p_value,
        "significant": is_significant,
        "sig_indicator": sig_indicator,
        "test_type": test_type,
    }


def print_comparison_results(results):
    """
    Print formatted results from z-test comparison.
    
    Parameters:
    -----------
    results : dict
        Dictionary with test results from perform_z_test_comparison
    """
    print(f"\n  Comparison: {results['label_a']} vs {results['label_b']}")
    print(f"  Test Type: {results['test_type']}")
    print(f"    {results['label_a']}: mean = {results['mu_a']:.4f} ± {results['sd_a']:.4f} m³/h")
    print(f"    {results['label_b']}: mean = {results['mu_b']:.4f} ± {results['sd_b']:.4f} m³/h")
    print(f"    Difference: {results['difference']:.4f} m³/h")
    print(f"    Combined variance: {results['var_diff']:.6f}")
    print(f"    Z-statistic: {results['z_stat']:.4f}")
    print(f"    P-value: {results['p_value']:.6f}")
    print(f"    Significant at α={STATISTICAL_CONFIG['alpha']}: {results['significant']} ({results['sig_indicator']})")


# ============================================================================
# MAIN ANALYSIS
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("STATISTICAL ANALYSIS: WUI CADR COMPARISON (FIGURE 4)")
    print("=" * 80)
    print(f"Configuration: α = {STATISTICAL_CONFIG['alpha']}")
    print(f"Significance indicators: *** p<0.001, ** p<0.01, * p<0.05, ns not significant")
    print("=" * 80)
    
    # Extract data
    lab_mean = CADR_DATA["Laboratory_Standard"]["mean"]
    lab_sd = CADR_DATA["Laboratory_Standard"]["uncertainty"]
    lab_label = CADR_DATA["Laboratory_Standard"]["label"]
    
    sealed_mean = CADR_DATA["Sealed_Room"]["mean"]
    sealed_sd = CADR_DATA["Sealed_Room"]["uncertainty"]
    sealed_label = CADR_DATA["Sealed_Room"]["label"]
    
    four_mean = CADR_DATA["Four_Air_Cleaners"]["mean"]
    four_sd = CADR_DATA["Four_Air_Cleaners"]["uncertainty"]
    four_label = CADR_DATA["Four_Air_Cleaners"]["label"]
    
    # ========================================================================
    # SENTENCE 1 ANALYSIS
    # ========================================================================
    print("\n\n" + "=" * 80)
    print("SENTENCE 1 ANALYSIS")
    print("=" * 80)
    print("\nOriginal sentence:")
    print("\"When we sealed the bedroom (HVAC duct and door, 33 m³), the CADR for")
    print("the same portable air cleaner agreed (within error) with the CADR from")
    print("the laboratory test using a similar volume (31.5 m³).\"")
    print("\nComparison: 1 Air Cleaner (Sealed Room) vs Laboratory Standard")
    print("-" * 80)
    
    # Two-tailed test: Are they different?
    sentence1_results = perform_z_test_comparison(
        mu_a=sealed_mean,
        sd_a=sealed_sd,
        mu_b=lab_mean,
        sd_b=lab_sd,
        label_a=sealed_label,
        label_b=lab_label,
        test_type="two-tailed"
    )
    
    print_comparison_results(sentence1_results)
    
    # Interpretation
    print("\n  Interpretation:")
    if sentence1_results['significant']:
        print(f"    The difference IS statistically significant (p = {sentence1_results['p_value']:.6f}).")
        print("    The two values do NOT agree within statistical uncertainty.")
        print("    Suggested revision: Consider revising 'within error' phrasing.")
    else:
        print(f"    The difference is NOT statistically significant (p = {sentence1_results['p_value']:.6f}).")
        print("    The two values agree within statistical uncertainty.")
        print("    The 'within error' phrasing is statistically supported.")
    
    # ========================================================================
    # SENTENCE 2 ANALYSIS
    # ========================================================================
    print("\n\n" + "=" * 80)
    print("SENTENCE 2 ANALYSIS")
    print("=" * 80)
    print("\nOriginal sentence:")
    print("\"When we deployed four portable air cleaners throughout the house (324 m³),")
    print("we measured a per-cleaner CADR (460 m³/h ± 25 m³/h) that was greater than")
    print("the predicted laboratory performance and more than three times larger than")
    print("when deploying only one in the same space.\"")
    print("\nComparison: 4 Air Cleaners (House) per cleaner vs Laboratory Standard")
    print("-" * 80)
    
    # Two-tailed test: Are they different?
    print("\n--- Two-Tailed Test (Are they different?) ---")
    sentence2_twotailed = perform_z_test_comparison(
        mu_a=four_mean,
        sd_a=four_sd,
        mu_b=lab_mean,
        sd_b=lab_sd,
        label_a=four_label,
        label_b=lab_label,
        test_type="two-tailed"
    )
    
    print_comparison_results(sentence2_twotailed)
    
    # One-tailed test: Is 4 air cleaners greater than lab?
    print("\n--- One-Tailed Test (Is 4 air cleaners > Laboratory?) ---")
    sentence2_onetailed = perform_z_test_comparison(
        mu_a=four_mean,
        sd_a=four_sd,
        mu_b=lab_mean,
        sd_b=lab_sd,
        label_a=four_label,
        label_b=lab_label,
        test_type="one-tailed-greater"
    )
    
    print_comparison_results(sentence2_onetailed)
    
    # Interpretation
    print("\n  Interpretation:")
    if sentence2_twotailed['significant']:
        print(f"    Two-tailed: The difference IS statistically significant (p = {sentence2_twotailed['p_value']:.6f}).")
    else:
        print(f"    Two-tailed: The difference is NOT statistically significant (p = {sentence2_twotailed['p_value']:.6f}).")
    
    if sentence2_onetailed['significant']:
        print(f"    One-tailed: 4 air cleaners IS significantly greater than laboratory (p = {sentence2_onetailed['p_value']:.6f}).")
        print("    Suggested addition: Include statistical significance in sentence.")
    else:
        print(f"    One-tailed: 4 air cleaners is NOT significantly greater than laboratory (p = {sentence2_onetailed['p_value']:.6f}).")
    
    # ========================================================================
    # SUMMARY OF RESULTS
    # ========================================================================
    print("\n\n" + "=" * 80)
    print("SUMMARY OF STATISTICAL RESULTS")
    print("=" * 80)
    
    print("\n1. Sealed Room vs Laboratory (Sentence 1):")
    print(f"   - Difference: {sentence1_results['difference']:.2f} m³/h")
    print(f"   - Z-statistic: {sentence1_results['z_stat']:.4f}")
    print(f"   - P-value (two-tailed): {sentence1_results['p_value']:.6f} ({sentence1_results['sig_indicator']})")
    print(f"   - Statistically significant: {sentence1_results['significant']}")
    
    print("\n2. Four Air Cleaners vs Laboratory (Sentence 2):")
    print(f"   - Difference: {sentence2_twotailed['difference']:.2f} m³/h")
    print(f"   - Z-statistic: {sentence2_twotailed['z_stat']:.4f}")
    print(f"   - P-value (two-tailed): {sentence2_twotailed['p_value']:.6f} ({sentence2_twotailed['sig_indicator']})")
    print(f"   - P-value (one-tailed, greater): {sentence2_onetailed['p_value']:.6f} ({sentence2_onetailed['sig_indicator']})")
    print(f"   - Statistically significant (two-tailed): {sentence2_twotailed['significant']}")
    print(f"   - Statistically greater (one-tailed): {sentence2_onetailed['significant']}")
    
    print("\n" + "=" * 80)
    print("END OF STATISTICAL ANALYSIS")
    print("=" * 80)
    print("\nNotes:")
    print("- All uncertainties represent 95% confidence intervals")
    print("- Z-test approach follows Type B uncertainty methodology")
    print("- Reference: Lowhorn et al. (2011) J. Mater. Res., Vol. 26, No. 15")
    print("- Methodology from John Lu's email (September 17, 2025)")
    print("=" * 80)
