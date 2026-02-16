import pandas as pd
import numpy as np
import os
import sys
from scipy import stats
from scipy.stats import friedmanchisquare
import itertools

def get_gmean_results(base_path, abtype, window_length, abnormal_parameter, algorithms_csv_names):
    """
    Reads a specific CSV file and returns a dictionary mapping algorithm names
    to their G-Mean values at Captured Time = 1000.
    """
    results = {alg: 0.0 for alg in algorithms_csv_names}
    
    dir_name = f"abtype{abtype}"
    file_name = f"abtype{abtype}_w{window_length}_t{round(abnormal_parameter, 4)}.csv"
    file_path = os.path.join(base_path, dir_name, file_name)

    if not os.path.exists(file_path):
        print(f"Warning: File not found at {file_path}. Returning zeros.", file=sys.stderr)
        return results

    try:
        df = pd.read_csv(file_path)
        df_filtered = df[df['Captured Time'] == 1000]

        for alg_csv_name in algorithms_csv_names:
            alg_data = df_filtered[df_filtered['Algorithm'] == alg_csv_name]
            if not alg_data.empty:
                g_mean = alg_data['G-Mean'].iloc[0]
                results[alg_csv_name] = g_mean

    except Exception as e:
        print(f"Error processing {file_path}: {e}", file=sys.stderr)
    
    return results


def collect_all_gmean_data():
    """
    Collects G-Mean data for all abnormal patterns with their respective
    window lengths and parameter values.
    Returns a DataFrame with scenarios as rows and algorithms as columns.
    """
    BASE_PATH = r'C:\Users\pault\Documents\3. AI and Machine Learning\2. Deep Learning\1c. App\Projects\CCPR_project\results'
    
    LATEX_TO_CSV_ALGO_MAP = {
        "PA": "PA", 
        "PA-I": "PA1", 
        "PA-II": "PA2", 
        "CSPA_1": "PA1_Csplit", 
        "CSPA_2": "PA2_Csplit",
        "CSPA-ℓ1": "PA_L1", 
        "CSPA-ℓ2": "PA_L2", 
        "CSPA_1-ℓ^I": "PA1_L1", 
        "CSPA_1-ℓ^II": "PA1_L2",
        "CSPA_2-ℓ^I": "PA2_L1", 
        "CSPA_2-ℓ^II": "PA2_L2",
    }
    
    # Configuration for all patterns
    PATTERN_CONFIGS = [
        {"pattern_name": "Up-trend", "abtype": 1, "param_val": 0.051, "window_length": 20},
        {"pattern_name": "Down-trend", "abtype": 2, "param_val": 0.051, "window_length": 20},
        {"pattern_name": "Up-shift", "abtype": 3, "param_val": 0.236, "window_length": 20},
        {"pattern_name": "Down-shift", "abtype": 4, "param_val": 0.236, "window_length": 20},
        {"pattern_name": "Systematic", "abtype": 5, "param_val": 0.236, "window_length": 20},
        {"pattern_name": "Cyclic", "abtype": 6, "param_val": 0.097, "window_length": 20},
        # {"pattern_name": "Stratification", "abtype": 7, "param_val": 0.35, "window_length": 20},
    ]
    
    csv_algo_names = list(LATEX_TO_CSV_ALGO_MAP.values())
    latex_algo_names = list(LATEX_TO_CSV_ALGO_MAP.keys())
    
    data_rows = []
    
    for config in PATTERN_CONFIGS:
        gmean_data = get_gmean_results(
            BASE_PATH, 
            config['abtype'], 
            config['window_length'], 
            config['param_val'], 
            csv_algo_names
        )
        
        row = {'Pattern': config['pattern_name']}
        for latex_name in latex_algo_names:
            csv_name = LATEX_TO_CSV_ALGO_MAP[latex_name]
            row[latex_name] = gmean_data.get(csv_name, 0.0)
        
        data_rows.append(row)
    
    df = pd.DataFrame(data_rows)
    df.set_index('Pattern', inplace=True)
    
    return df, latex_algo_names


def compute_rankings(df):
    """
    Compute rankings for each row (pattern/dataset).
    Ranks are assigned such that higher G-Mean gets better (lower) rank.
    Returns DataFrame with rankings.
    """
    rankings = df.rank(axis=1, ascending=False, method='average')
    return rankings


def friedman_test(rankings_df):
    """
    Performs the Friedman test on the rankings.
    
    Parameters:
    - rankings_df: DataFrame with rankings (rows=datasets/patterns, cols=algorithms)
    
    Returns:
    - chi_square_f: Chi-square statistic
    - F_f: F-statistic
    - p_value: p-value from scipy
    - reject_h0: Boolean indicating if H0 should be rejected
    """
    N = rankings_df.shape[0]  # Number of datasets/patterns
    k = rankings_df.shape[1]  # Number of algorithms
    
    # Mean ranking for each algorithm
    mean_rankings = rankings_df.mean(axis=0)
    
    # Calculate chi-square statistic
    sum_Rj_squared = np.sum(mean_rankings ** 2)
    chi_square_f = (12 * N / (k * (k + 1))) * (sum_Rj_squared - (k * (k + 1)**2 / 4))
    
    # Calculate F statistic
    F_f = ((N - 1) * chi_square_f) / (N * (k - 1) - chi_square_f)
    
    # Degrees of freedom
    df1 = k - 1
    df2 = (k - 1) * (N - 1)
    
    # Critical value and p-value
    p_value = 1 - stats.f.cdf(F_f, df1, df2)
    alpha = 0.05
    critical_value = stats.f.ppf(1 - alpha, df1, df2)
    
    reject_h0 = F_f > critical_value
    
    # Also use scipy's friedmanchisquare for verification
    data_for_scipy = [rankings_df[col].values for col in rankings_df.columns]
    scipy_stat, scipy_pvalue = friedmanchisquare(*data_for_scipy)
    
    print("="*80)
    print("FRIEDMAN TEST RESULTS")
    print("="*80)
    print(f"Number of datasets/patterns (N): {N}")
    print(f"Number of algorithms (k): {k}")
    print(f"\nMean Rankings:")
    for algo, rank in mean_rankings.items():
        print(f"  {algo}: {rank:.4f}")
    print(f"\nChi-square statistic (χ²_F): {chi_square_f:.4f}")
    print(f"F statistic (F_F): {F_f:.4f}")
    print(f"Degrees of freedom: df1={df1}, df2={df2}")
    print(f"Critical value (α=0.05): {critical_value:.4f}")
    print(f"P-value: {p_value:.6f}")
    print(f"\nScipy verification - Chi-square: {scipy_stat:.4f}, P-value: {scipy_pvalue:.6f}")
    print(f"\nDecision: {'REJECT H0' if reject_h0 else 'FAIL TO REJECT H0'}")
    
    if reject_h0:
        print("→ There is a significant difference among the algorithms.")
        print("→ Proceed with Nemenyi post-hoc test.")
    else:
        print("→ No significant difference found among the algorithms.")
    print("="*80)
    print()
    
    return chi_square_f, F_f, p_value, reject_h0, mean_rankings


def nemenyi_test(rankings_df, mean_rankings, alpha=0.05):
    """
    Performs the Nemenyi post-hoc test.
    
    Parameters:
    - rankings_df: DataFrame with rankings
    - mean_rankings: Series with mean rankings for each algorithm
    - alpha: Significance level (default 0.05)
    
    Returns:
    - cd: Critical difference
    - pairwise_differences: DataFrame with all pairwise differences
    - significant_pairs: List of significantly different pairs
    """
    N = rankings_df.shape[0]  # Number of datasets
    k = rankings_df.shape[1]  # Number of algorithms
    
    # Critical values for Nemenyi test (q_alpha values)
    # These are approximate values from Nemenyi tables for α=0.05
    q_alpha_values = {
        2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850,
        7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164, 11: 3.219, 12: 3.268,
        13: 3.313, 14: 3.354, 15: 3.391
    }
    
    if k in q_alpha_values:
        q_alpha = q_alpha_values[k]
    else:
        # Approximation for k > 12 using Studentized range distribution
        q_alpha = stats.studentized_range.ppf(1 - alpha, k, np.inf) / np.sqrt(2)
    
    # Calculate critical difference
    std_error = np.sqrt((k * (k + 1)) / (6 * N))
    cd = q_alpha * std_error
    
    print("="*80)
    print("NEMENYI POST-HOC TEST RESULTS")
    print("="*80)
    print(f"Number of datasets (N): {N}")
    print(f"Number of algorithms (k): {k}")
    print(f"q_alpha (α={alpha}): {q_alpha:.4f}")
    print(f"Critical Difference (CD): {cd:.4f}")
    print()
    
    # Compute all pairwise differences
    algorithms = mean_rankings.index.tolist()
    pairwise_results = []
    significant_pairs = []
    
    for i, algo1 in enumerate(algorithms):
        for j, algo2 in enumerate(algorithms):
            if i < j:  # Only upper triangle
                diff = abs(mean_rankings[algo1] - mean_rankings[algo2])
                q_stat = diff / std_error

                try:
                    # Convert to studentized range statistic to obtain p-value
                    p_value = stats.studentized_range.sf(q_stat * np.sqrt(2), k, np.inf)
                except Exception:
                    p_value = np.nan

                is_significant = p_value <= alpha if not np.isnan(p_value) else diff >= cd
                
                # Determine which is better (lower rank is better)
                if mean_rankings[algo1] < mean_rankings[algo2]:
                    better = algo1
                else:
                    better = algo2
                
                pairwise_results.append({
                    'Algorithm 1': algo1,
                    'Algorithm 2': algo2,
                    'Rank 1': mean_rankings[algo1],
                    'Rank 2': mean_rankings[algo2],
                    'Difference': diff,
                    'q_stat': q_stat,
                    'p-value': p_value,
                    'Significant': is_significant,
                    'Better': better if is_significant else 'No difference'
                })
                
                if is_significant:
                    significant_pairs.append((algo1, algo2, better, diff, p_value))
    
    pairwise_df = pd.DataFrame(pairwise_results)
    
    print("Pairwise Comparisons:")
    print("-" * 80)
    for _, row in pairwise_df.iterrows():
        sig_marker = "***" if row['Significant'] else ""
        p_val = row['p-value']
        p_str = f"{p_val:.6f}" if not np.isnan(p_val) else "nan"
        print(f"{row['Algorithm 1']:15s} vs {row['Algorithm 2']:15s} | "
              f"Diff: {row['Difference']:6.4f} | q: {row['q_stat']:6.4f} | "
              f"p-value: {p_str:>10} | "
              f"{sig_marker:3s} | Better: {row['Better']}")
    
    print()
    print("="*80)
    print("SIGNIFICANT DIFFERENCES (α=0.05):")
    print("="*80)
    
    if significant_pairs:
        for algo1, algo2, better, diff, p_val in significant_pairs:
            print(f"• {better} is significantly better than "
                  f"{'[' + algo1 + ']' if better != algo1 else '[' + algo2 + ']'} "
                  f"(difference: {diff:.4f}, p-value: {p_val:.6f})")
    else:
        print("No significant differences found between any pair of algorithms.")
    
    print("="*80)
    print()
    
    return cd, pairwise_df, significant_pairs


def rank_algorithms(mean_rankings):
    """
    Ranks algorithms from best to worst based on mean rankings.
    Lower rank = better performance.
    """
    sorted_rankings = mean_rankings.sort_values()
    
    print("="*80)
    print("OVERALL ALGORITHM RANKING (Best to Worst)")
    print("="*80)
    for rank, (algo, mean_rank) in enumerate(sorted_rankings.items(), 1):
        print(f"{rank}. {algo:20s} - Mean Rank: {mean_rank:.4f}")
    print("="*80)
    print()
    
    return sorted_rankings


def generate_statistical_report():
    """
    Main function to generate complete statistical analysis report.
    """
    print("\n" + "="*80)
    print("STATISTICAL ANALYSIS: FRIEDMAN AND NEMENYI TESTS")
    print("Algorithm Comparison Across Abnormal Patterns")
    print("="*80)
    print()
    
    # Step 1: Collect data
    print("Step 1: Collecting G-Mean data for all patterns...")
    gmean_df, algorithm_names = collect_all_gmean_data()
    
    print("\nG-Mean Values:")
    print(gmean_df.to_string())
    print()
    
    # Step 2: Compute rankings
    print("Step 2: Computing rankings (lower rank = better performance)...")
    rankings_df = compute_rankings(gmean_df)
    
    print("\nRankings:")
    print(rankings_df.to_string())
    print()
    
    # Step 3: Friedman test
    print("Step 3: Performing Friedman test...")
    chi_sq, F_stat, p_val, reject_h0, mean_ranks = friedman_test(rankings_df)
    
    # Step 4: Nemenyi test (if H0 rejected)
    if reject_h0:
        print("Step 4: Performing Nemenyi post-hoc test...")
        cd, pairwise_df, sig_pairs = nemenyi_test(rankings_df, mean_ranks)
        
        # Step 5: Final ranking
        print("Step 5: Overall algorithm ranking...")
        final_ranking = rank_algorithms(mean_ranks)
        
        return {
            'gmean_data': gmean_df,
            'rankings': rankings_df,
            'mean_rankings': mean_ranks,
            'friedman': {'chi_square': chi_sq, 'F_stat': F_stat, 'p_value': p_val},
            'nemenyi': {'cd': cd, 'pairwise': pairwise_df, 'significant_pairs': sig_pairs},
            'final_ranking': final_ranking
        }
    else:
        print("Nemenyi test not needed (H0 not rejected).")
        final_ranking = rank_algorithms(mean_ranks)
        
        return {
            'gmean_data': gmean_df,
            'rankings': rankings_df,
            'mean_rankings': mean_ranks,
            'friedman': {'chi_square': chi_sq, 'F_stat': F_stat, 'p_value': p_val},
            'final_ranking': final_ranking
        }


if __name__ == '__main__':
    results = generate_statistical_report()
    
    # Optionally save results to CSV
    print("\nSaving results to CSV files...")
    results['gmean_data'].to_csv('gmean_values.csv')
    results['rankings'].to_csv('rankings.csv')
    results['mean_rankings'].to_csv('mean_rankings.csv')
    
    if 'nemenyi' in results:
        results['nemenyi']['pairwise'].to_csv('nemenyi_pairwise_comparisons.csv', index=False)
    
    print("Analysis complete! Results saved to CSV files.")