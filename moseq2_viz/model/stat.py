import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from itertools import combinations
from statsmodels.stats.multitest import multipletests


def get_session_mean_df(df, statistic="usage", max_syllable=40, type='syllable'):
    """
    Compute a given mean syllable statistic grouped by groups and UUIDs.

    Parameters
    ----------
    df (pd.DataFrame): Output of moseq2_viz.model.compute_behavioral_statistics().
    nrows -> correspond to max_syllable * n_uuids,
    ncols -> 26 (including group, uuid, syllable, usage, duration and syllable key).
    statistic (str): statistic to compute mean for, (any of the columns in input df);
    for example: 'usage', 'duration', 'velocity_2d_mm', etc.
    max_syllable (int): Maximum number of syllables to include

    Returns
    -------
    df_pivot (pd.DataFrame): Mean syllable statistic per session; shape=(n_sessions, max_syllable)
    """

    if type == 'syllable':
        df_pivot = (
            df[df.syllable < max_syllable]
                .pivot_table(index=["group", "uuid"], columns="syllable", values=statistic)
                .replace(np.nan, 0)
        )
    else:
        df_pivot = df.groupby(['uuid', 'group'], as_index=False).mean()

    return df_pivot


def bootstrap_me(usages, n_iters=10000):
    """
    Bootstraps the inputted stat data using random sampling with replacement.

    Parameters
    ----------
    usages (np.array): Data to bootstrap; shape = (n_mice, n_syllables)
    n_iters (int): Number of samples to return.

    Returns
    -------
    boots (np.array): Bootstrapped input array of shape: (n_iters, n_syllables)
    """

    n_mice = usages.shape[0]
    return np.nanmean(usages[np.random.choice(n_mice, size=(n_mice, n_iters))], axis=0)


def ztest_vect(d1, d2):
    """
    Performs a z-test on a pair of bootstrapped syllable statistics.

    Parameters
    ----------
    d1 (np.array): bootstrapped syllable stat array from group 1; shape = (n_boots, n_syllables)
    d2 (np.array): bootstrapped syllable stat array from group 2; shape = (n_boots, n_syllables)

    Returns
    -------
    p-values (np.array): array of computed p-values of len == n_syllables.
    """

    mu1 = d1.mean(0)
    mu2 = d2.mean(0)
    std1 = d1.std(0)
    std2 = d2.std(0)
    std = np.sqrt(std1 ** 2 + std2 ** 2)
    return np.minimum(1.0, 2 * stats.norm.cdf(-np.abs(mu1 - mu2) / std))


def bootstrap_group_means(df, group1, group2, statistic="usage", max_syllable=40, type='syllable'):
    """

    Parameters
    ----------
    df (pd.DataFrame): Output of moseq2_viz.model.compute_behavioral_statistics().
    nrows -> correspond to max_syllable * n_uuids,
    ncols -> 26 (including group, uuid, syllable, usage, duration and syllable key).
    group1 (str): Name of group 1 to compare.
    group2 (str): Name of group 2 to compare.
    statistic (str): Syllable statistic to compute bootstrap means for.
    max_syllable (int): Maximum syllables to compute mean statistic for.

    Returns
    -------
    boots (dict): dictionary of group name (keys) paired with their bootstrapped statistics numpy array.
    """

    # get separated group variables
    group_stat = get_session_mean_df(df, statistic, max_syllable, type=type)

    groups = (group1, group2)
    if type == 'syllable':
        usages = {k: group_stat.loc[k].values for k in groups}
    else:
        usages = {k: np.reshape(group_stat[group_stat['group'] == k][statistic].values, (-1,1)) for k in groups}

    boots = {k: bootstrap_me(v) for k, v in usages.items()}

    return boots


def get_tie_correction(x, N_m):
    """
    Kruskal-Wallis helper function that assigns tied rank values the average of the ranks
    they would have received if they had not been tied.

    Parameters
    ----------
    x (pd.Series): syllable usages for a single session.
    N_m (int): Number of total sessions.

    Returns
    -------
    corrected_rank (float): average of the inputted tied rank
    """

    vc = x.value_counts()
    tie_sum = 0
    if (vc > 1).any():
        tie_sum += np.sum(vc[vc != 1] ** 3 - vc[vc != 1])
    return tie_sum / (12.0 * (N_m - 1))


def run_manual_KW_test(
        df_usage,
        merged_usages_all,
        num_groups,
        n_per_group,
        cum_group_idx,
        n_perm=10000,
        seed=0,
):
    """

    Runs a manual KW test: ranks the syllables, computes the square sums for each group, computes the H-statistic,
    and finally ensures that the results agree with the scipy.stats implementation.

    Parameters
    ----------
    df_usage (pd.DataFrame): DataFrame containing only pre-computed syllable stats. shape = (N_m, n_syllables)
    merged_usages_all (np.array): numpy array format of the df_usage DataFrame.
    num_groups (int): Number of unique groups
    n_per_group (list): list of value counts for sessions per group. len == num_groups.
    cum_group_idx (list): list of indices for different groups. len == num_groups + 1.
    n_perm (int): Number of permuted samples to generate.
    seed (int): Random seed used to initialize the pseudo-random number generator.

    Returns
    -------
    h_all (np.array): Array of H-stats computed for given n_syllables; shape = (n_perms, N_s)
    real_ranks (np.array): Array of syllable ranks, shape = (N_m, n_syllables)
    X_ties (np.array): 1-D list of tied ranks, where if value > 0, then rank is tied. len(X_ties) = n_syllables
    """

    N_m, N_s = merged_usages_all.shape

    # create random index array n_perm times
    rnd = np.random.RandomState(seed=seed)
    perm = rnd.rand(n_perm, N_m).argsort(-1)

    # get degrees of freedom
    dof = num_groups - 1

    real_ranks = np.apply_along_axis(stats.rankdata, 0, merged_usages_all)
    X_ties = df_usage.apply(get_tie_correction, 0, N_m=N_m).values
    KW_tie_correct = np.apply_along_axis(stats.tiecorrect, 0, real_ranks)

    # rank data
    perm_ranks = real_ranks[perm]

    # get square of sums for each group
    ssbn = np.zeros((n_perm, N_s))
    for i in range(num_groups):
        ssbn += (
                perm_ranks[:, cum_group_idx[i]: cum_group_idx[i + 1]].sum(1) ** 2
                / n_per_group[i]
        )

    # h-statistic
    h_all = 12.0 / (N_m * (N_m + 1)) * ssbn - 3 * (N_m + 1)
    h_all /= KW_tie_correct
    p_vals = stats.chi2.sf(h_all, df=dof)

    # check that results agree
    p_i = np.random.randint(n_perm)
    s_i = np.random.randint(N_s)
    kr = stats.kruskal(
        *np.array_split(
            merged_usages_all[perm[p_i, :], s_i], np.cumsum(n_per_group[:-1])
        )
    )
    assert (kr.statistic == h_all[p_i, s_i]) & (
            kr.pvalue == p_vals[p_i, s_i]
    ), "manual KW is incorrect"

    return h_all, real_ranks, X_ties


def plot_H_stat_significance(df_k_real, h_all, N_s):
    """
    Plots the assigned H-statistic for each syllable computed via manual KW test.
    Syllables with H-statistic > critical H-value are considered significant.

    Parameters
    ----------
    df_k_real (pd.DataFrame): DataFrame containing columns= [KW (pvalue), H-stats (statistic), is_sig]
    h_all (np.array): Array of H-stats computed for given n_syllables; shape = (n_perms, N_s)
    N_s (int): Number of syllables to plot

    Returns
    -------
    fig (pyplot figure): plotted H-stats plot
    ax (pyplot axis): plotted H-stats axis
    """

    fig, ax = plt.subplots(figsize=(9, 6))

    pcts = np.percentile(h_all, [2.5, 97.5], axis=0)

    ax.fill_between(np.arange(N_s), pcts[0, :], pcts[1, :], alpha=0.2, color="k")

    ax.scatter(
        np.arange(N_s)[~df_k_real.is_sig],
        df_k_real.statistic[~df_k_real.is_sig],
        label="Not significant",
    )
    ax.scatter(
        np.arange(N_s)[df_k_real.is_sig],
        df_k_real.statistic[df_k_real.is_sig],
        label="Significant Syllable",
    )

    ax.set_xlabel("Syllable")
    ax.set_ylabel(f"$H$")
    ax.legend()
    sns.despine()

    return fig, ax


def run_kruskal(
        df,
        statistic="usage",
        max_syllable=40,
        n_perm=10000,
        seed=42,
        thresh=0.05,
        mc_method="fdr_bh",
):
    """
    Runs Kruskal-Wallis Hypothesis test and Dunn's posthoc multiple comparisons test for a
    permuted randomly sampled (with replacement) syllable statistic. If len(unique_groups) > 2, then
    function will return the signficant syllables between all permutations of all the model groups.

    Parameters
    ----------
    df (pd.DataFrame): Output of moseq2_viz.model.compute_behavioral_statistics().
     nrows -> correspond to max_syllable * n_uuids,
    ncols -> 26 (including group, uuid, syllable, usage, duration and syllable key).
    stat (str): statistic to compute mean for, (any of the columns in input df);
    for example: 'usage', 'duration', 'velocity_2d_mm', etc.
    max_syllable (int): Maximum number of syllables to include.
    n_perm (int): Number of permuted samples to generate.
    seed (int): Random seed used to initialize the pseudo-random number generator.
    thresh (float): Alpha threshold to consider syllable significant.
    mc_method (str): Multiple Corrections method to use.
    Options can be found here: https://www.statsmodels.org/dev/generated/statsmodels.stats.multitest.multipletests.html

    Returns
    -------
    df_k_real (pd.DataFrame): DataFrame of KW test results.
    n_rows=max_syllable, n_cols=['statistic', 'pvalue', 'emp_fdr', 'is_sig']
    dunn_results_df (pd.DataFrame): DataFrame of Dunn's test results for permuted group pairs.
    n_rows=(max_syllable*n_group_pairs), n_cols=['syllable', 'variable_0', 'variable_1', 'value']
    intersect_sig_syllables (dict): dictionary containing intersecting significant syllables between
    KW and Dunn's tests. Keys = ('group1', 'group2') -> Value: array of significant syllables.
    """

    rnd = np.random.RandomState(seed=seed)

    # get mean grouped data
    grouped_data = get_session_mean_df(df, statistic, max_syllable).reset_index()

    # KW Constants
    vc = grouped_data.group.value_counts().loc[grouped_data.group.unique()]
    n_per_group = vc.values
    group_names = vc.index

    cum_group_idx = np.insert(np.cumsum(n_per_group), 0, 0)
    num_groups = len(group_names)

    df_only_usage = grouped_data[range(max_syllable)]
    merged_usages_all = df_only_usage.values

    N_m, N_s = merged_usages_all.shape

    # Run KW and return H-stats
    h_all, real_ranks, X_ties = run_manual_KW_test(
        df_usage=df_only_usage,
        merged_usages_all=merged_usages_all,
        num_groups=num_groups,
        n_per_group=n_per_group,
        cum_group_idx=cum_group_idx,
        n_perm=n_perm,
        seed=seed,
    )

    df_k_real = pd.DataFrame(
        [
            stats.kruskal(
                *np.array_split(merged_usages_all[:, s_i], np.cumsum(n_per_group[:-1]))
            )
            for s_i in range(N_s)
        ]
    )

    df_k_real["p_adj"] = multipletests(
        ((h_all > df_k_real.statistic.values).sum(0) + 1) / n_perm,
        alpha=thresh,
        method=mc_method,
    )[1]

    df_k_real["is_sig"] = df_k_real["p_adj"] <= thresh
    print(
        f"Found {df_k_real['is_sig'].sum()} syllables that pass threshold {thresh} with {mc_method}"
    )

    # Run Dunn's z-test statistics
    (
        null_zs_within_group,
        real_zs_within_group,
    ) = dunns_z_test_permute_within_group_pairs(
        grouped_data, vc, real_ranks, X_ties, N_m, group_names, rnd, n_perm
    )

    # Compute p-values from Dunn's z-score statistics
    df_pair_corrected_pvalues, _ = compute_pvalues_for_group_pairs(
        real_zs_within_group,
        null_zs_within_group,
        df_k_real,
        group_names,
        n_perm,
        thresh,
        mc_method,
    )

    # combine Dunn's test results into single DataFrame
    df_z = pd.DataFrame(real_zs_within_group)
    df_z.index = df_z.index.set_names("syllable")
    dunn_results_df = df_z.reset_index().melt(id_vars="syllable")

    # take the intersection of KW and Dunn's tests
    print("permutation within group-pairs (2 groups)")
    print("intersection of Kruskal-Wallis and Dunn's tests")

    # Get intersecting significant syllables between
    intersect_sig_syllables = {}
    for pair in df_pair_corrected_pvalues.columns.tolist():
        intersect_sig_syllables[pair] = np.where(
            (df_pair_corrected_pvalues[pair] < thresh) & (df_k_real.is_sig)
        )[0]
        print(pair, len(intersect_sig_syllables[pair]), intersect_sig_syllables[pair])

    return df_k_real, dunn_results_df, intersect_sig_syllables


def compute_pvalues_for_group_pairs(
        real_zs_within_group,
        null_zs,
        df_k_real,
        group_names,
        n_perm=10000,
        thresh=0.05,
        mc_method="fdr_bh",
):
    """
    Adjusts the p-values from Dunn's z-test statistics and computes the resulting significant syllables with the
    adjusted p-values.

    Parameters
    ----------
    real_zs_within_group (dict): dict of group pair keys paired with vector of Dunn's z-test statistics
    null_zs  (dict): dict of group pair keys paired with vector of Dunn's z-test statistics of the null hypothesis.
    df_k_real (pd.DataFrame): DataFrame of KW test results.
    n_rows=max_syllable, n_cols=['statistic', 'pvalue', 'emp_fdr', 'is_sig']
    group_names (pd.Index): Index list of unique group names.
    n_perm (int): Number of permuted samples to generate.
    thresh (float): Alpha threshold to consider syllable significant.
    mc_method (str): Multiple Corrections method to use.

    Returns
    -------
    df_pval_corrected (pd.DataFrame): DataFrame containing Dunn's test results with corrected p-values.
    significant_syllables (list): List of corrected KW significant syllables (syllables with p-values < thresh)
    """

    # do empirical p-val calculation for all group permutation
    print(f"Permutation across all {len(group_names)} groups")
    print(f"significant syllables for each pair with FDR < {thresh}")

    p_vals_allperm = {}
    for pair in combinations(group_names, 2):
        p_vals_allperm[pair] = ((null_zs[pair] > real_zs_within_group[pair]).sum(0) + 1) / n_perm

    # summarize into df
    df_pval = pd.DataFrame(p_vals_allperm)

    correct_p = lambda x: multipletests(x, alpha=thresh, method=mc_method)[1]
    df_pval_corrected = df_pval.apply(correct_p, axis=1, result_type="broadcast")

    return df_pval_corrected, ((df_pval_corrected[df_k_real.is_sig] < thresh).sum(0))


def dunns_z_test_permute_within_group_pairs(
        df_usage, vc, real_ranks, X_ties, N_m, group_names, rnd, n_perm
):
    """
    Runs Dunn's z-test statistic on combinations of all group pairs, handling pre-computed tied ranks.

    Parameters
    ----------
    df_usage (pd.DataFrame): DataFrame containing only pre-computed syllable stats. shape = (N_m, n_syllables)
    vc (pd.Series): value counts of sessions in each group.
    real_ranks (np.array): Array of syllable ranks, shape = (N_m, n_syllables)
    X_ties (np.array): 1-D list of tied ranks, where if value > 0, then rank is tied. len(X_ties) = n_syllables
    N_m (int): Number of sessions.
    group_names (pd.Index): Index list of unique group names.
    rnd (np.random.RandomState): Pseudo-random number generator.
    n_perm (int): Number of permuted samples to generate.

    Returns
    -------
    null_zs_within_group (dict): dict of group pair keys paired with vector of Dunn's z-test statistics of the null hypothesis.
    real_zs_within_group (dict): dict of group pair keys paired with vector of Dunn's z-test statistics
    """

    null_zs_within_group = {}
    real_zs_within_group = {}

    A = N_m * (N_m + 1.0) / 12.0

    for (i_n, j_n) in combinations(group_names, 2):
        is_i = df_usage.group == i_n
        is_j = df_usage.group == j_n

        n_mice = is_i.sum() + is_j.sum()

        ranks_perm = real_ranks[(is_i | is_j)][rnd.rand(n_perm, n_mice).argsort(-1)]
        diff = np.abs(
            ranks_perm[:, : is_i.sum(), :].mean(1)
            - ranks_perm[:, is_i.sum():, :].mean(1)
        )
        B = 1.0 / vc.loc[i_n] + 1.0 / vc.loc[j_n]

        # also do for real data
        group_ranks = real_ranks[(is_i | is_j)]
        real_diff = np.abs(
            group_ranks[: is_i.sum(), :].mean(0) - group_ranks[is_i.sum():, :].mean(0)
        )

        # add to dict
        pair = (i_n, j_n)
        null_zs_within_group[pair] = diff / np.sqrt((A - X_ties) * B)
        real_zs_within_group[pair] = real_diff / np.sqrt((A - X_ties) * B)

    return null_zs_within_group, real_zs_within_group

def run_pairwise_stats(df, group1, group2, test_type="mw", **kwargs):
    """Wrapper for staistical test

    Parameters
    ----------
    df (pd.DataFrame): Output of moseq2_viz.model.compute_behavioral_statistics().
    group1 (str): Name of first group
    group2 (str): Name of second group
    test_type (str): one of ["mw", "z_test", "t_test"], specifying which type of statistical test

    Returns:
        df_pvals (pd.DataFrame): Dataframe listing the p-values and which syllables are significant
    """
    test_types = ["mw", "z_test", "t_test"]
    if test_type not in test_types:
        raise ValueError(f"`test_type` must one of {test_types}")
    if test_type == "mw":
        return mann_whitney(df, group1, group2, **kwargs)
    elif test_type == "z_test":
        return ztest(df, group1, group2, **kwargs)
    elif test_type == "t_test":
        return ttest(df, group1, group2, **kwargs)

def run_pairwise_scalar_stats(scalar_df, group1, group2, test_type='t_test', **kwargs):
    """

    Parameters
    ----------
    scalar_df
    group1
    group2
    test_type
    kwargs

    Returns
    -------

    """

    test_types = ["mw", "z_test", "t_test"]
    if test_type not in test_types:
        raise ValueError(f"`test_type` must one of {test_types}")
    if test_type == "mw":
        return mann_whitney(scalar_df, group1, group2, type='scalar', **kwargs)
    elif test_type == "z_test":
        return ztest(scalar_df, group1, group2, type='scalar', **kwargs)
    elif test_type == "t_test":
        return ttest(scalar_df, group1, group2, type='scalar', **kwargs)

def mann_whitney(df, group1, group2, statistic="usage", max_syllable=40, type='syllable', **kwargs):
    """
    Runs a Mann-Whitney hypothesis test on two given groups to find significant syllables.
    Also runs multiple corrections test to find syllables to exclude.

    Parameters
    ----------
    df (pd.DataFrame): Output of moseq2_viz.model.compute_behavioral_statistics().
    nrows -> correspond to max_syllable * n_uuids,
    ncols -> 26 (including group, uuid, syllable, usage, duration and syllable key).
    group1 (str): Name of first group
    group2 (str): Name of second group
    statistic (str): Name of statistic to compute z-test on.
    max_syllable (int): Maximum number of syllables to include
    thresh (float): Alpha threshold to consider syllable significant.
    mc_method (str): Multiple Corrections method to use.

    Returns
    -------
    df_mw_real (pd.DataFrame): DataFrame containing Mann-Whitney U corrected results.
    shape = (max_syllable, 3), columns = ['statistic', 'pvalue', 'is_sig']
    exclude_sylls (list): list of syllables that were excluded via multiple comparisons test.
    """
    # get mean grouped data
    grouped_data = get_session_mean_df(df, statistic, max_syllable, type=type).reset_index()

    vc = grouped_data.group.value_counts().loc[[group1, group2]]
    n_per_group = vc.values

    if type == 'syllable':
        merged_usages_all = grouped_data[range(max_syllable)].values
    else:
        group1_stat = np.mean(grouped_data[grouped_data['group'] == group1][statistic].values)
        group2_stat = np.mean(grouped_data[grouped_data['group'] == group2][statistic].values)
        merged_usages_all = np.vstack([group1_stat, group2_stat])

    _, N_s = merged_usages_all.shape

    df_mw_real = pd.DataFrame(
        [
            stats.mannwhitneyu(
                *np.array_split(merged_usages_all[:, s_i], np.cumsum(n_per_group[:-1]))
            )
            for s_i in range(N_s)
        ]
    )

    return get_sig_syllables(df_mw_real, type=type, **kwargs)

def ztest(df, group1, group2, statistic="usage", max_syllable=40, type='syllable', **kwargs):
    """
    Computes a z hypothesis test on 2 (bootstrapped) selected groups.
    Also runs multiple corrections test to find syllables to exclude.

    Parameters
    ----------
    df (pd.DataFrame): Output of moseq2_viz.model.compute_behavioral_statistics().
    nrows -> correspond to max_syllable * n_uuids,
    ncols -> 26 (including group, uuid, syllable, usage, duration and syllable key).
    group1 (str): Name of first group
    group2 (str): Name of second group
    statistic (str): Name of statistic to compute z-test on.
    max_syllable (int): Maximum number of syllables to include
    thresh (float): Alpha threshold to consider syllable significant.
    mc_method (str): Multiple Corrections method to use.

    Returns
    -------
    pvals_ztest_boots (np.array): Computed array of p-values
    syllables_to_include (list): List of significant syllables after multiple corrections.
    """
    boots = bootstrap_group_means(df, group1, group2, statistic, max_syllable, type=type)

    # do a ztest on the bootstrap distributions of your 2 conditions
    pvals_ztest_boots = ztest_vect(boots[group1], boots[group2])
    df_z = pd.DataFrame(pvals_ztest_boots, columns=["pvalue"])
    return get_sig_syllables(df_z, type=type, **kwargs)

def normalize(group_in_df):
    """

    Parameters
    ----------
    group_in_df

    Returns
    -------
    group_out_df
    """

    group_out_df = (group_in_df - np.min(group_in_df)) / (np.max(group_in_df) - np.min(group_in_df))
    return group_out_df

def ttest(df, group1, group2, statistic="usage", max_syllable=40, type='syllable', **kwargs):
    """
    Computes a t-hypothesis test on 2 selected groups to find significant syllables.
    Also runs multiple corrections test to find syllables to exclude.

    Parameters
    ----------
    df (pd.DataFrame): Output of moseq2_viz.model.compute_behavioral_statistics().
    nrows -> correspond to max_syllable * n_uuids,
    ncols -> 26 (including group, uuid, syllable, usage, duration and syllable key).
    group1 (str): Name of first group
    group2 (str): Name of second group
    statistic (str): Name of statistic to compute t-test on.
    max_syllable (int): Maximum number of syllables to include
    thresh (float): Alpha threshold to consider syllable significant.
    mc_method (str): Multiple Corrections method to use.

    Returns
    -------
    p (np.array): Computed array of p-values
    syllables_to_include (list): List of significant syllables after multiple corrections.
    """
    # get separated group variables
    group_stat = get_session_mean_df(df, statistic, max_syllable, type=type)
    groups = (group1, group2)
    if type == 'syllable':
        usages = {k: group_stat.loc[k].values for k in groups}
    else:
        usages = {k: normalize(group_stat[group_stat['group'] == k][statistic].values) for k in groups}

    # run t-test
    df_t = pd.DataFrame(
        stats.ttest_ind(usages[group1], usages[group2]), index=["t", "pvalue"]
    ).T

    return get_sig_syllables(df_t, type=type, **kwargs)

def get_sig_syllables(df_pvals, thresh=0.05, mc_method="fdr_bh", type='syllable'):
    """
    Runs multiple p-value comparisons test given a set alpha Threshold, and multiple corrections method.

    Parameters
    ----------
    df_pvals (pd.DataFrame): dataframe listing raw p-values

    Returns
    -------
    df_pvals (pd.DataFrame): updated dataframe listing adjusted p-values
    """
    df_pvals["p_adj"] = multipletests(df_pvals.pvalue, alpha=thresh, method=mc_method)[
        1
    ]
    df_pvals["is_sig"] = df_pvals["p_adj"] <= thresh
    n_sig = df_pvals["is_sig"].sum()

    # print list of significant syllables
    sig_sylls = list(df_pvals[df_pvals["is_sig"] == True].index)

    if type == 'syllable':
        print(f"Found {n_sig} syllables that pass threshold {thresh} with {mc_method}")
        print(sig_sylls)
    else:
        print(f"Found {n_sig} groups that pass threshold {thresh} with {mc_method}")

    return df_pvals
