"""
1. 在逻辑回归模型上（Logistic Regression），穷举训练一次所有特征组合，每个组合计算 f2_0, precision0, recall0, auc, ap。
2. 训练完成后，自动执行多方法特征选择，并对每个指标输出排名前10的组合
3. 所有可调参数集中在文件开头
4. 输出文件有两个，“new_feature_combination_results_all.csv”是黑箱模型的训练输出，另一个文件为根据训练结果，分别在不同的单一评价指标下计算得到的最佳top10组合
"""

import pandas as pd
import numpy as np
from itertools import combinations
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LassoCV
from sklearn.metrics import (precision_recall_fscore_support,
                             roc_auc_score, average_precision_score)
from scipy.stats import norm
from sklearn.feature_selection import mutual_info_classif
import time
import warnings
from multiprocessing import Pool, cpu_count

warnings.filterwarnings('ignore')

# ---------- 文件路径 ----------
TRAIN_CSV = r"D:\PythonProject\sequence_pattern_structures - 副本\train_and_test\window_size_4_3\full\train_subseq_flattened.csv"
TEST_CSV  = r"D:\PythonProject\sequence_pattern_structures - 副本\train_and_test\window_size_4_3\full\test_subseq_flattened.csv"
OUTPUT_DIR = Path(r"D:\PythonProject\sequence_pattern_structures - 副本\search_all_feature_combinations")
OUTPUT_CSV = OUTPUT_DIR / "new_feature_combination_results_all.csv"
TEMP_FILE  = OUTPUT_DIR / "new_feature_combination_results_all.csv.tmp"

# ---------- 特征定义 ----------
NUMERIC_FEATURES = [
    "hb", "wbc", "plt", "neutrophils",
    "creatinine", "urea", "alt", "ast", "total_bilirubin", "albumin", "ldh", "esr", "ecog"
]
CATEGORICAL_FEATURES = ["admissions", "chemo_active", "radiotherapy_active"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# ---------- 训练参数 ----------
RANDOM_STATE = 42
MAX_ITER = 1000
SOLVER = 'liblinear'
TOL = 1e-4
N_JOBS = 24
CHUNK_SIZE = 100
SAVE_INTERVAL = 500

# ---------- 寻优参数（详细说明）----------
"""
以下参数控制多方法特征选择的严格程度和偏好：
1. FDR_Q (假发现率控制): 
   - 增大 → 允许更多组合通过显著性检验，输出候选增多，但可能引入噪声特征。
   - 减小 → 筛选更严格，仅保留统计上最显著的组合，输出更少但更可靠。
2. WEAK_MODEL_MAX_FEAT (零分布估计用的最大特征数):
   - 增大 → 使用更多小模型估计零分布，可能降低显著性阈值，输出更多组合。
   - 减小 → 仅用极少特征模型估计基线，显著性要求更高，输出更少。
3. MI_BINS (互信息离散化桶数):
   - 增大 → 互信息估计更精细，信息饱和点可能更晚，保留更多特征用于覆盖率计算。
   - 减小 → 离散化粗糙，可能丢失非线性关系，信息饱和点提前，特征覆盖率要求更易满足。
4. MI_INFO_THRESHOLD (信息饱和阈值):
   - 增大 (如0.99) → 需要更多特征才能达到高覆盖率，输出组合必须包含更多重要特征。
   - 减小 (如0.8) → 允许较少特征通过覆盖率筛选，输出组合更简洁但可能遗漏信息。
5. LASSO_CV (交叉验证折数):
   - 增大 → 更稳健的LASSO预测，但计算时间增加，预测值更平滑。
   - 减小 → 预测波动更大，可能过拟合。
6. LASSO_MAX_ITER / LASSO_N_ALPHAS: 通常保持默认，影响LASSO收敛精度和搜索粒度。
7. INFO_COVERAGE_THRESHOLD (信息覆盖率最低要求):
   - 增大 (如0.8) → 要求候选组合必须覆盖80%以上互信息重要特征，输出组合更全面。
   - 减小 (如0.3) → 允许组合缺失较多重要特征，输出更灵活但可能缺失关键信息。
8. PENALTY_LAMBDA (特征数量惩罚系数):
   - 增大 → 强烈惩罚特征多的组合，最终排名倾向于特征少且性能好的组合。
   - 减小 → 几乎不惩罚特征数量，排名主要依据原始指标（如f2_0），容易选出复杂模型。
9. TOP_K: 仅影响输出数量，不影响算法逻辑。
"""
FDR_Q = 0.05                # 假发现率控制
WEAK_MODEL_MAX_FEAT = 2     # 零分布估计用的最大特征数
MI_BINS = 10                # 互信息离散化桶数
MI_INFO_THRESHOLD = 0.95    # 信息饱和阈值
LASSO_CV = 5
LASSO_MAX_ITER = 10000
LASSO_N_ALPHAS = 100
LASSO_RANDOM_STATE = 42
INFO_COVERAGE_THRESHOLD = 0.5   # 信息覆盖率最低要求
PENALTY_LAMBDA = 0.001          # 特征数量惩罚系数
TOP_K = 10                      # 每个指标输出前K个候选组合
# ================================================================

# ---------- 辅助函数 ----------
def f2_score(precision, recall):
    if precision + recall == 0:
        return 0.0
    return (5 * precision * recall) / (4 * precision + recall)

# 全局变量用于多进程
_global_X_train = None
_global_X_test = None
_global_y_train = None
_global_y_test = None
_global_var_to_cols = None
_global_constant_cols = None

def load_global_data():
    train_df = pd.read_csv(TRAIN_CSV, low_memory=False)
    test_df = pd.read_csv(TEST_CSV, low_memory=False)
    label_col = "label"
    feature_cols = [c for c in train_df.columns if c not in [label_col, "sample_id", "source_patient_id"]]
    for col in feature_cols:
        train_df[col] = pd.to_numeric(train_df[col], errors='coerce')
        test_df[col] = pd.to_numeric(test_df[col], errors='coerce')
    var_to_cols = {}
    for var in ALL_FEATURES:
        cols = [c for c in feature_cols if c.startswith(var + '_') or c == var]
        var_to_cols[var] = cols
    X_train = train_df[feature_cols].copy()
    y_train = train_df[label_col].copy()
    X_test = test_df[feature_cols].copy()
    y_test = test_df[label_col].copy()
    for col in X_train.columns:
        med = X_train[col].median()
        X_train[col] = X_train[col].fillna(med)
        X_test[col] = X_test[col].fillna(med)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=feature_cols)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=feature_cols)
    col_variance = X_train_scaled_df.var()
    constant_cols = set(col_variance[col_variance < 1e-12].index)
    return X_train_scaled_df, X_test_scaled_df, y_train, y_test, var_to_cols, constant_cols

def init_worker():
    global _global_X_train, _global_X_test, _global_y_train, _global_y_test, _global_var_to_cols, _global_constant_cols
    (_global_X_train, _global_X_test, _global_y_train, _global_y_test,
     _global_var_to_cols, _global_constant_cols) = load_global_data()

def evaluate_combo(combo):
    selected_cols = []
    for var in combo:
        selected_cols.extend(_global_var_to_cols.get(var, []))
    if not selected_cols:
        return None
    if all(col in _global_constant_cols for col in selected_cols):
        return None
    X_train_sub = _global_X_train[selected_cols]
    X_test_sub = _global_X_test[selected_cols]
    if (X_train_sub.var() == 0).any():
        return None
    try:
        clf = LogisticRegression(max_iter=MAX_ITER, solver=SOLVER,
                                 tol=TOL, random_state=RANDOM_STATE)
        clf.fit(X_train_sub, _global_y_train)
        y_pred = clf.predict(X_test_sub)
        proba = clf.predict_proba(X_test_sub)[:, 1]
        precision, recall, _, _ = precision_recall_fscore_support(
            _global_y_test, y_pred, labels=[0, 1], zero_division=0
        )
        p0, p1 = precision[0], precision[1]
        r0, r1 = recall[0], recall[1]
        f2_0 = f2_score(p0, r0)
        f2_1 = f2_score(p1, r1)
        auc = roc_auc_score(_global_y_test, proba)
        ap = average_precision_score(_global_y_test, proba)
        return {
            "features": ",".join(combo),
            "precision0": p0,
            "recall0": r0,
            "f2_0": f2_0,
            "precision1": p1,
            "recall1": r1,
            "f2_1": f2_1,
            "auc": auc,
            "ap": ap
        }
    except Exception:
        return None

def get_completed_combos(temp_path):
    if not temp_path.exists():
        return set()
    df = pd.read_csv(temp_path)
    if "features" not in df.columns:
        return set()
    return set(df["features"].dropna().tolist())

def train_all_combos():
    print("=" * 60)
    print("开始穷举训练所有特征组合（逻辑回归）")
    print("=" * 60)
    start_time = time.time()
    _, _, _, _, var_to_cols, _ = load_global_data()
    all_combos = []
    for r in range(1, len(ALL_FEATURES) + 1):
        combos = list(combinations(ALL_FEATURES, r))
        all_combos.extend(combos)
        print(f"特征数量 {r}: {len(combos)} 个组合")
    total_combos = len(all_combos)
    print(f"总组合数: {total_combos}")
    completed_set = get_completed_combos(TEMP_FILE)
    print(f"已有临时结果中完成组合数: {len(completed_set)}")
    remaining_combos = [c for c in all_combos if ",".join(c) not in completed_set]
    print(f"剩余待处理组合数: {len(remaining_combos)}")
    if len(remaining_combos) == 0:
        print("所有组合已训练完成，直接加载已有结果。")
        return pd.read_csv(OUTPUT_CSV)
    if TEMP_FILE.exists():
        existing_df = pd.read_csv(TEMP_FILE)
        existing_results = existing_df.to_dict('records')
    else:
        existing_results = []
    print(f"使用 {N_JOBS} 个进程并行训练...")
    with Pool(processes=N_JOBS, initializer=init_worker) as pool:
        new_results = []
        processed = 0
        for res in pool.imap_unordered(evaluate_combo, remaining_combos, chunksize=CHUNK_SIZE):
            if res is not None:
                new_results.append(res)
            processed += 1
            if processed % SAVE_INTERVAL == 0:
                combined = existing_results + new_results
                temp_df = pd.DataFrame(combined)
                temp_df.to_csv(TEMP_FILE, index=False)
                print(f"  进度: {processed}/{len(remaining_combos)} 新组合，有效 {len(new_results)}，已保存临时文件")
    final_results = existing_results + new_results
    result_df = pd.DataFrame(final_results)
    result_df.to_csv(OUTPUT_CSV, index=False)
    if TEMP_FILE.exists():
        TEMP_FILE.unlink()
    print(f"训练完成！成功组合数: {len(final_results)} / {total_combos}")
    print(f"总耗时: {(time.time()-start_time)/60:.2f} 分钟")
    return result_df

# ==================== 寻优函数（输出每个指标前K个候选，包含所有原始评估指标） ====================
def benjamini_hochberg(p_values, q):
    m = len(p_values)
    sorted_idx = np.argsort(p_values)
    sorted_p = p_values[sorted_idx]
    threshold = [(i+1)/m * q for i in range(m)]
    significant = sorted_p <= threshold
    if np.any(significant):
        max_k = np.max(np.where(significant)[0])
        significant[:max_k+1] = True
        significant[max_k+1:] = False
    original = np.zeros(m, dtype=bool)
    original[sorted_idx] = significant
    return original

def info_analysis(df, top_k_features):
    def coverage(feat_str):
        feats = set(feat_str.split(','))
        return len(set(top_k_features) & feats) / len(top_k_features) if top_k_features else 0
    df['info_coverage'] = df['features'].apply(coverage)
    return df

def factor_analysis(df, metric, design):
    y = df[metric].values
    lasso = LassoCV(cv=LASSO_CV, random_state=LASSO_RANDOM_STATE,
                    max_iter=LASSO_MAX_ITER, n_alphas=LASSO_N_ALPHAS)
    lasso.fit(design, y)
    pred = lasso.predict(design)
    return pred

def select_best_features(df):
    print("\n" + "=" * 60)
    print("开始多方法最优特征选择（输出每个指标前{}个候选）".format(TOP_K))
    print("=" * 60)
    
    df['num_feat'] = df['features'].apply(lambda s: len(s.split(',')) if s.strip() else 0)
    
    # 信息论部分
    # 信息论部分（仅使用训练集，无信息泄露）
    print("\n计算特征互信息...")
    train_df = pd.read_csv(TRAIN_CSV, low_memory=False)
    test_df = pd.read_csv(TEST_CSV, low_memory=False)  # 仅用于后续可能的需求，但不参与互信息计算
    label_col = "label"
    feature_cols = [c for c in train_df.columns if c not in [label_col, "sample_id", "source_patient_id"]]

    # 关键：将所有特征列转换为数值类型（处理字符串 '0'、'1' 等）
    for col in feature_cols:
        train_df[col] = pd.to_numeric(train_df[col], errors='coerce')
        test_df[col] = pd.to_numeric(test_df[col], errors='coerce')  # 虽然测试集不用于互信息，但保持一致性

    # 只使用训练集计算互信息
    X_combined = train_df[feature_cols].copy()
    y_combined = train_df[label_col].copy()

    # 缺失值填充（用训练集的中位数）
    for col in X_combined.columns:
        med = X_combined[col].median()
        X_combined[col] = X_combined[col].fillna(med)

    # 构建聚合特征矩阵（每个原始变量取所有时间步的均值）
    var_to_cols = {}
    for var in ALL_FEATURES:
        cols = [c for c in feature_cols if c.startswith(var + '_') or c == var]
        var_to_cols[var] = cols

    X_agg = pd.DataFrame(index=X_combined.index)
    for var in ALL_FEATURES:
        cols = var_to_cols.get(var, [])
        if len(cols) == 0:
            X_agg[var] = 0
        else:
            X_agg[var] = X_combined[cols].mean(axis=1)

    # 离散化函数（与原代码相同）
    def discretize(X, bins=MI_BINS):
        X_disc = X.copy()
        for col in X.columns:
            if col in CATEGORICAL_FEATURES:
                continue
            uniq = X[col].nunique()
            if uniq <= 1:
                X_disc[col] = 0
            else:
                try:
                    X_disc[col] = pd.qcut(X[col], q=min(bins, uniq), duplicates='drop', labels=False)
                except ValueError:
                    X_disc[col] = pd.cut(X[col], bins=min(bins, uniq), labels=False)
        return X_disc

    X_disc = discretize(X_agg)

    # 计算互信息（只使用训练集标签）
    mi = mutual_info_classif(X_disc, y_combined, discrete_features='auto', random_state=LASSO_RANDOM_STATE)
    mi_df = pd.DataFrame({'feature': ALL_FEATURES, 'MI': mi}).sort_values('MI', ascending=False)
    total_mi = mi_df['MI'].sum()
    cum_mi = mi_df['MI'].cumsum()
    k_sat = np.searchsorted(cum_mi, MI_INFO_THRESHOLD * total_mi) + 1
    top_k_features = mi_df.head(k_sat)['feature'].tolist()
    print(f"信息饱和点: 前{k_sat}个特征贡献{MI_INFO_THRESHOLD*100:.0f}%互信息: {top_k_features}")
    df = info_analysis(df, top_k_features)
    
    # 构建设计矩阵
    print("\n构建设计矩阵...")
    feat_to_idx = {f: i for i, f in enumerate(ALL_FEATURES)}
    def parse(feature_str):
        mask = np.zeros(len(ALL_FEATURES), dtype=int)
        for f in feature_str.split(','):
            if f in feat_to_idx:
                mask[feat_to_idx[f]] = 1
        return 2 * mask - 1
    masks = np.array([parse(f) for f in df['features']])
    n = len(ALL_FEATURES)
    interactions = []
    for i in range(n):
        for j in range(i+1, n):
            interactions.append(masks[:, i] * masks[:, j])
    design = np.hstack([masks, np.array(interactions).T])
    print(f"设计矩阵形状: {design.shape}")
    
    metrics = ['f2_0', 'precision0', 'recall0', 'auc', 'ap']
    all_top10_rows = []  # 存储所有指标的前10名详细数据
    
    for metric in metrics:
        print(f"\n{'='*50}")
        print(f"分析指标: {metric}")
        # 统计过滤
        weak = df[df['num_feat'] <= WEAK_MODEL_MAX_FEAT][metric].values
        mu0, sigma0 = weak.mean(), weak.std()
        z = (df[metric] - mu0) / sigma0
        p = 2 * (1 - norm.cdf(np.abs(z)))
        sig = benjamini_hochberg(p, FDR_Q)
        df[f'sig_{metric}'] = sig
        sig_combos = df[sig]
        print(f"  显著组合数: {len(sig_combos)}")
        
        # LASSO预测
        pred = factor_analysis(df, metric, design)
        df[f'pred_{metric}'] = pred
        
        # 特征数量惩罚分数
        df[f'penalized_{metric}'] = df[metric] - PENALTY_LAMBDA * df['num_feat']
        
        # 候选池：显著且信息覆盖率达标
        candidate_pool = df[df[f'sig_{metric}'] & (df['info_coverage'] >= INFO_COVERAGE_THRESHOLD)]
        if len(candidate_pool) == 0:
            candidate_pool = df[df[f'sig_{metric}']]  # 降级为所有显著组合
        if len(candidate_pool) == 0:
            candidate_pool = df  # 最终降级为全部组合
        
        # 按惩罚分数排序取前TOP_K
        top_candidates = candidate_pool.nlargest(TOP_K, f'penalized_{metric}')
        print(f"  前{TOP_K}候选组合 (按惩罚分数):")
        for idx, row in top_candidates.iterrows():
            print(f"    {row['features']} ({metric}={row[metric]:.6f}, 特征数={row['num_feat']}, 覆盖率={row['info_coverage']:.2%})")

        for rank, (idx, row) in enumerate(top_candidates.iterrows(), start=1):
            all_top10_rows.append({
                'evaluation criterion': metric,
                'rank': rank,
                'features': row['features'],
                'num_features': row['num_feat'],
                'info_coverage': row['info_coverage'],
                'significant': row[f'sig_{metric}'],
                'penalized_score': row[f'penalized_{metric}'],
                'lasso_pred': row[f'pred_{metric}'],
                'precision0': row['precision0'],
                'recall0': row['recall0'],
                'f2_0': row['f2_0'],
                'precision1': row['precision1'],
                'recall1': row['recall1'],
                'f2_1': row['f2_1'],
                'auc': row['auc'],
                'ap': row['ap']
            })
    
    # 保存每个指标的前10名
    top10_df = pd.DataFrame(all_top10_rows)
    # 按指标和排名排序
    top10_df = top10_df.sort_values(['evaluation criterion', 'rank'])
    top10_output = OUTPUT_DIR / "top10_candidates_per_metric.csv"
    top10_df.to_csv(top10_output, index=False)
    print(f"\n每个指标的前{TOP_K}个候选组合已保存至: {top10_output}")
    return top10_df

def main():
    if OUTPUT_CSV.exists():
        print(f"结果文件已存在: {OUTPUT_CSV}")
        print("直接加载已有结果，跳过训练...")
        df = pd.read_csv(OUTPUT_CSV)
        if 'auc' not in df.columns or 'ap' not in df.columns:
            print("警告: 现有结果文件缺少 'auc' 和 'ap' 列。建议删除文件重新运行以获取这些指标。")
    else:
        print("未找到结果文件，开始训练...")
        df = train_all_combos()
    
    select_best_features(df)
    print("\n" + "="*60)
    print("全流程结束！")
    print("="*60)

if __name__ == "__main__":
    main()