import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
import matplotlib.pyplot as plt

def extract_rules_from_top_features(df, top_features_file='selected_features_80pct.csv', target_col='label', max_depth=3):
    """
    使用单棵浅层决策树，从选出的核心特征中提取人类可读的 If-Else 选股规则。
    
    参数:
    - df: 包含特征和 label 的 DataFrame (例如从 build_df_for_xgb 出来的 df)
    - top_features_file: SHAP 挑选出的核心特征 CSV 文件路径
    - max_depth: 决策树最大深度，建议 3-4，太深规则会过于复杂
    """
    print(f"1. 加载核心特征列表: {top_features_file}")
    try:
        top_features_df = pd.read_csv(top_features_file)
        # 提取特征名列表
        core_features = top_features_df['feature'].tolist()
        print(f"   成功加载 {len(core_features)} 个核心特征。")
    except FileNotFoundError:
        print(f"   [警告] 找不到文件 {top_features_file}，为了演示，将使用 df 中前 5 个特征列。")
        # 排除基础列
        exclude_cols = ['fund', 'stock', 'date', 'label']
        core_features = [c for c in df.columns if c not in exclude_cols][:5]

    # 2. 准备数据
    print(f"\n2. 准备决策树训练数据 (仅使用核心特征)...")
    # 确保所选特征在 df 中存在
    valid_features = [f for f in core_features if f in df.columns]
    
    X = df[valid_features]
    y = df[target_col]
    
    # 填充可能存在的缺失值 (决策树不支持 NaN，用中位数填充较为稳妥)
    X = X.fillna(X.median())
    
    # 3. 训练浅层决策树
    print(f"\n3. 训练浅层决策树 (最大深度={max_depth})...")
    # 设置 class_weight='balanced' 是因为正负样本比例通常是 1:10
    tree_clf = DecisionTreeClassifier(max_depth=max_depth, class_weight='balanced', random_state=42)
    tree_clf.fit(X, y)
    
    # 4. 打印纯文本的 If-Else 规则
    print("\n=======================================================")
    print("🌟 提取的选股规则 (If-Else 形式) 🌟")
    print("说明: class: 1 代表基金可能买入(正样本)，class: 0 代表不买(负样本)")
    print("=======================================================\n")
    
    tree_rules = export_text(tree_clf, feature_names=valid_features)
    print(tree_rules)
    
    # 5. 寻找高胜率的“买入规则” (叶子节点分析)
    print("\n--- 高胜率买入路径分析 ---")
    analyze_leaf_nodes(tree_clf, valid_features, X, y)

    # 6. 可视化并保存决策树结构图
    print("\n4. 正在生成决策树可视化图 (decision_tree_rules.png)...")
    plt.figure(figsize=(20, 10))
    plot_tree(
        tree_clf, 
        feature_names=valid_features, 
        class_names=['Not Buy (0)', 'Buy (1)'], 
        filled=True, 
        rounded=True,
        proportion=True,
        fontsize=10
    )
    plt.savefig('decision_tree_rules.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("   保存成功！可以通过查看该图片直观地了解各个阈值划分。")

def get_lineage(tree, feature_names):
    """
    遍历决策树，提取从根节点到每个叶子节点的完整规则路径。
    返回一个字典：{leaf_node_id: [(feature, operator, threshold), ...]}
    """
    left      = tree.tree_.children_left
    right     = tree.tree_.children_right
    threshold = tree.tree_.threshold
    features  = [feature_names[i] if i != -2 else "undefined!" for i in tree.tree_.feature]

    def recurse(left, right, child, lineage=None):
        if lineage is None:
            lineage = [child]
        if child in left:
            parent = np.where(left == child)[0].item()
            split = 'L'
        else:
            parent = np.where(right == child)[0].item()
            split = 'R'

        # 记录：如果是左子树，说明满足 <= 阈值；如果是右子树，说明满足 > 阈值
        op = "<=" if split == 'L' else ">"
        val = threshold[parent]
        feat = features[parent]
        
        rule = (feat, op, val)
        
        lineage.append(rule)

        if parent == 0:
            lineage.reverse()
            return lineage
        else:
            return recurse(left, right, parent, lineage)

    # 获取所有的叶子节点ID
    leaves = np.where(left == -1)[0]
    
    paths = {}
    for leaf in leaves:
        if leaf == 0:
            # 特殊情况：如果根节点就是叶子节点（树只有深度为0）
            paths[leaf] = []
            continue
        path = recurse(left, right, leaf)
        # 第一个是根节点0，最后一个是叶子节点自身编号，中间是我们构建的 rule tuple
        # 提取 rule 列表
        rules_only = [item for item in path if isinstance(item, tuple)]
        paths[leaf] = rules_only
    return paths

def analyze_leaf_nodes(tree_clf, feature_names, X, y):
    """
    分析决策树的叶子节点，找出预测为正类(买入)且纯度较高(胜率高)的规则路径，
    并直接输出可用于 Pandas/回测 的 Python 代码。
    """
    tree_ = tree_clf.tree_
    paths = get_lineage(tree_clf, feature_names)
    
    high_prob_rules = []
    
    # 遍历所有叶子节点
    for leaf_id, path in paths.items():
        # 获取该叶子节点的样本分布 [负样本权重和, 正样本权重和]
        value = tree_.value[leaf_id][0]
        # 计算预测为正类 (买入) 的概率
        buy_prob = value[1] / (value[0] + value[1])
        # 样本数量
        samples = tree_.n_node_samples[leaf_id]
        
        # 定义高潜规则：买入概率 > 60% 且 覆盖样本数 > 总样本的 1%
        if buy_prob > 0.6 and samples > len(X) * 0.01:
            # 拼接 Pandas 的查询条件
            if not path:
                # 树只有根节点，没有规则
                continue
            
            conditions = []
            for feat, op, val in path:
                # 例如： (df['feature_A'] > 3.5)
                conditions.append(f"(df['{feat}'] {op} {val:.4f})")
            
            pandas_rule = " & ".join(conditions)
            
            high_prob_rules.append({
                'node_id': leaf_id,
                'buy_prob': buy_prob,
                'samples': samples,
                'rule_code': pandas_rule
            })
            
            print(f"⭐ 发现高潜选股节点 (节点ID: {leaf_id}):")
            print(f"   - 覆盖样本数: {samples} (占总样本 {samples/len(X)*100:.1f}%)")
            print(f"   - 预测买入胜率 (加权): {buy_prob*100:.1f}%")
            print(f"   - 实际命中正样本数: {int(value[1])} (估算)")
            print(f"   - 【回测提取代码】:\n     selected_stocks = df[{pandas_rule}]\n")

    if high_prob_rules:
        print("\n=======================================================")
        print("🚀 【汇总】可直接复制到回测框架的最终组合选股代码:")
        print("=======================================================")
        print("import pandas as pd\n")
        print("def select_stocks(df):")
        print("    \"\"\"根据决策树提取的高胜率规则选股\"\"\"")
        print("    # 只要满足以下任意一条高潜规则，即买入 (逻辑或 OR)")
        print("    mask = (")
        
        for i, rule in enumerate(high_prob_rules):
            prefix = "        " if i == 0 else "      | "
            print(f"{prefix}({rule['rule_code']})  # 规则 {i+1} (胜率 {rule['buy_prob']*100:.1f}%)")
            
        print("    )")
        print("    return df[mask]")
        print("=======================================================\n")
    else:
        print("\n⚠️ 未找到满足条件 (胜率>60% 且 覆盖>1%) 的买入规则。")
        print("   建议调整树的深度 (max_depth) 或降低胜率阈值。")

def extract_rules_with_rulefit(df, top_features_file='selected_features_80pct.csv', target_col='label', max_rules=20):
    """
    使用 RuleFit 算法从核心特征中提取带有权重 (系数) 的选股规则。
    RuleFit 结合了树模型的非线性特征组合和 Lasso 回归的特征选择能力，
    能自动剔除冗余规则，留下最具代表性的高胜率路径。
    """
    try:
        from imodels import RuleFitClassifier
    except ImportError:
        print("\n[错误] 缺少 imodels 库。请先运行: pip install imodels")
        return

    print(f"\n=======================================================")
    print(f"🌟 启动 RuleFit 算法规则提取 (最大规则数: {max_rules}) 🌟")
    print(f"=======================================================\n")

    # 1. 加载核心特征
    try:
        top_features_df = pd.read_csv(top_features_file)
        core_features = top_features_df['feature'].tolist()
        print(f"1. 成功加载 {len(core_features)} 个核心特征。")
    except FileNotFoundError:
        print(f"1. [警告] 找不到文件 {top_features_file}，将使用 df 中的前列作为演示。")
        exclude_cols = ['fund', 'stock', 'date', 'label']
        core_features = [c for c in df.columns if c not in exclude_cols][:5]

    # 2. 准备数据
    valid_features = [f for f in core_features if f in df.columns]
    X = df[valid_features].fillna(df[valid_features].median())
    y = df[target_col]

    print(f"2. 准备训练数据完成，特征维度: {X.shape}")

    # 3. 训练 RuleFit
    print(f"3. 正在训练 RuleFit 模型，请稍候...")
    rf = RuleFitClassifier(max_rules=max_rules, random_state=42)
    
    # 捕获并忽略内部的 Lasso 警告
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rf.fit(X, y, feature_names=valid_features)

    # 4. 提取并过滤规则
    df_rules = rf._get_rules()
    
    # 我们只关心预测正类（买入）的规则，即 coef > 0
    buy_rules = df_rules[(df_rules['type'] == 'rule') & (df_rules['coef'] > 0)].copy()
    
    if buy_rules.empty:
        print("\n⚠️ RuleFit 未能找到具有正向权重的买入规则，建议增加 max_rules 或检查数据。")
        return

    # 按重要性（或系数大小）排序
    buy_rules = buy_rules.sort_values(by='coef', ascending=False)

    print("\n--- 🎯 RuleFit 提取的高价值买入规则 ---")
    
    final_pandas_rules = []
    
    for i, row in buy_rules.iterrows():
        rule_str = row['rule']
        coef = row['coef']
        importance = row['importance']
        support = row['support']  # 规则覆盖的样本比例
        
        # 将 RuleFit 的规则字符串 (例如: feature_A <= 4.2 and feature_B > 8.8)
        # 转换为 Pandas 查询格式: (df['feature_A'] <= 4.2) & (df['feature_B'] > 8.8)
        conditions = rule_str.split(' and ')
        pd_conds = []
        for cond in conditions:
            # 解析特征、操作符和值
            parts = cond.split(' ')
            if len(parts) == 3:
                feat, op, val = parts[0], parts[1], parts[2]
                pd_conds.append(f"(df['{feat}'] {op} {val})")
        
        pandas_rule = " & ".join(pd_conds)
        final_pandas_rules.append({'code': pandas_rule, 'coef': coef})
        
        print(f"⭐ 规则 {len(final_pandas_rules)}:")
        print(f"   - 逻辑: {rule_str}")
        print(f"   - 权重 (Coef): +{coef:.4f} (正向打分越大，买入概率越高)")
        print(f"   - 覆盖率: {support*100:.1f}%")
        print(f"   - 【回测代码】: df[{pandas_rule}]\n")

    print("=======================================================")
    print("🚀 【汇总】可直接复制到回测框架的多因子打分代码:")
    print("=======================================================")
    print("import pandas as pd\n")
    print("def score_stocks_with_rulefit(df):")
    print("    \"\"\"根据 RuleFit 提取的规则对股票进行加权打分\"\"\"")
    print("    # 初始得分为 0")
    print("    df['rulefit_score'] = 0.0\n")
    
    for i, rule in enumerate(final_pandas_rules):
        print(f"    # 规则 {i+1} (权重: +{rule['coef']:.4f})")
        print(f"    mask_{i+1} = {rule['code']}")
        print(f"    df.loc[mask_{i+1}, 'rulefit_score'] += {rule['coef']:.4f}\n")
        
    print("    # 选出得分最高的 Top K 只股票")
    print("    return df.sort_values(by='rulefit_score', ascending=False)")
    print("=======================================================\n")

if __name__ == "__main__":
    # 这里构造一个非常简单的假 df 用于演示脚本运行
    print("--- 启动决策树规则提取演示 ---")
    np.random.seed(42)
    # 假设我们有 1000 个样本，正负比例 1:10
    n_samples = 1100
    mock_df = pd.DataFrame({
        'label': [1]*100 + [0]*1000,
        'feature_A': np.concatenate([np.random.normal(5, 1, 100), np.random.normal(2, 1, 1000)]), # 正样本 A 较高
        'feature_B': np.concatenate([np.random.normal(10, 2, 100), np.random.normal(15, 2, 1000)]), # 正样本 B 较低
        'feature_C': np.random.normal(0, 1, 1100) # 噪音特征
    })
    
    # 我们随便建一个假的 selected_features_80pct.csv
    mock_features = pd.DataFrame({'feature': ['feature_A', 'feature_B']})
    mock_features.to_csv('dummy_features.csv', index=False)
    
    # 运行单棵树提取
    extract_rules_from_top_features(mock_df, top_features_file='dummy_features.csv', target_col='label', max_depth=2)
    
    # 运行 RuleFit 提取
    extract_rules_with_rulefit(mock_df, top_features_file='dummy_features.csv', target_col='label', max_rules=10)
