import os
import sys
import json
import logging
from typing import List, Optional, Any, Dict
import argparse

import pandas as pd
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('LLMFundCluster')


class ClusterDefinition(BaseModel):
    cluster_id: int = Field(..., description="簇编号，0 到 n_clusters-1")
    cluster_name: str = Field(..., description="簇名称，尽量简短")
    description: str = Field(..., description="簇的投资主题/风格描述")
    keywords: List[str] = Field(..., description="该簇常见关键词")


class FundAssignment(BaseModel):
    fund_code: str = Field(..., description="基金代码")
    fund_name: str = Field(..., description="基金简称")
    cluster_id: int = Field(..., description="簇编号，0 到 n_clusters-1")
    reason: str = Field(..., description="分配理由（简短）")


class FundClusteringSchema(BaseModel):
    cluster_definitions: List[ClusterDefinition] = Field(..., description="簇定义列表，长度必须等于 n_clusters")
    assignments: List[FundAssignment] = Field(..., description="基金分配列表，必须覆盖输入基金")


LLM_CLUSTER_PROMPT = """
你是一个专业的量化基金研究员。
我有一份包含许多基金基本信息的列表。每行代表一只基金，各字段由竖线 `|` 分隔，格式为：
`基金代码 | 基金名称 | 投资类型 | 投资风格`

例如：
`000001.OF | 华夏成长 | 混合型 | 偏股混合型`

请根据这些基金的名称语义和官方风格，将它们进行聚类（分类）。

要求：
1. 请先总结并定义 {n_clusters} 个聚类主题（Cluster），主题名称应尽可能涵盖市场上常见的主题赛道（如：医药医疗、大消费、新能源、科技制造、大盘宽基、量化指增、高股息红利等）。
2. 然后，为输入的每一只基金分配一个最合适的 Cluster ID（0 到 {n_clusters_minus_1}）。
3. 必须输出严格的 JSON 格式，不要有任何多余的 Markdown 标记。

{format_instructions}

输入基金列表：
{fund_list_text}
"""


class LLMFundClusterer:
    def __init__(
        self,
        model: str,
        base_url: Optional[str],
        api_key: Optional[str],
        temperature: float,
        timeout: int,
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL")

        if not self.api_key:
            logger.error("未找到 OPENAI_API_KEY，请设置环境变量或通过 --api-key 传入")
            sys.exit(1)

        self.llm = ChatOpenAI(
            model=model,
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=timeout,
            temperature=temperature,
        )

    def cluster(self, funds: List[Dict[str, Any]], n_clusters: int) -> Optional[Dict[str, Any]]:
        parser = JsonOutputParser(pydantic_object=FundClusteringSchema)
        prompt = ChatPromptTemplate.from_template(
            template=LLM_CLUSTER_PROMPT,
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )
        chain = prompt | self.llm | parser

        # 构建极简文本输入，去掉冗余的 JSON Key，节省大量 Token
        # 格式: "000001.OF | 华夏成长 | 混合型 | 偏股混合型"
        fund_lines = []
        for row in funds:
            line = f"{row.get('fund_code', '')} | {row.get('fund_name', '')} | {row.get('invest_type', '')} | {row.get('invest_style', '')}"
            fund_lines.append(line)
            
        fund_list_text = "\n".join(fund_lines)

        logger.info(f"发送 LLM 请求进行聚类，输入行数: {len(fund_lines)}，设定聚类数: {n_clusters}")
        try:
            res = chain.invoke({
                "n_clusters": n_clusters,
                "n_clusters_minus_1": n_clusters - 1,
                "fund_list_text": fund_list_text
            })
            return res
        except Exception as e:
            logger.error(f"LLM 聚类请求失败: {e}")
            return None


def load_funds_from_processed_holdings(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols = ["S_INFO_WINDCODE", "F_INFO_NAME", "F_INFO_FIRSTINVESTTYPE", "F_INFO_FIRSTINVESTSTYLE"]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"processed_fund_holdings 缺少列: {missing}")
    df_fund = df[cols].drop_duplicates().copy()
    df_fund = df_fund.rename(
        columns={
            "S_INFO_WINDCODE": "fund_code",
            "F_INFO_NAME": "fund_name",
            "F_INFO_FIRSTINVESTTYPE": "invest_type",
            "F_INFO_FIRSTINVESTSTYLE": "invest_style",
        }
    )
    return df_fund


def save_cluster_outputs(
    df_fund: pd.DataFrame,
    clustering: Dict[str, Any],
    output_csv: str,
    output_definitions_json: Optional[str],
) -> None:
    assignments = pd.DataFrame(clustering["assignments"])
    merged = df_fund.merge(assignments[["fund_code", "cluster_id"]], on="fund_code", how="left")
    if merged["cluster_id"].isna().any():
        missing = merged[merged["cluster_id"].isna()]["fund_code"].head(20).tolist()
        raise ValueError(f"LLM 未覆盖部分基金，示例: {missing}")

    out = merged.rename(columns={"fund_code": "S_INFO_WINDCODE", "cluster_id": "Cluster_Opt"})
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    out.to_csv(output_csv, encoding="utf-8-sig", index=False)

    if output_definitions_json:
        os.makedirs(os.path.dirname(os.path.abspath(output_definitions_json)), exist_ok=True)
        with open(output_definitions_json, "w", encoding="utf-8") as f:
            json.dump(clustering["cluster_definitions"], f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description="LLM 基金名称语义聚类（一次性请求）")
    parser.add_argument("--input", type=str, default="fund/processed_fund_holdings.csv", help="持仓加工后的基金数据 CSV")
    parser.add_argument("--n-clusters", type=int, default=40, help="聚类簇数量")
    parser.add_argument("--output", type=str, default="fund/fund_cluster_llm.csv", help="聚类输出 CSV")
    parser.add_argument("--definitions", type=str, default="fund/fund_cluster_llm_definitions.json", help="簇定义输出 JSON")
    parser.add_argument("--model", type=str, default=os.getenv("OPENAI_MODEL", "gpt-4o"), help="模型名称")
    parser.add_argument("--base-url", type=str, default=os.getenv("OPENAI_BASE_URL"), help="LLM API Base URL")
    parser.add_argument("--api-key", type=str, default=os.getenv("OPENAI_API_KEY"), help="LLM API KEY")
    parser.add_argument("--temperature", type=float, default=0.2, help="采样温度")
    parser.add_argument("--timeout", type=int, default=240, help="超时时间（秒）")

    args = parser.parse_args()

    df_fund = load_funds_from_processed_holdings(args.input)
    funds = df_fund.to_dict(orient="records")
    logger.info(f"输入基金数: {len(funds)}")

    clusterer = LLMFundClusterer(
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        temperature=args.temperature,
        timeout=args.timeout,
    )
    clustering = clusterer.cluster(funds=funds, n_clusters=args.n_clusters)
    save_cluster_outputs(
        df_fund=df_fund,
        clustering=clustering,
        output_csv=args.output,
        output_definitions_json=args.definitions,
    )
    logger.info(f"已生成: {args.output}")


if __name__ == "__main__":
    main()

