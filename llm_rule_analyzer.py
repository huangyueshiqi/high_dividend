import os
import sys
import json
import logging
from typing import List, Dict, Optional, Any
import argparse

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('LLMRuleAnalyzer')


class RuleItem(BaseModel):
    expression: str = Field(..., description="一条规则表达式（可包含AND/OR）")
    explanation: str = Field(..., description="用金融语言解释这条规则在表达什么")


class StrategyAnalysisSchema(BaseModel):
    one_liner: str = Field(..., description="一句话策略描述")
    rule_interpretation: List[RuleItem] = Field(..., description="规则逐条解释")
    robust_suggestions: List[str] = Field(..., description="稳健化建议")
    risk_points: List[str] = Field(..., description="风险点")
    assumptions: List[str] = Field(..., description="因输入缺失而做出的假设或需要进一步确认的信息")


LLM_RULE_ANALYSIS_PROMPT = """
你是量化研究员助手。输入是一段“决策树提取出的买入规则 + 字段含义/公式/单位”的文本，以及已知的部分回测元信息。
你的任务是将其转写为严格 JSON，帮助研究员形成可解释、可落地且可审阅的交易策略描述。

要求：
1) 只基于输入文本推理，不要编造字段含义或数据来源。
2) 【关键推理任务】：请根据输入的规则字段频率（如是否包含月度、日度特征）自行推理最合适的“调仓周期”和“选股数量”。
   - 调仓周期限制：**必须至少为月频（即不能长于一个月，例如可以是日频、周频、月频，但不能是季频/年频）**。
   - 选股规则：若无明确说明，可默认假设为“满足条件即买入，等权配置”或“打分选前20只”。
3) one_liner 输出示例风格：回测区间：{start_date}到{end_date}。寻找低市盈率(PE_TTM)且高ROE的股票，每20个交易日调仓，等权持仓，选前20只。
4) rule_interpretation 中请尽量把每个条件解释为“趋势/动量/成交活跃/风险偏好/流动性”等金融语言，并指出该条件属于择时过滤还是偏向选股特征。
5) robust_suggestions 中必须包含至少一条“用分位数/标准化/滚动窗口替代绝对阈值”的建议（若输入中存在绝对阈值）。
6) 若有些参数实在无法推断，请在 assumptions 中列出你做出的默认假设。

输入规则文本：
{user_input}

已知回测信息：
- 开始日期：{start_date}
- 结束日期：{end_date}

{format_instructions}
"""


class LLMRuleAnalyzer:
    def __init__(
        self,
        model: str = "gpt-4o",
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.2,
        timeout: int = 240,
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

    def analyze(self, user_input: str, output_file: str, start_date: str = "未知", end_date: str = "未知") -> Optional[Dict[str, Any]]:
        parser = JsonOutputParser(pydantic_object=StrategyAnalysisSchema)
        prompt = ChatPromptTemplate.from_template(
            template=LLM_RULE_ANALYSIS_PROMPT,
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )
        chain = prompt | self.llm | parser

        try:
            result = chain.invoke({
                "user_input": user_input,
                "start_date": start_date,
                "end_date": end_date
            })
            os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            logger.info(f"策略分析结果已保存至: {output_file}")
            return result
        except Exception as e:
            logger.error(f"生成策略分析失败: {e}")
            return None


def read_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def main():
    parser = argparse.ArgumentParser(description="LLM 决策树规则 -> 策略解释 JSON 生成器")
    parser.add_argument("--rule-text", type=str, default=None, help="规则文本（包含字段含义）")
    parser.add_argument("--rule-file", type=str, default=None, help="规则文本文件路径（utf-8）")
    parser.add_argument("--output", type=str, default="config/strategy_analysis.json", help="输出 JSON 文件路径")
    parser.add_argument("--start-date", type=str, default="未知", help="回测开始日期")
    parser.add_argument("--end-date", type=str, default="未知", help="回测结束日期")
    parser.add_argument("--model", type=str, default=os.getenv("OPENAI_MODEL", "gpt-4o"), help="模型名称")
    parser.add_argument("--base-url", type=str, default=os.getenv("OPENAI_BASE_URL"), help="LLM API Base URL")
    parser.add_argument("--api-key", type=str, default=os.getenv("OPENAI_API_KEY"), help="LLM API KEY（建议用环境变量）")
    parser.add_argument("--temperature", type=float, default=0.2, help="采样温度")

    args = parser.parse_args()

    if not args.rule_text and not args.rule_file:
        logger.error("必须提供 --rule-text 或 --rule-file 之一")
        sys.exit(2)

    user_input = args.rule_text or read_text_file(args.rule_file)

    analyzer = LLMRuleAnalyzer(
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        temperature=args.temperature,
    )
    analyzer.analyze(
        user_input=user_input, 
        output_file=args.output,
        start_date=args.start_date,
        end_date=args.end_date
    )


if __name__ == "__main__":
    main()

