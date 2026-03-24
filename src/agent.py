import ast
import operator
import os
from datetime import datetime

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI


ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}


def _safe_eval(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in ALLOWED_OPERATORS:
        return ALLOWED_OPERATORS[type(node.op)](
            _safe_eval(node.left), _safe_eval(node.right)
        )
    if isinstance(node, ast.UnaryOp) and type(node.op) in ALLOWED_OPERATORS:
        return ALLOWED_OPERATORS[type(node.op)](_safe_eval(node.operand))
    raise ValueError("Expressao invalida. Use apenas numeros e + - * / **")


@tool
def calculadora(expressao: str) -> str:
    """Calcula expressoes matematicas simples. Exemplo: (2 + 3) * 4"""
    try:
        tree = ast.parse(expressao, mode="eval")
        result = _safe_eval(tree.body)
        return f"Resultado: {result}"
    except Exception as exc:
        return f"Erro ao calcular: {exc}"


@tool
def data_hora_atual(_: str = "") -> str:
    """Retorna data e hora atuais no formato ISO."""
    return datetime.now().isoformat(timespec="seconds")


@tool
def inverter_texto(texto: str) -> str:
    """Inverte a ordem dos caracteres de um texto."""
    return texto[::-1]


def build_agent_executor() -> AgentExecutor:
    model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    llm = ChatOpenAI(model=model_name, temperature=0)

    tools = [calculadora, data_hora_atual, inverter_texto]

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Voce e um agente simples e objetivo. "
                "Use tools quando fizer sentido e responda em portugues.",
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=False)
