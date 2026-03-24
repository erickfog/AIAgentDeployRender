import ast
import operator
import os
import random
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
    """Calcula expressoes matematicas (stake, ROI percentual, etc.). Exemplo: (100 * 1.85) - 100"""
    try:
        tree = ast.parse(expressao, mode="eval")
        result = _safe_eval(tree.body)
        return f"Resultado: {result}"
    except Exception as exc:
        return f"Erro ao calcular: {exc}"


@tool
def probabilidade_implicita(odd_decimal: str) -> str:
    """Probabilidade implicita (sem margem da casa) a partir da odd decimal. Ex: 2.0 -> 50%."""
    try:
        odd = float(str(odd_decimal).strip().replace(",", "."))
        if odd <= 1:
            return "Odd decimal deve ser maior que 1."
        pct = (1 / odd) * 100
        return f"Probabilidade implicita: {pct:.2f}% (1 / {odd})"
    except ValueError:
        return "Informe um numero valido, ex: 1.90 ou 2.5"


@tool
def retorno_aposta(entrada: str) -> str:
    """Retorno bruto e lucro de uma aposta simples. Formato: stake,odd_decimal  Ex: 50,2.10"""
    try:
        parts = [p.strip().replace(",", ".") for p in entrada.split(",")]
        if len(parts) != 2:
            return "Use: stake,odd_decimal  Exemplo: 100,1.75"
        stake, odd = float(parts[0]), float(parts[1])
        if stake <= 0 or odd <= 1:
            return "Stake deve ser > 0 e odd decimal > 1."
        bruto = stake * odd
        lucro = bruto - stake
        return (
            f"Retorno bruto: {bruto:.2f} | Lucro: {lucro:.2f} "
            f"(stake {stake:.2f} x odd {odd})"
        )
    except ValueError:
        return "Valores invalidos. Exemplo: 100,1.90"


@tool
def odd_parlay(odds: str) -> str:
    """Odd combinada (multipla/parlay) multiplicando odds decimais. Ex: 1.50,2.00,1.80"""
    items = [p.strip().replace(",", ".") for p in odds.split(",") if p.strip()]
    if len(items) < 2:
        return "Passe pelo menos duas odds separadas por virgula."
    try:
        mult = 1.0
        for s in items:
            o = float(s)
            if o <= 1:
                return f"Cada odd deve ser > 1. Problema em: {o}"
            mult *= o
        return f"Odd combinada: {mult:.4f} (produto de {len(items)} selecoes)"
    except ValueError:
        return "Use numeros validos, ex: 1.55,1.60,2.10"


@tool
def data_hora_atual(_: str = "") -> str:
    """Retorna data e hora atuais no formato ISO."""
    return datetime.now().isoformat(timespec="seconds")


@tool
def sortear_mercado(opcoes: str) -> str:
    """Apenas entretenimento: sorteia UMA opcao entre alternativas que o usuario listou (virgula). Nao e palpite profissional."""
    items = [p.strip() for p in opcoes.split(",") if p.strip()]
    if len(items) < 2:
        return "Passe pelo menos duas opcoes separadas por virgula."
    return f"Sorteio (nao e recomendacao): {random.choice(items)}"


_DICAS_RESPONSAVEIS = (
    "Aposte apenas o que pode perder; trate como entretenimento, nao como renda.",
    "Defina limite de tempo e valor antes de comecar — e pare quando bater o limite.",
    "Odd alta nao significa 'certeza'; probabilidade implicita mostra o que o mercado precifica.",
    "Evite recuperar perdas com apostas maiores (tilt); faz parte do jogo responsavel.",
    "Se o jogo deixar de ser divertido, busque apoio (ex.: CVV 188 no Brasil).",
)


@tool
def lembrete_jogo_responsavel(_: str = "") -> str:
    """Retorna um lembrete curto sobre jogo responsavel (aleatorio)."""
    return random.choice(_DICAS_RESPONSAVEIS)


def build_agent_executor() -> AgentExecutor:
    model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    llm = ChatOpenAI(model=model_name, temperature=0.45)

    tools = [
        calculadora,
        probabilidade_implicita,
        retorno_aposta,
        odd_parlay,
        data_hora_atual,
        sortear_mercado,
        lembrete_jogo_responsavel,
    ]

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Voce e um assistente especializado em APOSTAS ESPORTIVAS e mercados de probabilidade, "
                "em portugues do Brasil. Ajude a entender odds (principalmente decimais), retorno, "
                "probabilidade implicita, apostas multiplas e calculos com as ferramentas — use-as "
                "para numeros exatos. "
                "NUNCA garanta resultado de evento, insider, ou 'certeza' de lucro. Deixe claro que "
                "previsoes sao incertas e que a casa tem margem. Quando falar de escolha de mercado, "
                "seja educativo (conceitos, risco, gestao de banca) e, se o usuario pedir 'palpite', "
                "explique que voce nao tem dados ao vivo nem estatisticas proprietarias: pode discutir "
                "linhas de raciocinio genericas, nao conselho financeiro. "
                "Mencione jogo responsavel quando fizer sentido. Respostas diretas; listas curtas quando "
                "couber.",
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=False)
