import os
from datetime import datetime

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI


@tool
def data_hora_atual(_: str = "") -> str:
    """Retorna data e hora atuais no formato ISO."""
    return datetime.now().isoformat(timespec="seconds")


@tool
def dias_entre_datas(inicio: str, fim: str, formato: str = "YYYY-MM-DD") -> str:
    """Calcula quantos dias (inteiros) existem entre duas datas (fim - inicio).

    Exemplo: inicio=2026-03-26, fim=2026-03-30 -> 4
    """
    try:
        # (Mantemos formato simples por enquanto: YYYY-MM-DD.)
        if formato != "YYYY-MM-DD":
            return "Formato nao suportado. Use inicio/fim em YYYY-MM-DD."
        dt_inicio = datetime.strptime(inicio.strip(), "%Y-%m-%d")
        dt_fim = datetime.strptime(fim.strip(), "%Y-%m-%d")
        delta = (dt_fim - dt_inicio).days
        if delta < 0:
            return "A data fim precisa ser igual ou posterior a data inicio."
        return f"{delta} dia(s)"
    except ValueError:
        return "Datas invalidas. Use inicio e fim no formato YYYY-MM-DD."


@tool
def orcamento_simples(
    preco_noite: str,
    noites: str,
    passagem: str = "0",
    alimentacao: str = "0",
    passeios: str = "0",
    outros: str = "0",
) -> str:
    """Soma um orcamento simples de viagem (valores em R$).

    Exemplo de uso: preco_noite=350, noites=3, passagem=900, alimentacao=450, passeios=300
    """

    def _to_float(v: str) -> float:
        return float(str(v).strip().replace(",", "."))

    try:
        p_noite = _to_float(preco_noite)
        n = int(float(_to_float(noites)))
        if p_noite < 0 or n < 1:
            return "Verifique preco_noite (>= 0) e noites (>= 1)."

        itens = [
            p_noite * n,
            _to_float(passagem),
            _to_float(alimentacao),
            _to_float(passeios),
            _to_float(outros),
        ]
        total = sum(itens)
        hospedagem = itens[0]
        return (
            f"Total estimado: R$ {total:.2f} (hospedagem: R$ {hospedagem:.2f})"
        )
    except ValueError:
        return "Valores invalidos. Use numeros (ex: 1.90 ou 190,50)."


@tool
def trilha_itinerario(destino: str, dias: str, interesses: str = "") -> str:
    """Gera uma trilha de itinerario (dia a dia) de forma generica (sem dados ao vivo).

    - destino: ex 'Gramado'
    - dias: ex '4'
    - interesses: ex 'natureza, gastronomia, cultura'
    """
    try:
        d = int(float(dias.strip().replace(",", ".")))
        if d < 1:
            return "Informe dias >= 1."
    except ValueError:
        return "Informe dias como numero inteiro (ex: 4)."

    base = [x.strip().lower() for x in interesses.split(",") if x.strip()]
    if not base:
        base = ["cultura", "gastronomia", "natureza"]

    # Mapeamos interesses para atividades genericas.
    sugestoes = []
    for item in base:
        if "praia" in item:
            sugestoes.append("manha na orla + almoco local")
        elif "natureza" in item or "trilha" in item or "parque" in item:
            sugestoes.append("passeio de natureza / mirante")
        elif "cultura" in item or "museu" in item:
            sugestoes.append("visita cultural (museu/centro historico)")
        elif "gastronomia" in item or "comida" in item:
            sugestoes.append("roteiro gastronomico (cafes + jantar)")
        elif "compras" in item:
            sugestoes.append("tempo para compras e feiras/lojas locais")
        else:
            sugestoes.append(f"atividade ligada a {item}")

    # Preenche dia a dia com um padrao simples.
    diario = []
    for i in range(1, d + 1):
        foco = sugestoes[(i - 1) % len(sugestoes)]
        diario.append(f"Dia {i}: {foco} + descanso a tarde em {destino.strip()}.")

    return "\n".join(diario)


def build_agent_executor() -> AgentExecutor:
    model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    llm = ChatOpenAI(model=model_name, temperature=0.7)

    tools = [
        data_hora_atual,
        dias_entre_datas,
        orcamento_simples,
        trilha_itinerario,
    ]

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Voce e a consultora Camila, de uma agencia de viagem chamada 'Rumo Leve'. "
                "Ajuda o usuario a planejar roteiros, prazos e estimar custos de forma educativa. "
                "Use as ferramentas para calculos exatos (dias entre datas, orcamento) e para gerar "
                "um itinerario generico sem dados ao vivo. "
                "Quando faltarem informacoes, facca 3-5 perguntas objetivas. "
                "Se houver dados suficientes, responda com: (1) roteiro dia a dia, (2) estimativa de custos "
                "(3) checklist do que levar e (4) perguntas finais para refinamento. "
                "Sempre responda em portugues do Brasil. Evite prometer reserva ou disponibilidade real; "
                "trata-se de uma sugestao de planejamento.",
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=False)
