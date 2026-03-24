# AIAgentDeployRender

Projeto minimo de agente com:
- Frontend em `Streamlit`
- Agente com `LangChain`
- Uso de `tools` (calculadora, data/hora e inverter texto)
- Deploy no Render com `Dockerfile`

## Estrutura

- `app.py`: interface web com chat
- `src/agent.py`: construcao do agente e tools
- `requirements.txt`: dependencias Python
- `Dockerfile`: imagem para deploy no Render

## Rodando localmente

1. Crie e ative um ambiente virtual
2. Instale dependencias:

```bash
pip install -r requirements.txt
```

3. Crie um arquivo `.env` na raiz:

```bash
OPENAI_API_KEY="sua_chave"
OPENAI_MODEL="gpt-4o-mini"
```

4. Rode:

```bash
streamlit run app.py
```

## Deploy no Render (Docker)

1. Suba este repositorio para GitHub
2. No Render, crie um novo **Web Service**
3. Escolha o repo e use deploy por `Dockerfile`
4. Defina a variavel de ambiente:
   - `OPENAI_API_KEY`
   - (opcional) `OPENAI_MODEL` (padrao: `gpt-4o-mini`)
5. Deploy

O comando de inicializacao ja esta no `Dockerfile`, usando `PORT` fornecida pelo Render.