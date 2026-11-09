"""
Script para fazer pull de prompts do LangSmith Prompt Hub.

Este script:
1. Conecta ao LangSmith usando credenciais do .env
2. Faz pull do prompt semente do desafio
3. Salva localmente em prompts/bug_to_user_story_v1.yml

O pull usa o cliente do LangSmith (`Client.pull_prompt`). O parâmetro
`dangerously_pull_public_prompt=True` é necessário porque o identificador tem
dono explícito ("owner/nome"): um prompt do Hub é um objeto LangChain
serializado e o LangSmith bloqueia esse pull por padrão. Aqui o prompt é o
semente do desafio, então o risco é conhecido.
"""

import sys
from pathlib import Path
from dotenv import load_dotenv
from langsmith import Client
from utils import save_yaml, check_env_vars, print_section_header

load_dotenv()

# Prompt de origem definido pelo enunciado do desafio.
SOURCE_PROMPT = "leonanluppi/bug_to_user_story_v1"

# Nome da chave no YAML e caminho de destino (relativo à raiz do projeto).
PROMPT_KEY = "bug_to_user_story_v1"
OUTPUT_PATH = "prompts/bug_to_user_story_v1.yml"


def _extract_messages(prompt_template) -> dict:
    """
    Extrai system_prompt e user_prompt de um ChatPromptTemplate.

    `prompt.messages` é uma lista de SystemMessagePromptTemplate /
    HumanMessagePromptTemplate; cada item expõe o texto em `.prompt.template`.
    Se o objeto for um PromptTemplate simples, o texto vira o system_prompt.
    """
    extracted = {}

    messages = getattr(prompt_template, "messages", None)
    if not messages:
        template = getattr(prompt_template, "template", None)
        if template:
            extracted["system_prompt"] = template
        return extracted

    system_parts = []
    user_parts = []

    for message in messages:
        template = getattr(getattr(message, "prompt", None), "template", None)
        if template is None:
            continue

        class_name = type(message).__name__
        if "System" in class_name:
            system_parts.append(template)
        elif "Human" in class_name or "User" in class_name:
            user_parts.append(template)

    if system_parts:
        extracted["system_prompt"] = "\n\n".join(system_parts)
    if user_parts:
        extracted["user_prompt"] = "\n\n".join(user_parts)

    return extracted


def pull_prompts_from_langsmith() -> bool:
    """
    Faz pull do prompt semente do LangSmith Hub e salva localmente em YAML.

    Returns:
        True se sucesso, False caso contrário
    """
    print(f"📥 Fazendo pull de: {SOURCE_PROMPT}")

    try:
        client = Client()
        prompt_template = client.pull_prompt(
            SOURCE_PROMPT,
            dangerously_pull_public_prompt=True,
        )
    except Exception as e:
        print(f"❌ Falha ao fazer pull de '{SOURCE_PROMPT}': {e}\n")
        print("Verifique:")
        print("  - LANGSMITH_API_KEY está configurada corretamente no .env")
        print("  - O prompt existe e está público")
        print("  - Sua conexão com a internet está funcionando")
        return False

    extracted = _extract_messages(prompt_template)

    if not extracted.get("system_prompt"):
        print("❌ Não foi possível extrair o system_prompt do objeto retornado.")
        print(f"   Tipo recebido: {type(prompt_template).__name__}")
        return False

    input_variables = list(getattr(prompt_template, "input_variables", []) or [])

    prompt_data = {
        PROMPT_KEY: {
            "description": "Prompt para converter relatos de bugs em User Stories",
            **extracted,
            "version": "v1",
            "source": SOURCE_PROMPT,
            "input_variables": input_variables,
            "tags": ["bug-analysis", "user-story", "product-management"],
        }
    }

    print(f"   Variáveis de entrada: {input_variables or 'nenhuma'}")
    print(f"💾 Salvando em: {OUTPUT_PATH}")

    if not save_yaml(prompt_data, OUTPUT_PATH):
        return False

    print("✅ Prompt salvo com sucesso.")
    return True


def main():
    """Função principal"""
    print_section_header("PULL DE PROMPTS DO LANGSMITH HUB")

    if not check_env_vars(["LANGSMITH_API_KEY"]):
        return 1

    if not pull_prompts_from_langsmith():
        return 1

    print_section_header("CONCLUÍDO")
    print(f"Prompt disponível em: {Path(OUTPUT_PATH).resolve()}")
    print("\nPróximo passo: otimizar o prompt em prompts/bug_to_user_story_v2.yml")
    return 0


if __name__ == "__main__":
    sys.exit(main())
