"""
Script para fazer push de prompts otimizados ao LangSmith Prompt Hub.

Este script:
1. Lê os prompts otimizados de prompts/bug_to_user_story_v2.yml
2. Valida os prompts
3. Faz push PÚBLICO para o LangSmith Hub
4. Adiciona metadados (tags, descrição, técnicas utilizadas)

O push usa o cliente do LangSmith (`Client.push_prompt`) com `is_public=True`.
As técnicas aplicadas vão no readme do prompt, visível no Hub.
"""

import os
import sys
from dotenv import load_dotenv
from langsmith import Client
from langchain_core.prompts import ChatPromptTemplate
from utils import load_yaml, check_env_vars, print_section_header, validate_prompt_structure

load_dotenv()

# Arquivo de origem e nome do prompt publicado.
PROMPT_FILE = "prompts/bug_to_user_story_v2.yml"
PROMPT_NAME = "bug_to_user_story_v2"

# Variável de entrada única esperada pelo evaluate.py (chave do dataset).
EXPECTED_INPUT_VARIABLES = {"bug_report"}


def _build_chat_prompt(prompt_data: dict) -> ChatPromptTemplate:
    """Monta o ChatPromptTemplate (system + user) a partir dos dados do YAML."""
    system_prompt = prompt_data["system_prompt"]
    user_prompt = prompt_data.get("user_prompt") or "{bug_report}"

    return ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", user_prompt),
    ])


def _build_readme(prompt_name: str, prompt_data: dict) -> str:
    """Readme publicado no Hub, com versão e técnicas aplicadas."""
    lines = [
        f"# {prompt_name}",
        "",
        prompt_data.get("description", "").strip(),
        "",
        f"Versão: {prompt_data.get('version', 'v2')}",
        "",
        "## Técnicas aplicadas",
        "",
    ]
    lines += [f"- {technique}" for technique in prompt_data.get("techniques_applied", [])]
    return "\n".join(lines)


def push_prompt_to_langsmith(prompt_name: str, prompt_data: dict) -> bool:
    """
    Faz push do prompt otimizado para o LangSmith Hub (PÚBLICO).

    Args:
        prompt_name: Nome do prompt
        prompt_data: Dados do prompt

    Returns:
        True se sucesso, False caso contrário
    """
    username = os.getenv("USERNAME_LANGSMITH_HUB", "").strip()
    if not username:
        print("❌ USERNAME_LANGSMITH_HUB não configurada no .env")
        print("   Veja no .env.example como criar seu handle público do Hub.")
        return False

    repo_full_name = f"{username}/{prompt_name}"

    try:
        chat_prompt = _build_chat_prompt(prompt_data)
    except Exception as e:
        print(f"❌ Erro ao montar o ChatPromptTemplate: {e}")
        return False

    # Guarda crítica: qualquer variável além de bug_report faz o evaluate.py
    # falhar em todos os exemplos (chaves literais não escapadas no texto).
    input_variables = set(chat_prompt.input_variables)
    if input_variables != EXPECTED_INPUT_VARIABLES:
        print("❌ Variáveis de entrada inesperadas no template.")
        print(f"   Esperado: {sorted(EXPECTED_INPUT_VARIABLES)}")
        print(f"   Encontrado: {sorted(input_variables)}")
        print("\n   Causa provável: chaves '{' ou '}' literais não escapadas no")
        print("   YAML. Escape-as como '{{' e '}}'.")
        return False

    description = prompt_data.get("description", "").strip()
    tags = [str(tag) for tag in prompt_data.get("tags", [])]
    techniques = prompt_data.get("techniques_applied", [])

    print(f"📤 Fazendo push PÚBLICO de: {repo_full_name}")
    print(f"   Tags: {tags}")
    print(f"   Técnicas: {len(techniques)}")

    try:
        client = Client()
        url = client.push_prompt(
            repo_full_name,
            object=chat_prompt,
            is_public=True,
            description=description,
            readme=_build_readme(prompt_name, prompt_data),
            tags=tags,
        )
    except Exception as e:
        print(f"❌ Falha ao fazer push de '{repo_full_name}': {e}\n")
        print("Verifique:")
        print("  - LANGSMITH_API_KEY está configurada corretamente no .env")
        print("  - USERNAME_LANGSMITH_HUB corresponde ao seu handle do Hub")
        print("  - Sua conexão com a internet está funcionando")
        return False

    print(f"✅ Push concluído: {url}")
    return True


def validate_prompt(prompt_data: dict) -> tuple[bool, list]:
    """
    Valida estrutura básica de um prompt (versão simplificada).

    Args:
        prompt_data: Dados do prompt

    Returns:
        (is_valid, errors) - Tupla com status e lista de erros
    """
    if not isinstance(prompt_data, dict):
        return (False, ["Conteúdo do YAML não é um dicionário"])

    _, errors = validate_prompt_structure(prompt_data)

    # O user_prompt é quem deve carregar a variável...
    if "{bug_report}" not in (prompt_data.get("user_prompt") or ""):
        errors.append("user_prompt deve conter a variável {bug_report}")

    # ...e o system_prompt não deve duplicá-la (erro clássico da v1).
    if "{bug_report}" in (prompt_data.get("system_prompt") or ""):
        errors.append(
            "system_prompt não deve conter {bug_report} (duplicação com user_prompt)"
        )

    return (len(errors) == 0, errors)


def main():
    """Função principal"""
    print_section_header("PUSH DE PROMPTS PARA O LANGSMITH HUB")

    if not check_env_vars(["LANGSMITH_API_KEY", "USERNAME_LANGSMITH_HUB"]):
        return 1

    print(f"📖 Lendo: {PROMPT_FILE}")
    prompt_data = load_yaml(PROMPT_FILE)
    if prompt_data is None:
        return 1

    # O v2 é plano na raiz, mas aceita-se um wrapper (formato do v1).
    if PROMPT_NAME in prompt_data:
        prompt_data = prompt_data[PROMPT_NAME]

    print("🔎 Validando estrutura do prompt...")
    is_valid, errors = validate_prompt(prompt_data)
    if not is_valid:
        print("❌ Prompt inválido:")
        for error in errors:
            print(f"   - {error}")
        return 1

    print("✅ Estrutura válida.\n")

    if not push_prompt_to_langsmith(PROMPT_NAME, prompt_data):
        return 1

    print_section_header("CONCLUÍDO")
    print("Próximo passo: executar a avaliação com `python src/evaluate.py`")
    return 0


if __name__ == "__main__":
    sys.exit(main())
