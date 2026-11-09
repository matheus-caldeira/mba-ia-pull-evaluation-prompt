"""
Testes automatizados para validação de prompts.
"""
import pytest
import yaml
import sys
from pathlib import Path

# Adicionar src ao path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils import validate_prompt_structure

PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "bug_to_user_story_v2.yml"


def load_prompts(file_path: str):
    """Carrega prompts do arquivo YAML."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def prompt():
    """Dados do prompt otimizado (v2), já sem wrapper se houver."""
    data = load_prompts(str(PROMPT_PATH))
    assert data is not None, "YAML vazio ou inválido"

    # O v2 é plano na raiz; aceita wrapper por compatibilidade.
    if "bug_to_user_story_v2" in data:
        data = data["bug_to_user_story_v2"]

    return data


class TestPrompts:
    def test_prompt_has_system_prompt(self, prompt):
        """Verifica se o campo 'system_prompt' existe e não está vazio."""
        assert "system_prompt" in prompt, "Campo 'system_prompt' ausente no YAML"

        system_prompt = prompt["system_prompt"]
        assert isinstance(system_prompt, str), "'system_prompt' deve ser string"
        assert system_prompt.strip(), "'system_prompt' está vazio"

        # A estrutura completa também precisa passar na validação oficial.
        is_valid, errors = validate_prompt_structure(prompt)
        assert is_valid, f"Estrutura inválida: {errors}"

    def test_prompt_has_role_definition(self, prompt):
        """Verifica se o prompt define uma persona (ex: "Você é um Product Manager")."""
        system_prompt = prompt["system_prompt"]

        assert "Você é" in system_prompt, (
            "O prompt não define uma persona explícita ('Você é ...')"
        )

        personas = ["Product Manager", "Product Owner", "PO"]
        assert any(persona in system_prompt for persona in personas), (
            f"Nenhuma persona de produto encontrada. Esperado um de: {personas}"
        )

    def test_prompt_mentions_format(self, prompt):
        """Verifica se o prompt exige formato Markdown ou User Story padrão."""
        system_prompt = prompt["system_prompt"]

        # Formato canônico da user story.
        assert "Como um" in system_prompt, "Formato 'Como um <persona>' não exigido"
        assert "eu quero" in system_prompt, "Formato 'eu quero <objetivo>' não exigido"
        assert "para que" in system_prompt, "Formato 'para que <benefício>' não exigido"

        # Seção de critérios em Gherkin.
        assert "Critérios de Aceitação" in system_prompt, (
            "Seção 'Critérios de Aceitação' não exigida"
        )

        gherkin = ["Dado que", "Quando", "Então"]
        for keyword in gherkin:
            assert keyword in system_prompt, (
                f"Palavra-chave Gherkin ausente no prompt: '{keyword}'"
            )

    def test_prompt_has_few_shot_examples(self, prompt):
        """Verifica se o prompt contém exemplos de entrada/saída (técnica Few-shot)."""
        system_prompt = prompt["system_prompt"]

        # Pares entrada -> saída explícitos.
        input_markers = system_prompt.count("Relato de Bug:")
        output_markers = system_prompt.count("User Story:")

        assert input_markers >= 2, (
            f"Few-shot exige pelo menos 2 exemplos de entrada, encontrados: {input_markers}"
        )
        assert output_markers >= 2, (
            f"Few-shot exige pelo menos 2 exemplos de saída, encontrados: {output_markers}"
        )

        # A técnica deve estar declarada nos metadados.
        techniques = " ".join(prompt.get("techniques_applied", [])).lower()
        assert "few-shot" in techniques or "few shot" in techniques, (
            "Técnica Few-shot não declarada em 'techniques_applied'"
        )

    def test_prompt_no_todos(self, prompt):
        """Garante que você não esqueceu nenhum `[TODO]` no texto."""
        campos = ["description", "system_prompt", "user_prompt"]

        for campo in campos:
            valor = prompt.get(campo) or ""
            assert "[TODO]" not in valor, f"'{campo}' ainda contém [TODO]"
            assert "TODO" not in valor, f"'{campo}' ainda contém TODO"

        # As técnicas declaradas também não podem ter placeholders.
        for technique in prompt.get("techniques_applied", []):
            assert "TODO" not in technique, (
                f"Técnica com placeholder pendente: {technique}"
            )

    def test_minimum_techniques(self, prompt):
        """Verifica (através dos metadados do yaml) se pelo menos 2 técnicas foram listadas."""
        assert "techniques_applied" in prompt, (
            "Campo 'techniques_applied' ausente nos metadados"
        )

        techniques = prompt["techniques_applied"]
        assert isinstance(techniques, list), "'techniques_applied' deve ser uma lista"
        assert len(techniques) >= 2, (
            f"Mínimo de 2 técnicas requeridas, encontradas: {len(techniques)}"
        )

        # Nenhuma entrada vazia.
        for technique in techniques:
            assert str(technique).strip(), "Técnica vazia em 'techniques_applied'"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
