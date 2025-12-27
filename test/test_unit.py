import pytest
from unittest.mock import AsyncMock, patch

from app import extract_label_and_categories, ask

@pytest.mark.unit
def test_extract_label_logic():
    assert extract_label_and_categories("Safety: Safe")[0] == "Safe"
    assert extract_label_and_categories("Safety: Controversial")[0] == "Controversial"
    assert extract_label_and_categories("Safety: Unsafe")[0] == "Unsafe"
    assert extract_label_and_categories("No label")[0] == "Unknown"

@pytest.mark.unit
@pytest.mark.asyncio
async def test_ask_guardrail_blocks():
    with patch("app.run_guardrail_check", new=AsyncMock(return_value=("Unsafe", "Blocked"))):
        with patch("app.get_generator_response", new=AsyncMock(return_value={"gen_content": "NO"})):
            result = await ask("bad prompt")
            assert "guard_content" in result
            assert "gen_content" not in result

@pytest.mark.unit
@pytest.mark.asyncio
async def test_ask_guardrail_allows():
    with patch("app.run_guardrail_check", new=AsyncMock(return_value=("Safe", "OK"))):
        with patch("app.get_generator_response", new=AsyncMock(return_value={"gen_content": "YES"})):
            result = await ask("good prompt")
            assert result == {"gen_content": "YES"}
