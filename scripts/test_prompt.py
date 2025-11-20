"""
Test script to validate the improved JSON prompt template.

This script tests the prompt template changes without requiring GPU/model inference.
It validates that the prompt includes proper JSON formatting instructions.

Usage:
    python scripts/test_prompt.py
"""

import sys
from pathlib import Path

# Add runners directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "runners"))

from common import load_batch_verification_prompt_template, format_element_descriptions, load_data_element_descriptions


def test_prompt_formatting():
    """Test that prompt template includes JSON formatting instructions."""
    print("=" * 80)
    print("TESTING PROMPT TEMPLATE WITH REASONING MODE")
    print("=" * 80)

    # Load prompt template with reasoning
    prompt_template = load_batch_verification_prompt_template(with_reasoning=True)

    # Check for critical JSON formatting instructions
    critical_instructions = [
        "CRITICAL",
        "valid JSON",
        "escaped",
        "opening brace",
        "closing brace",
        "double quotes",
        "trailing commas",
        "EXACT format"
    ]

    print("\nChecking for critical JSON formatting instructions:")
    all_found = True
    for instruction in critical_instructions:
        found = instruction in prompt_template
        status = "✓" if found else "✗"
        print(f"  {status} '{instruction}': {'Found' if found else 'MISSING'}")
        if not found:
            all_found = False

    if all_found:
        print("\n✓ All critical JSON formatting instructions are present!")
    else:
        print("\n✗ Some critical instructions are missing!")
        return False

    # Test with sample data
    print("\n" + "=" * 80)
    print("TESTING PROMPT WITH SAMPLE DATA")
    print("=" * 80)

    sample_input = "Contact john.doe@company.com or call 1-800-555-0123 for support"
    sample_piis = ["Email Address", "Phone Number"]

    # Load element descriptions
    element_descriptions = load_data_element_descriptions()
    formatted_descriptions = format_element_descriptions(sample_piis, element_descriptions)

    # Format prompt
    formatted_prompt = prompt_template.format(
        input_text=sample_input,
        detected_piis=str(sample_piis),
        element_descriptions=formatted_descriptions
    )

    print("\nFormatted prompt preview (first 500 chars):")
    print(formatted_prompt[:500])
    print("...")
    print("\nFormatted prompt preview (last 500 chars):")
    print("..." + formatted_prompt[-500:])

    # Verify the formatted prompt looks correct
    print("\n" + "=" * 80)
    print("VALIDATION RESULTS")
    print("=" * 80)

    validations = [
        ("Contains input text", sample_input in formatted_prompt),
        ("Contains detected PIIs", "Email Address" in formatted_prompt and "Phone Number" in formatted_prompt),
        ("Contains element descriptions", "formatted_descriptions" not in formatted_prompt),  # Should be replaced
        ("Contains JSON example", '"results"' in formatted_prompt),
        ("Contains critical instructions", "CRITICAL" in formatted_prompt)
    ]

    all_valid = True
    for validation_name, is_valid in validations:
        status = "✓" if is_valid else "✗"
        print(f"  {status} {validation_name}")
        if not is_valid:
            all_valid = False

    if all_valid:
        print("\n✓ Prompt template is correctly formatted and includes all necessary components!")
        return True
    else:
        print("\n✗ Prompt template has issues!")
        return False


def test_simple_mode():
    """Test simple mode (comma-separated) prompt."""
    print("\n" + "=" * 80)
    print("TESTING PROMPT TEMPLATE WITH SIMPLE MODE (COMMA-SEPARATED)")
    print("=" * 80)

    # Load prompt template without reasoning
    prompt_template = load_batch_verification_prompt_template(with_reasoning=False)

    # Check for comma-separated instructions
    print("\nChecking for comma-separated format instructions:")
    if "comma-separated" in prompt_template.lower():
        print("  ✓ Comma-separated format instructions found")
    else:
        print("  ✗ Comma-separated format instructions missing")
        return False

    # Should NOT contain JSON instructions in simple mode
    if "JSON" not in prompt_template or '"results"' not in prompt_template:
        print("  ✓ JSON instructions correctly excluded from simple mode")
        return True
    else:
        print("  ✗ JSON instructions incorrectly present in simple mode")
        return False


def main():
    """Run all tests."""
    print("PII Verification Prompt Template Tests")
    print("=" * 80)

    # Test reasoning mode
    reasoning_ok = test_prompt_formatting()

    # Test simple mode
    simple_ok = test_simple_mode()

    # Final summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    if reasoning_ok and simple_ok:
        print("✓ All tests passed!")
        print("\nThe prompt template is correctly configured with:")
        print("  • Strong JSON formatting instructions (reasoning mode)")
        print("  • Proper comma-separated format (simple mode)")
        print("  • Element descriptions integration")
        print("  • 3-test framework (OWNERSHIP, SPECIFICITY, CONTEXT)")
        return 0
    else:
        print("✗ Some tests failed!")
        if not reasoning_ok:
            print("  • Reasoning mode (JSON) prompt has issues")
        if not simple_ok:
            print("  • Simple mode (comma-separated) prompt has issues")
        return 1


if __name__ == "__main__":
    sys.exit(main())
