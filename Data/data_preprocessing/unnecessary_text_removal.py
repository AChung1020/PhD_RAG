import re


def remove_page_numbers(input_file, output_file):
    """
    Remove page number indicators in the format 'Page X of Y' from a markdown file.

    Args:
        input_file (str): Path to the input markdown file
        output_file (str): Path to save the cleaned markdown file
    """
    # Read the input file
    with open(input_file, "r", encoding="utf-8") as f:
        content = f.read()

    # Remove page numbers using regex
    # This pattern matches:
    # - Optional whitespace at the start of the line
    # - "Page" (case-insensitive)
    # - Followed by spaces and numbers
    # - "of" (case-insensitive)
    # - Followed by spaces and numbers
    # - Optional whitespace at the end of the line
    cleaned_content = re.sub(
        r"^\s*Page\s+\d+\s+of\s+\d+\s*$",
        "",
        content,
        flags=re.MULTILINE | re.IGNORECASE,
    )

    # Remove any double blank lines that might be created
    cleaned_content = re.sub(r"\n\s*\n\s*\n", "\n\n", cleaned_content)

    # Write the cleaned content to the output file
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(cleaned_content)


# Example usage
if __name__ == "__main__":
    input_file = "/Users/andrewchung/PycharmProjects/PhD_RAG/Data/MD_handbooks/laney-graduate-studies-handbook.md"
    output_file = "/Data/MD_handbooks/laney-graduate-studies-handbook-cleaned.md"
    remove_page_numbers(input_file, output_file)
