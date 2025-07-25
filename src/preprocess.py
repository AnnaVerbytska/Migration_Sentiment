import re

def clean_text(text):
    # Keep the original text intact
    text = text.strip()

    # Remove markdown links (keep the text part if available)
    text = re.sub(r"\[([^\]]+)\]\(https?://[^\)]+\)", r"\1", text)

    # Remove bare URLs
    text = re.sub(r"http\S+|www\.\S+", "", text)

    # Remove markdown bold/italic formatting
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)  # **bold**
    text = re.sub(r"\*(.*?)\*", r"\1", text)      # *italic*

    # Replace newlines, carriage returns, tabs with space
    text = text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')

    # Remove repeated whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


