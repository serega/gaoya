pub fn tokenize_text<'a>(text: &'a str) -> impl Iterator<Item = &'a str> {
    return text
        .split(|c: char| c.is_ascii_punctuation() || c.is_ascii_whitespace())
        .filter(|&x| !x.is_empty());
}
