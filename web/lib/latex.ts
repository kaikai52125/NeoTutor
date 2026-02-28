/**
 * Utility functions for LaTeX processing
 *
 * remark-math only supports $...$ and $$...$$ delimiters by default.
 * Many LLMs output LaTeX using \(...\) and \[...\] delimiters.
 * This utility converts between formats.
 */

/**
 * Convert LaTeX delimiters from \(...\) and \[...\] to $...$ and $$...$$
 * This makes the content compatible with remark-math for ReactMarkdown rendering.
 */
export function convertLatexDelimiters(content: string): string {
  if (!content) return content;

  let result = content;

  // Convert \[...\] to $$...$$ (block math)
  result = result.replace(/\\\[([\s\S]*?)\\\]/g, "\n$$\n$1\n$$\n");

  // Convert \(...\) to $...$ (inline math)
  result = result.replace(/\\\(([\s\S]*?)\\\)/g, " $$$1$$ ");

  // Clean up multiple consecutive newlines
  result = result.replace(/\n{3,}/g, "\n\n");

  return result;
}

/**
 * Remove common leading whitespace from all non-empty lines (dedent).
 * Prevents LLM-indented content from being treated as Markdown code blocks
 * (Markdown interprets 4+ leading spaces as a code block).
 */
function dedentContent(content: string): string {
  const lines = content.split("\n");
  const minIndent = lines.reduce((min, line) => {
    if (line.trim() === "") return min;
    const indent = line.match(/^(\s*)/)?.[1].length ?? 0;
    return Math.min(min, indent);
  }, Infinity);
  if (minIndent === 0 || minIndent === Infinity) return content;
  return lines.map((line) => line.slice(minIndent)).join("\n");
}

/**
 * If the entire content is wrapped in a single fenced code block (```markdown ... ```)
 * by the LLM, unwrap it so ReactMarkdown renders it as Markdown, not as code.
 */
function unwrapFencedCodeBlock(content: string): string {
  const trimmed = content.trim();
  const match = trimmed.match(/^```(?:markdown|md)?\n([\s\S]*?)\n?```$/);
  return match ? match[1] : content;
}

/**
 * Process content for ReactMarkdown rendering with proper LaTeX support.
 * Applies fenced-block unwrapping, dedent, and LaTeX delimiter conversion.
 */
export function processLatexContent(content: string): string {
  if (!content) return "";

  const str = String(content);

  // Unwrap if LLM wrapped the whole response in a ```markdown block
  const unwrapped = unwrapFencedCodeBlock(str);

  // Remove common leading indentation so Markdown doesn't treat lines as code blocks
  const dedented = dedentContent(unwrapped);

  // Apply LaTeX delimiter conversion
  return convertLatexDelimiters(dedented);
}
