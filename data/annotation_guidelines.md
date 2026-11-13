# Annotation Guidelines

## Goal

For each audited model output, review the item, the gold answer or rubric, the
model response, and the automatic component labels. Record whether the following
binary labels should be kept or overridden:

- `SemanticCorrect`
- `LangCorrect`
- `ConstraintCorrect`

`JointSuccess` is derived from these labels:

```text
JointSuccess = SemanticCorrect * LangCorrect * ConstraintCorrect
```

`CodeSwitchRatio` records the automatic contamination-ratio measurement and is
not manually re-scored in the human audit.

## 1. SemanticCorrect

Judge whether the model completed the semantic target of the item correctly,
regardless of output language or formatting.

Use the following scale:

- `0` = incorrect
- `1` = correct

### Semantic reversal

Consider:

- Does the output recover the intended stance rather than the surface wording?
- Does it identify the correct evaluative target?
- Does it avoid reversing the intended polarity?

### Final-state extraction

Consider:

- Did the output give the final valid value?
- Did it ignore outdated or superseded values?
- Did it avoid introducing an incorrect value?

### Language purity with update realization

Consider:

- Did the output preserve the main update?
- Did it avoid omitting the critical change?
- Did it avoid unsupported additions or contradictions?

## 2. LangCorrect

Judge whether the model answered in the required response language.

Use the following scale:

- `0` = target-language requirement is violated
- `1` = target-language requirement is satisfied

### Notes

- Pure numerals, single-letter labels, allowlisted proper names, and allowlisted
  platform or system names may remain unchanged.
- Non-target-language lexical items or structural framing should be marked as
  violations unless they fall under the allowlist.
- For example, `327` may be acceptable in a Chinese response, while `Room 327`
  is English structural framing.

## 3. ConstraintCorrect

Judge whether the model satisfied the explicit output constraint in the
instruction.

Use the following scale:

- `0` = explicit constraint is violated
- `1` = explicit constraint is satisfied

Recurring constraint types include yes-or-no with brief reason, output-only,
single sentence, banned word, and starts-with prefix.

## 4. JointSuccess

Set:

- `1` if `SemanticCorrect = 1`, `LangCorrect = 1`, and `ConstraintCorrect = 1`
- `0` otherwise

This is intentionally strict.

## Borderline cases

### Case A: Content is correct, but language is wrong

- `SemanticCorrect = 1`
- `LangCorrect = 0`
- `JointSuccess = 0`

### Case B: Language is correct, but content is wrong

- `SemanticCorrect = 0`
- `LangCorrect = 1`
- `JointSuccess = 0`

### Case C: Content and language are correct, but an explicit constraint is violated

- `SemanticCorrect = 1`
- `LangCorrect = 1`
- `ConstraintCorrect = 0`
- `JointSuccess = 0`

## Recommended annotation process

1. Read the item and gold answer or rubric.
2. Judge semantic correctness first.
3. Judge target-language adherence second.
4. Judge explicit constraint satisfaction third.
5. Derive `JointSuccess` last.