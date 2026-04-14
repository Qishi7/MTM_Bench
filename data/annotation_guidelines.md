# Annotation Guidelines

## Goal
For each model output, label four fields:
- `TaskCorrect`
- `LangCorrect`
- `CodeSwitch`
- `JointSuccess`

## 1. TaskCorrect
Judge whether the model completed the requested task correctly **regardless of output language**.

Use the following scale:
- `0` = incorrect
- `1` = partially correct
- `2` = correct

### Reading comprehension
Consider:
- Did the output answer the actual question?
- Is the answer consistent with the content?
- Does it capture the key intended meaning?
- Does it avoid clear hallucination?

### Extraction
Consider:
- Were the correct fields extracted?
- Were important fields omitted?
- Were incorrect values introduced?

### Constrained summarization / rewriting
Consider:
- Did the output preserve the main meaning?
- Did it satisfy task constraints such as length or format?
- Did it avoid unsupported additions?

## 2. LangCorrect
Judge whether the model answered in the required response language.

Use the following scale:
- `0` = wrong language
- `1` = target language is dominant but mixed with noticeable non-target language
- `2` = clearly and consistently in the target language

### Notes
- A brief quoted phrase from the source may still be acceptable if the answer is otherwise in the target language.
- If most of the answer is in the wrong language, mark `0`.
- If the answer alternates between languages or includes large untranslated spans, usually mark `1` or `0` depending on severity.

## 3. CodeSwitch
Judge the degree of non-target-language mixing.

Use the following scale:
- `0` = no obvious code-switching
- `1` = mild code-switching
- `2` = severe code-switching

### Examples
- A mostly English answer with one short Chinese quoted term may be `1`.
- An answer split between English and Chinese paragraphs is usually `2`.

## 4. JointSuccess
Set:
- `1` if `TaskCorrect = 2` and `LangCorrect = 2`
- `0` otherwise

This is intentionally strict.

## Borderline cases
### Case A: Content is correct, but language is wrong
- `TaskCorrect = 2`
- `LangCorrect = 0`
- `JointSuccess = 0`

### Case B: Language is correct, but content is wrong
- `TaskCorrect = 0`
- `LangCorrect = 2`
- `JointSuccess = 0`

### Case C: Content is mostly correct, but answer contains noticeable mixed language
- `TaskCorrect = 1` or `2` depending on content quality
- `LangCorrect = 1`
- `JointSuccess = 0`

## Recommended annotation process
1. Read the item and gold/rubric.
2. Judge content correctness first.
3. Judge target-language adherence second.
4. Record code-switch severity.
5. Derive `JointSuccess` last.
