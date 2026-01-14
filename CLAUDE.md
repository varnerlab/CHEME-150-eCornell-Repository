# Notebook Review Workflow for Claude

## Standard Review Process

When reviewing new lecture or example notebooks, follow this workflow:

### Step 1: Read the New Notebook
- Read the unreleased lecture or example notebook provided by the user
- Get familiar with its content, structure, and current state

### Step 2: Review Recent Released Examples
- Read previously released example or lecture notebooks that match the topic/style
- Use these as reference for tone, style, and formatting

**CRITICAL: Section Formatting Pattern**
- All interpretive sections follow this exact pattern:
  1. **Blockquote explanation** (wrap in `> `)
  2. **Code cell** (if applicable)
  3. **Simple transition sentence** (NOT a long paragraph)
- Do NOT create lengthy interpretive paragraphs after code
- Do NOT format headers as `__Text:__` followed by paragraph text
- Match the formatting of existing sections exactly
- When unsure, make sections shorter, not longer
- Check nearby sections for formatting patterns before writing new content

### Step 3: Generate Learning Objectives and Summary Section
Generate three components:

#### A. Learning Objectives (3 items)
- Create three specific, actionable learning objectives
- Use format: `* __[Objective title]:__ [description]`
- Each should be concrete and measurable
- **Important**: Check the notebook for specific instructions or placeholders in the learning objectives section - the user may specify exactly what should go in each objective
- Only use direct, simple, and concise language
- Avoid adjectives
- Content must be directly supported by and derived from the notebook content - do not generate false or unsupported text

#### B. Summary Section with Key Takeaways
- Add a **Summary** section after the main content
- Include **Key Takeaways** subsection with exactly three bullet points
- Use format:
  ```
  * **[Takeaway title]:** [explanation]
  ```
- Each takeaway should be concise (1-2 sentences)
- **Important**: Check the notebook for specific instructions or placeholders in the summary section - the user may specify exactly what should go in each takeaway
- Only use direct, simple, and concise language
- Avoid adjectives
- Content must be directly supported by and derived from the notebook content - do not generate false or unsupported text

#### C. Structure
- Add a **Direct Summary Sentence**: One sentence that captures the core essence of the notebook
- Follow with **Conclusion Sentence**: One sentence about implications or next steps
- These should appear before or after the Key Takeaways section

**Tone & Style Requirements:**
- Match the tone, style, and formatting from the uploaded reference examples
- Use the same technical language and depth as reference materials
- Maintain consistent markdown formatting
- Ensure mathematical notation and code examples follow established patterns
- Use direct, simple, and concise language
- Avoid unnecessary adjectives
- **Do not use equations in Learning Objectives or Key Takeaways** - use conceptual descriptions instead

**Bullet Point Guidelines:**
- When using bullet points, use conversational style matching the Learning Objectives format
- Format: `* __[Title]:__ [concise description]`
- Each bullet should be focused and scannable
- Prefer conversational bullet points over long prose paragraphs when listing key points
- Keep each bullet to 1-2 sentences maximum
- Example (from Learning Objectives): `* __Understand RNN architecture and memory mechanisms:__ Explain how RNNs maintain hidden states...`

### Step 4: Mathematical Notation Review
Check all mathematical notation for consistency and precision:

**Notation Consistency:**
- Verify all norms are explicitly specified (e.g., $\|\cdot\|_{2}$ for 2-norm, $\|\cdot\|_{2}^{2}$ for squared 2-norm, $\|\cdot\|_{1}$ for 1-norm)
- Do not use unspecified norms like $\|\cdot\|$ without a subscript
- Ensure vector/matrix notation is consistent throughout (e.g., always use $\mathbf{x}$ for vectors, $\mathbf{W}$ for matrices)
- Use non-bold notation for scalar components (e.g., $x_i$ for the $i$-th component of vector $\mathbf{x}$)
- Check that transpose notation is consistent (choose one style and use throughout)
- Verify superscript/subscript conventions are applied uniformly

**Blockquote Organization:**
- Never have multiple consecutive blockquotes without connective text between them
- Add at least one sentence of regular text between blockquotes to provide context and flow
- Always add a transition sentence or phrase between a blockquote and a code cell
- Do not use multi-level bullet points within blockquote sections
- Blockquotes should be self-contained explanatory sections, not fragmented pieces
- When presenting multi-step processes, consider using a single blockquote with internal structure rather than multiple separate blockquotes

**LaTeX Rendering Verification:**
- All LaTeX equations must be syntactically correct and render properly
- Ensure all `$$..$$` or `$...$` delimiters are properly matched
- Verify all `\begin{...}...\end{...}` blocks (align, align*, equation, etc.) are properly closed
- Check that blockquoted content with LaTeX is correctly formatted (blockquote prefix `>` should not break equation structure)
- Test multi-line equations in blockquotes: ensure `$$\begin{align*}...\end{align*}$$` is complete within blockquote context
- Avoid orphaned equation lines outside proper `\begin...\end` blocks
- For equations spanning blockquote content, close and reopen `$$` delimiters correctly

**Convergence and Stopping Criteria:**
- All convergence criteria must specify the exact norm or metric used
- Define tolerance parameters explicitly (e.g., $\epsilon > 0$)
- Ensure all variables in convergence checks are defined before use
- Specify whether comparing to previous iteration, initial value, or fixed point

**Algorithm Specifications:**
- All algorithm steps must be unambiguous and executable
- Define all variables before first use
- Specify data types where relevant (e.g., $\theta\in\mathbb{R}^{p}$)
- Include initialization requirements
- Clarify loop bounds and termination conditions

**Domain and Range:**
- Specify function domains and codomains (e.g., $f:\mathbb{R}^{n}\rightarrow\mathbb{R}^{m}$)
- Ensure activation function signatures match their descriptions
- Verify set membership notation is correct

### Step 4a: CRITICAL MATHEMATICAL AND LOGICAL EVALUATION
**This is a mandatory deep-dive verification step.** Do not skip or minimize this step. Mathematical or logical errors undermine the entire educational value of the notebook.

#### Mathematical Correctness Check
**Verify all mathematical derivations and statements:**

1. **Derivation Verification:**
   - Check each step in multi-step derivations for algebraic correctness
   - Verify intermediate results are logically sound
   - Confirm all algebraic manipulations are valid (no invalid cancellations, distributions, or operations)
   - Check that limits, integrals, and infinite series are handled correctly
   - Verify dimensional consistency (both sides of equations have same dimensions)

2. **Definition Consistency:**
   - Confirm all definitions are stated clearly and consistently used throughout
   - Verify definitions match standard mathematical conventions or clearly state when deviating
   - Check that mathematical objects (matrices, vectors, scalars) are consistently defined
   - Ensure notation introduction precedes use

3. **Theorem and Formula Correctness:**
   - Verify all cited theorems are correctly stated
   - Check that formula applications match theorem requirements/assumptions
   - Confirm boundary conditions and special cases are handled correctly
   - Validate that constraints (e.g., $\lambda > 0$, $m \leq n$) are properly enforced

4. **Boundary and Special Cases:**
   - Test edge cases (zero values, infinite values, negative values) against formulas
   - Verify behavior at boundaries matches mathematical logic
   - Check limiting behavior (as $x\to\infty$, $x\to 0$, etc.) is correct
   - Validate degenerate cases are handled properly

#### Logical Correctness Check
**Verify logical flow and reasoning:**

1. **Logical Chain of Reasoning:**
   - Confirm each claim follows logically from previous claims
   - Verify no logical gaps or unjustified leaps in reasoning
   - Check that necessary conditions are stated before using them
   - Ensure all quantifiers (for all, there exists) are explicit

2. **Proof Correctness:**
   - Verify proofs are complete and not missing steps
   - Check that conclusions actually follow from stated premises
   - Confirm no circular reasoning (assuming conclusion to prove conclusion)
   - Validate that proof techniques are correctly applied

3. **Equivalence and Implication Verification:**
   - When stating "$A = B$", verify they are truly equal (not approximately equal)
   - When using "$\Rightarrow$", verify the implication is valid in all cases
   - Check "$\Leftrightarrow$" statements are truly bidirectional
   - Confirm transformations preserve the intended mathematical meaning

4. **Assumption and Dependency Tracking:**
   - List all implicit assumptions made
   - Verify assumptions are reasonable for the context
   - Check that later results don't violate earlier assumptions
   - Confirm all dependencies between concepts are stated

#### Content Accuracy Check
**Verify factual and conceptual correctness:**

1. **Conceptual Accuracy:**
   - Confirm interpretations of mathematical concepts are correct
   - Verify descriptions of algorithms match their formal definitions
   - Check that intuitive explanations accurately reflect the mathematics
   - Validate that examples correctly illustrate concepts

2. **Cross-Notebook Consistency:**
   - Verify notation and definitions match other notebooks in the same course
   - Check that concepts build consistently across lectures
   - Confirm no contradictions with previously taught material
   - Validate that references to other notebooks are accurate

3. **Algorithm and Procedure Correctness:**
   - Verify algorithm steps are in correct order
   - Check that all initialization steps are necessary and sufficient
   - Confirm termination conditions are correct
   - Validate convergence claims are mathematically justified

4. **Claim Verification:**
   - For each major claim, verify it is mathematically sound
   - Check that performance or complexity claims are justified
   - Confirm "always/never/guaranteed" claims are actually universal
   - Validate numerical examples match stated formulas

#### Documentation of Issues
If any mathematical or logical errors are found:
- **Flag each error clearly** with its location (section and equation numbers)
- **Explain the error** in detail (what is wrong and why)
- **Provide the correction** with justification
- **Suggest how to fix it** in the notebook
- **Do not proceed** to final rating until all errors are resolved or acknowledged as intentional

If notebook passes all checks, state: **"✓ Mathematical and logical verification complete: No errors found"**

### Step 5: Final Quality Check
- Fix all spelling errors
- Fix all grammar issues
- Fix punctuation errors
- Fix awkward or unclear text
- Ensure consistency with reference examples
- **Verify technical accuracy** (already done in Step 4a for mathematics)
- **Language quality**: Verify content uses direct, simple, and concise language without unnecessary adjectives
- **Content accuracy**: Confirm all generated content is directly supported by the notebook - no unsupported claims
- **Mathematical rigor**: Confirm all mathematical content has passed Step 4a verification

### Step 6: Rate the Notebook
Provide a rating on a scale of **0-10** based on:

**Quality Criteria:**
- **Flow**: Does the notebook have a clear logical progression?
- **Technical Correctness**: Are all technical concepts accurate?
- **Mathematical Correctness**: Are all equations and mathematical statements correct? (Verified in Step 4a)
- **Mathematical Precision**: Are norms, convergence criteria, and technical specifications explicit and unambiguous?
- **Clarity**: Is the content understandable to the target audience?
- **Completeness**: Are all necessary components included?

**Important:** Do not assign a rating above 7/10 if any mathematical or logical errors were found in Step 4a. Mathematical errors are critical deficiencies.

**Rating Scale:**
- 0-2: Really bad (major issues throughout, including mathematical errors)
- 3-4: Poor (significant problems including mathematical/logical errors)
- 5-6: Adequate (acceptable but needs improvement, mathematical errors fixed)
- 7-8: Good (mostly correct with minor issues, no mathematical errors)
- 9-10: Really great (excellent, no mathematical/logical errors, minimal or no other issues)

**Provide:**
- Overall rating (e.g., `8/10`)
- Brief explanation of rating
- 2-3 specific strengths
- 1-2 specific areas for improvement (if any)
- **Confirm mathematical and logical verification status** from Step 4a

---

## Critical Guidelines

### Language Standards
- Use direct, simple, and concise language at all times
- Avoid adjectives unless essential for technical accuracy
- Remove any flowery, descriptive, or vague language
- Prefer active voice and clear subject-verb-object structure

### Content Integrity
- All generated content (learning objectives, summaries, takeaways) must be directly supported by the notebook
- Do not generate unsupported text, claims, or ideas
- Do not extrapolate beyond what is explicitly covered in the notebook
- If unsure whether content is supported, do not include it
- Flag any areas where you cannot find supporting evidence in the notebook

### Handling Placeholders and Instructions
- Check all sections of the notebook carefully for existing instructions or placeholders
- Users may leave specific guidance about what should go in learning objectives or summary sections
- Follow these instructions precisely - they override general formatting guidelines
- Ask for clarification if instructions are ambiguous
