# ASTevolve Case Sheet Schema

`case_sheet.json` is a case-level specification used to make each design task
more concrete than a runnable `design_state.json`. It is read by
`engine.case_builder` and exposed to the outer-loop LLM as
`case_sheet_summary`.

The sheet does not directly score candidates yet. It documents the biological
contract that future objectives, masks, and screening filters should implement.

## Required Fields

- `schema_version`: use `ast_case_sheet_v1`.
- `case_name`: stable case id, for example `cd25_scfv`.
- `readiness`: runtime and biological-specificity status.
- `design_goal`: short formal goal plus allowed and frozen design regions.
- `reference_inputs`: current references and missing references needed for
  high-confidence design.
- `state_success_criteria`: state-by-state desired events and failure signals.
- `residue_level_constraints`: mutable, frozen, must-contact, avoid-contact,
  pharmacophore, or epitope constraints.
- `objective_thresholds`: provisional numeric thresholds used by current
  objectives or desired future filters.
- `provisional_assumptions`: assumptions that should not be treated as facts.
- `information_gaps`: concrete user inputs needed before full production runs.

## Design Rule

Treat `case_sheet.json` as the biological contract and `design_state.json` as
the executable runtime specification. When new structural or experimental
information becomes available, update the case sheet first, then translate it
into typed objectives, node masks, or residue-level constraints.

## Minimal Example

```json
{
  "schema_version": "ast_case_sheet_v1",
  "case_name": "example_case",
  "readiness": {
    "runtime": "ready",
    "biological_specificity": "provisional"
  },
  "design_goal": {
    "primary": "Design a protein with a specified binding or switching behavior.",
    "allowed_to_change": ["design_node"],
    "not_allowed_to_change": ["fixed_node"]
  },
  "reference_inputs": {
    "current_reference": {
      "name": "reference structure or sequence source",
      "available_in_case": true
    },
    "needed_for_high_confidence_design": []
  },
  "state_success_criteria": {},
  "residue_level_constraints": {},
  "objective_thresholds": {},
  "provisional_assumptions": [],
  "information_gaps": []
}
```
