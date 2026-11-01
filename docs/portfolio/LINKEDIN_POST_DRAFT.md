# LinkedIn post draft

| | |
|---|---|
| **Status** | Draft. **Do not publish.** It describes the Tier C target, which does not exist yet |
| **Owner** | Aziz Ben Ammar |
| **Date** | 2026-11-01 |
| **Next step** | Revise after the [release gate](LINKEDIN_DEMO_PLAN.md#release-gate) passes |
| **Readers** | AI, LLM, agent, MLOps, and cloud platform engineers, and recruiters |

Every `{{PLACEHOLDER}}` is a future measurement or link. A placeholder must be
replaced with a real result or the sentence must be deleted. Never estimate one.

## Before publishing

Sentences in **Tier C only** blocks describe an agent that does not exist. Delete
them for Tier A or B, and use the replacement text below.

| Tier | Replace the agent paragraph with |
|---|---|
| **A** | "The copilot runs in a deterministic mode today, with no language model. Every answer is built from verified, server-written claims, or it refuses." |
| **B** | "A real Claude model selects which verified claims answer a question. It writes none of them. I measured it against a deterministic baseline before keeping it." |

If M1 fails the decision rule, say so: "I tested a real model against an acceptance
rule I set in advance. It didn't meet it, so the deterministic selector stayed."

## Draft

> **Most AI demos show what a model can say. I wanted to build one that shows what it refuses to say.**
>
> FootballAI analyses football video, then answers questions about that analysis, only from evidence it can verify.
>
> Why build it? Football video is messy, and it is easy to produce confident numbers that nobody can check. I wanted a project that forced me to engineer, evaluate, and secure an AI application, not only call a model.
>
> **How it works**
>
> Video goes through an asynchronous pipeline: FastAPI, a queue, and a worker running YOLOv8 and ByteTrack. Each run produces immutable, versioned, checksummed artifacts. The system runs locally, and the platform has been validated on Azure Container Apps with a synthetic analysis profile.
>
> **[Tier C only]** On top sits a read-only AI agent. It chooses which approved evidence tools to call, and it can never write a fact itself. The server writes every sentence from validated evidence and attaches citations.
>
> Ask how many passes player 7 made, and it refuses, because pass recognition doesn't exist in this system.
>
> **What I measured**
>
> • A real full-match video of about 96 minutes ran end to end on a laptop, in about 4.5 hours.
> • It also produced 8,099 raw track IDs for about 22 players. So the product says "temporary tracklet", never "player".
> • The copilot failed closed on 10 of 10 scripted invalid model replies.
> • {{AGENT_DATASET_SIZE}} evaluation cases, {{TOOL_SELECTION_ACCURACY}} tool-selection accuracy, {{CITATION_VALIDITY}} citation validity, {{FALSE_ANSWER_COUNT}} false answers.
> • {{P95_LATENCY}} p95 latency and about {{COST_PER_QUESTION}} per question, at list price.
>
> These numbers come from a dataset I wrote myself, so treat them as engineering evidence, not proof of production reliability.
>
> **What I learned**
>
> 1. A valid citation proves a field exists, not that the sentence beside it is true. Reviewing my first design, I reproduced a fully cited answer about distance and goals the system can't measure. So I took the writing away from the model.
> 2. Measure the baseline first. My deterministic system had three false answers, and knowing that made every later result meaningful.
> 3. Set the acceptance rule before the experiment, so the result can't choose its own bar.
>
> **What it is not**
>
> It isn't a production application, it has no user authentication, and the demo runs locally. It doesn't identify players, and its tracking is heavily fragmented. Those limits are shown in the product, on purpose.
>
> **Stack:** Python, FastAPI, React, PostgreSQL, Azure Blob Storage, Service Bus, Container Apps, Terraform, GitHub Actions with OIDC, YOLOv8, ByteTrack, and the Claude API.
>
> I'm a final-year Computer Engineering student looking for an AI Engineering or AI Platform internship in France for 2027.
>
> Video below. Code and evaluation reports: {{GITHUB_REPOSITORY_URL}}

## Architecture diagram caption

> FootballAI's verified architecture: an asynchronous computer-vision pipeline produces immutable evidence, and a read-only copilot answers only from it.

Use this only on the current, verified diagram. If the target diagram is used, add:
"Dashed components are planned."

## Suggested hashtags

`#AIEngineering` `#LLMOps` `#AgenticAI` `#MLOps` `#ComputerVision`

Drop `#AgenticAI` unless the agent is validated.

## Facts to re-check before publishing

| Sentence | Check |
|---|---|
| "about 96 minutes", "about 4.5 hours" | 95.65-minute video, 16,084 seconds. One run on Apple MPS, with the older `v1_compat/1.0.0` pipeline. The report calls it the 94-minute run |
| "8,099 raw track IDs" | One run. Same report |
| "10 of 10" | Scripted contract cases, no real model. Re-run before publishing |
| "three false answers" | Baseline v1. Update if the dataset changes |
| "validated on Azure Container Apps" | `demo_fast` at P4. Do not imply real YOLO or the copilot in the cloud |
| Lesson 1 | Recorded in the ADR-003 amendment: "Player 7 ran 10.5 km and scored twice." came back `supported` with 8 valid citations, in a reproduction against the original code during review |
| "Claude API" in the stack | Remove if no live model call has happened |
| Internship sentence | Keep or remove by personal preference |

## Claims that must not appear

- "Production-ready", "production-grade", or "enterprise" for the application.
- "Football tactical intelligence" or "autonomous coach".
- "Identifies players", or player statistics of any kind.
- "Real-time" analysis.
- "Accurate" tracking or detection. No such accuracy is measured.
- Any comparison with other products or models.
- "Agent" or "LLM" for anything the release did not actually run.
