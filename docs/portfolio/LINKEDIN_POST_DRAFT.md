# LinkedIn post draft — Tier A only

**Draft, not authorized for publication.** Written for the generated local fixture. Confirm final footage, screenshots and CI before posting. The code link is [FootballAI on GitHub](https://github.com/azizbenammar7/FootballAi); use the eventual merged commit when publication is separately approved.

> I built FootballAI to make video-analysis claims inspectable.
>
> A FastAPI API accepts an analysis, a queue hands it to a worker, and each attempt registers versioned artifacts with SHA-256 integrity checks. A React dashboard shows the run, its provenance and its limitations.
>
> The attached **local demo uses a generated synthetic fixture**. Its `demo_fast` outputs are deterministic placeholders, not YOLO inference or football match measurements. Separately, I validated the YOLOv8m and ByteTrack path locally on private footage; that footage is excluded from this public demo.
>
> On top of the run evidence, a read-only copilot builds answers from server-written claims and citations. The demo's agent mode uses a **fixed offline script**, not a language model, to choose among five approved evidence tools. Its trace shows the tool that actually ran. Ask for a statistic the system does not measure, such as player 7's successful passes, and it abstains.
>
> The engineering work I wanted to show: asynchronous execution, clean storage and queue boundaries, immutable evidence, explicit model modes, server-side grounding, and tests for unsupported questions and trace behavior. The real Anthropic agent path exists behind a local gate, but its live quality has not been measured. Independent review of the v3 evaluation labels is also pending (0/142).
>
> This is a portfolio demonstration of a bounded local system. It does not verify player identities, possession, pass recognition, tactical understanding or tracking accuracy.
>
> Code, architecture and evaluation materials: https://github.com/azizbenammar7/FootballAi
>
> #AIEngineering #ComputerVision #MLOps #FastAPI #React

Do not add a model performance number, Azure availability claim, public try-it link or rights-cleared real-footage claim until separately verified. The final video should keep `LOCAL DEMO · SYNTHETIC FIXTURE` visible throughout UI footage.
