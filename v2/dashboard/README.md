# FootballAi V2 dashboard

Local React 19, TypeScript, Vite, Recharts, Vitest, Testing Library and
Playwright dashboard for the V2 upload, progress, and results API.

From the repository root, the complete local demo is:

```bash
make v2-demo
```

For frontend-only development, start the API first and then run:

```bash
cd v2/dashboard
npm ci
npm run dev
```

Quality checks:

```bash
npm run typecheck
npm run lint
npm run test
npm run build
npm run test:e2e
```

The frontend calls only the configured local API. It has no analytics,
external data, cloud, image, font, or authentication dependency.
