# VERITAS — Multi-Agent Deepfake Detection System

## Overview

A full-stack web application that detects deepfakes using four specialised AI agents (Spatial, Temporal, Audio, and Semantic). Built as a pnpm monorepo with a React frontend and an Express API backend.

## Stack

- **Frontend** (`artifacts/deepfake-detector`): React 19 + Vite + Tailwind CSS v4 + shadcn/ui + Framer Motion + TanStack Query + Wouter
- **API Server** (`artifacts/api-server`): Express 5 + TypeScript + Pino logging
- **Database** (`lib/db`): PostgreSQL via Drizzle ORM — schema lives in `lib/db/src/schema/index.ts`
- **Shared libs**: `lib/api-zod` (Zod schemas), `lib/api-client-react` (typed API client), `lib/api-spec` (OpenAPI spec + Orval codegen)

## How to Run

Dependencies are managed with pnpm. Install once from the workspace root:

```bash
pnpm install
```

Workflows start automatically:
- **Frontend**: `artifacts/deepfake-detector: web` → serves at `/`
- **API Server**: `artifacts/api-server: API Server` → serves at `/api`

## Environment Variables

- `DATABASE_URL` — managed automatically by Replit (PostgreSQL)
- `SESSION_SECRET` — stored as a Replit Secret

## Database

Schema is defined in `lib/db/src/schema/index.ts` using Drizzle ORM. Currently no tables are defined — add models there and run migrations with:

```bash
cd lib/db && pnpm drizzle-kit push
```

## Project Structure

```
artifacts/
  deepfake-detector/   # React frontend (landing page + console)
  api-server/          # Express REST API
  mockup-sandbox/      # Design canvas / component preview server
lib/
  db/                  # Drizzle ORM + PostgreSQL
  api-zod/             # Shared Zod validation schemas
  api-client-react/    # Auto-generated typed API client
  api-spec/            # OpenAPI YAML + Orval codegen config
```

## User Preferences

- Keep the existing monorepo structure (pnpm workspace)
- Do not restructure or migrate the stack
