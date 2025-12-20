# Bayleaf
A self-hosted web app that indexes local EPUB and PDF cookbooks, extracts recipes on a best-effort basis, and lets a user search by recipe name or ingredient and read books in a high-quality embedded reader.


## Next steps
- Add dev reload via docker-compose volume mounts
- Improve library UI and empty states
- Confirm external access setup (Cloudflare Tun)

## Recipe extraction tasks (next session)
- [x] Verify recipe extraction is actually running during /admin/reindex (capture logs + counts)
- [x] Add a debug report endpoint to sample extracted recipe titles for a specific book
- [x] Fix Crumbs & Doilies extractor if it yields zero recipes (validate class names + section boundaries)
- [x] Ensure recipe records are inserted with correct href + image href
- [x] Add a manual "re-extract recipes for this book" action in the UI

## Session notes (2024-12-20)
- Switched recipe extraction to prefer EPUB index anchors (from MVP engine) with fallback to section parsing
- Added image extraction based on recipe href to populate recipe cards
- Startup indexing now runs in the background to keep the app responsive
- Added async admin reindex flag (`/admin/reindex?async=true`) and health flags for indexing/reindexing
- Added `beautifulsoup4` dependency for index-based parsing

## What's next
- Tighten recipe title normalization + ignore rules for index-derived titles
- Improve image matching heuristics (skip icons, prefer figure/hero images)
- Expose indexing status in the UI and allow async reindex from the button
