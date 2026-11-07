# API Rate Limits

## Request limits
The Acme Cloud REST API allows 60 requests per minute on Starter, 600 requests per minute on Pro,
and 6,000 requests per minute on Enterprise. Limits are applied per API key.

## Exceeding limits
Requests over the limit receive HTTP status 429 Too Many Requests with a `Retry-After` header in seconds.
Clients should retry with exponential backoff.

## Payload limits
The maximum request body size is 10 MB. Bulk endpoints accept at most 1,000 items per call.

## API keys
Each project can have up to 25 active API keys. Keys can be scoped to read-only or read-write.
