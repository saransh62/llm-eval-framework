# Security Overview

## Encryption
All customer data is encrypted at rest with AES-256 and in transit with TLS 1.3.
Encryption keys are rotated every 90 days. Enterprise customers can bring their own keys (BYOK) through the Key Vault add-on.

## Access control
Multi-factor authentication (MFA) is required for all accounts with the Owner role.
Single sign-on (SSO) via SAML 2.0 is available on the Enterprise plan only.

## Vulnerability management
Acme Cloud runs an annual third-party penetration test and a public bug bounty program hosted on HackerOne.
Security issues can be reported to security@acmecloud.example.
